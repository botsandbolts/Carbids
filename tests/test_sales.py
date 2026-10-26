from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from carbids.data import normalize,assign_phases,load_cohort,TARGET

def fixture():
    return {'source':'bringatrailer','source_url':'https://bringatrailer.com/listing/test-car/',
        'auction_id':'lot-1','status':'sold','target_kind':TARGET,'currency':'USD','sale_price':15000,
        'price_evidence':'Sold for USD $15,000','sale_date':'2025-01-01T20:00:00Z','year':2014,'mileage_miles':60000,
        'make':'Toyota','model':'Camry','vin':'1HGCM82633A004352',
        'description_pre_sale':'Maintained sedan with service history, recent tires and a clean interior. A few exterior scratches are documented.',
        'text_scope':'pre_sale','text_as_of':'2024-12-28T12:00:00Z'}

def test_accepts_explicit_sale_and_pre_sale_text():
    d,reasons=normalize(pd.DataFrame([fixture()]),'auction_export')
    assert len(d)==1 and not reasons
    assert d.price.iloc[0]==15000 and d.target_kind.iloc[0]==TARGET
    assert 'sale_price' not in d.columns and 'price_evidence' in d

@pytest.mark.parametrize('change',[{'status':'reserve_not_met'},{'status':'live'},{'target_kind':'asking_price'},
 {'currency':'EUR'},{'price_evidence':''},{'price_evidence':'Bid to USD $15,000'},{'price_evidence':'Sold for USD $20,000'},{'source_url':'https://unverified.example/lot'},
 {'text_as_of':'2025-01-02T00:00:00Z'},{'text_scope':'post_sale'},{'vin':'unknown'},{'sale_price':0}])
def test_rejects_unreliable_targets_and_leaking_text(change):
    d,reasons=normalize(pd.DataFrame([{**fixture(),**change}]),'auction_export')
    assert d.empty and sum(reasons.values())==1

def test_price_and_comments_do_not_become_features():
    from carbids.tabular import features,fit_schema,CATEGORICAL
    row={c:'test' for c in CATEGORICAL};row.update(year=2014,odometer=30000,price=9000,mmr=9000,current_bid=9000,bid_count=10,comments='Sold for 9000')
    d=pd.DataFrame([row]);x=features(d,fit_schema(d))
    assert not {'price','mmr','current_bid','bid_count','comments','price_evidence'}&set(x)

def test_money_is_redacted():
    from carbids.common import clean
    assert '15000' not in clean('Maintained. Sold for $15000. Asking price: 16000.')

def test_whole_dates_are_not_split():
    d=pd.DataFrame({'row_id':np.arange(90).astype(str),'date':np.repeat(pd.date_range('2024-01-01',periods=30,tz='UTC').astype(str),3)})
    out=assign_phases(d)
    assert out.groupby('date').phase.nunique().max()==1
    assert out[out.phase=='train'].date.max()<out[out.phase=='validation'].date.min()

def test_missing_data_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError,match='carbids.data'):load_cohort(tmp_path/'missing.jsonl')

def test_refuses_asking_price_model_bundle(tmp_path):
    from carbids.inference import PricePipeline
    (tmp_path/'manifest.json').write_text(json.dumps({'status':'ready','pipeline':'tabular-lightgbm','target_kind':'asking_price','currency':'USD'}))
    with pytest.raises(ValueError,match='asking-price'):PricePipeline(tmp_path)


def test_weighted_conformal_handles_test_point_mass():
    from carbids.bridge import weighted_conformal_radius,effective_sample_size
    assert weighted_conformal_radius([1,2,3,4],[1,1,1,1],1,alpha=.2)==4
    assert np.isinf(weighted_conformal_radius([1,2,3,4],[1,1,1,1],100,alpha=.2))
    assert effective_sample_size([1,1,1,1])==4
    assert effective_sample_size([100,1,1,1])<2


def test_vehicle_overlap_index_has_no_false_negatives():
    from carbids.bridge import positions,seen_vehicle
    index={'bits':1024,'k':12};bitmap=bytearray(128);vin='1HGCM82633A004352'
    for p in positions('vin:'+vin,index['bits'],index['k']):bitmap[p//8]|=1<<(p%8)
    assert seen_vehicle(vin,index,bitmap)


def test_shipped_model_has_sale_target_and_exact_shap_sum():
    from carbids.inference import PricePipeline
    root=Path(__file__).resolve().parents[1]
    model=PricePipeline(root/'models/tabular')
    result=model.predict('',{'year':2014,'manufacturer':'toyota','model':'camry','odometer':65000,'condition':'4.0','source':'wholesale_2015'},explain=True)
    assert result['price']>0 and result['explanation']['additivity_error']<1e-7
    assert model.manifest['target_kind']==TARGET
    assert model.manifest['training_candidates']==model.manifest['trained_rows']


def test_overlap_export_refuses_vehicles_seen_in_training():
    from carbids.bridge import auxiliary_features
    root=Path(__file__).resolve().parents[1]
    # First indexed VIN is a real training vehicle. Never publish an in-sample transfer feature.
    data=root/'data/cohort.jsonl'
    if not data.exists():pytest.skip('Raw cohort is intentionally not committed')
    with data.open() as f:row=json.loads(next(f))
    fields={'vin':row['vehicle_group'].removeprefix('vin:'),'year':row['year'],'odometer':row['odometer']}
    with pytest.raises(ValueError,match='overlap'):auxiliary_features(pd.DataFrame([fields]),root/'models/tabular')
