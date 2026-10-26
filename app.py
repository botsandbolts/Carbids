"""CarBids Studio: streamlit run app.py"""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from carbids.data import prepare,load_cohort,TARGET
from carbids.inference import PricePipeline
from carbids.explanations import render_explanation

ROOT=Path(__file__).parent
st.set_page_config(page_title='CarBids Studio',page_icon='🏁',layout='wide')
st.markdown('''<style>.stApp{background:linear-gradient(135deg,#f6f8f3,#f0f5f4)}h1,h2,h3{color:#173c35}div[data-testid="stMetric"]{background:white;border:1px solid #d7e4dd;border-radius:12px;padding:18px}div[data-testid="stSidebar"]{background:#e5ede6}.stButton>button{border-radius:9px}code{font-size:.85em}</style>''',unsafe_allow_html=True)
st.title('🏁 CarBids + Pricing Studio')
st.caption('The auction is over. The number-crunching has just begun.')
st.info('Target: publicly reported completed-sale prices in USD. A published “sold” result is not independent proof of payment settlement. Buyer fees, taxes and shipping are outside the estimate.')

@st.cache_resource
def pipeline(path):return PricePipeline(path)
@st.cache_data
def report(path):return json.loads(Path(path).read_text())
@st.cache_data
def cohort(path,stamp):return load_cohort(path)

def choices():
    found=[]
    for folder in ['models','artifacts']:
        for p in (ROOT/folder).glob('**/manifest.json'):
            m=json.loads(p.read_text())
            if m.get('status')=='ready' and m.get('target_kind')==TARGET:found.append(str(p.parent.relative_to(ROOT)))
    return sorted(found)

with st.sidebar:
    st.header('Your auction desk')
    st.caption('This selector controls the historical auction workspaces. Pricing + CarBids uses its own bundled structural and asking models.')
    available=choices()
    location=st.selectbox('Trained model',available) if available else None
    st.caption('Baseline = structured vehicle fields. Enhanced = MiniLM description signal + LightGBM. Asking-price models are refused at load time.')
    st.link_button('Open carbids repository','https://github.com/botsandbolts/carbids')
    st.link_button('Kaggle notebooks','https://www.kaggle.com/code')
    st.write('**Status**')
    if location:
        manifest=report(str(ROOT/location/'manifest.json'));st.success('Trained model available')
        st.write('Rows used:',f"{manifest['trained_rows']:,}")
        st.write('Sources:',', '.join(manifest['sources']))
    else:manifest={};st.warning('No completed model yet. Import data and train; the app will not invent a price.')

compare,value,explain,data,performance,training,bridge,census,learn=st.tabs(['Pricing + CarBids','Historical auction model','Explainable ML','Auction explorer','Model report card','Data & training','Bridge to Pricing','Census context','Learn · two gears'])
with compare:
    from marketbridge.ui import render_comparison
    render_comparison(ROOT)
with value:
    st.subheader('What might a comparable auction sale achieve?')
    st.write('Enter pre-sale facts. “My neighbour thinks it is worth a fortune” is not currently a supported sensor.')
    enhanced=manifest.get('pipeline')=='fine-tuned-minilm-l6+lightgbm'
    with st.form('car'):
        a,b,c=st.columns(3)
        year=a.number_input('Model year',1886,2027,2014)
        make=b.text_input('Make','Toyota');model=c.text_input('Model / trim','Camry')
        a,b,c=st.columns(3)
        mileage=a.number_input('Mileage · miles',0,2000000,65000,step=1000)
        condition=b.text_input('Auction condition grade / description','4.0',help='Use the source’s actual grade. Different auction houses use different grading systems.')
        source=c.selectbox('Auction data source',manifest.get('sources',['wholesale_2015']))
        a,b,c=st.columns(3)
        transmission=a.selectbox('Transmission',['automatic','manual','unknown'])
        body=b.selectbox('Body type',['sedan','suv','coupe','hatchback','wagon','convertible','truck','van','unknown'])
        state=c.text_input('US state abbreviation','ca',max_chars=2)
        with st.expander('More than four wheels and a hopeful expression'):
            a,b,c=st.columns(3);fuel=a.selectbox('Fuel',['unknown','gas','diesel','hybrid','electric']);drive=b.selectbox('Drive',['unknown','fwd','rwd','4wd','awd']);color=c.text_input('Paint colour','white')
            a,b=st.columns(2);cylinders=a.text_input('Cylinders','unknown');title=b.selectbox('Title',['unknown','clean','salvage','rebuilt'])
        from datetime import date
        as_of=st.date_input('Valuation as-of date',value=date.today(),help='For a Census-enabled model, use only context released before this date. Historical model coverage still applies.')
        notes=st.text_area('Pre-sale seller description',max_chars=8000,height=160,disabled=not enhanced,
            placeholder='History, equipment, modifications and known flaws. No sold-price/result line, bid history or post-sale comments.')
        if not enhanced:st.caption('This trained tabular model has no description branch. An excellent paragraph cannot bribe it.')
        submitted=st.form_submit_button('Ask the trees 🏁',disabled=not location)
    if submitted:
        fields=dict(as_of=as_of.isoformat(),year=year,manufacturer=make,model=model,odometer=mileage,condition=condition,source=source,
            transmission=transmission,type=body,state=state,fuel=fuel,drive=drive,paint_color=color,cylinders=cylinders,title_status=title)
        try:
            if enhanced and len(notes.strip())<80:raise ValueError('Enhanced mode needs a meaningful pre-sale description (at least 80 characters).')
            p=pipeline(str(ROOT/location)).predict(notes,fields,explain=True)
            st.session_state['valuation']=(location,p,fields,notes)
        except Exception as e:st.error(str(e))
    saved=st.session_state.get('valuation')
    if saved and saved[0]==location:
        p=saved[1];a,b,c=st.columns(3)
        a.metric('Estimated reported sale price',f"${p['price']:,.0f}")
        b.metric('Indicative lower bound',f"${p['range'][0]:,.0f}");c.metric('Indicative upper bound',f"${p['range'][1]:,.0f}")
        st.caption('Indicative 80% residual band, calibrated on validation. Not an individual probability guarantee. The final deployment refit has no independent holdout.')
        st.download_button('Download this valuation',json.dumps({'target_kind':TARGET,'settlement_verified':False,'model':location,'inputs':saved[2],'prediction':p},indent=2),file_name='carbids-valuation.json',mime='application/json')
        st.success('Your price has an explanation. Open Explainable ML for the committee minutes.')
        if saved[2]['year']>manifest.get('model_year_range',[1886,2027])[-1]:st.warning('This model year is newer than the model’s training coverage.')
        from marketbridge.ui import render_asking_companion
        render_asking_companion(ROOT,saved[2])
        st.caption('Training sale dates: '+' to '.join(manifest.get('sale_dates',['see model report'])))
    with st.expander('Batch valuation · CSV in, sale estimates out'):
        batch=st.file_uploader('Pre-sale structured fields CSV',type=['csv'],key='batch')
        st.caption('Columns: year, manufacturer, model, odometer, condition, source, transmission, type, state. Enhanced models additionally require text. Maximum 2,000 rows per batch.')
        if batch is not None and location and st.button('Value this batch'):
            try:
                df=pd.read_csv(batch)
                if len(df)>2000:raise ValueError('Batch limit is 2,000 rows')
                outputs=[pipeline(str(ROOT/location)).predict(str(row.get('text','')),row) for row in df.to_dict('records')]
                df['predicted_sale_price_usd']=[p['price'] for p in outputs]
                df['band_low']=[p['range'][0] for p in outputs];df['band_high']=[p['range'][1] for p in outputs]
                st.dataframe(df.head(30));st.download_button('Download batch estimates',df.to_csv(index=False),'carbids-batch.csv','text/csv')
            except Exception as e:st.error(str(e))
with explain:
    saved=st.session_state.get('valuation')
    if saved and saved[0]==location:
        st.caption(f"Explaining submitted {saved[2]['year']} {saved[2]['manufacturer']} {saved[2]['model']}. Form edits require a new valuation.")
        render_explanation(saved[1],saved[2],saved[3],pipeline(str(ROOT/location)))
    else:st.info('Value a car first. Even a very talkative model needs a prediction to explain.')
with data:
    st.subheader('The auction room, without the shouting')
    file=ROOT/'data/cohort.jsonl'
    if file.exists():
        d=cohort(str(file),file.stat().st_mtime_ns)
        a,b,c=st.columns(3);a.metric('Eligible distinct vehicles',f'{len(d):,}');b.metric('Median recorded sale',f'${d.price.median():,.0f}');c.metric('Text-rich records',f'{d.text.fillna("").str.len().ge(80).sum():,}')
        selected=st.selectbox('Explore make',['All']+sorted(d.manufacturer.dropna().astype(str).unique()))
        subset=d if selected=='All' else d[d.manufacturer==selected]
        summary=subset.groupby('model').price.agg(['count','median']).sort_values('count',ascending=False).head(20)
        st.dataframe(summary,width='stretch');st.bar_chart(summary['median'],horizontal=True)
        st.dataframe(subset[[c for c in ['date','manufacturer','model','year','odometer','condition','price','source','phase'] if c in d]].tail(200),width='stretch',hide_index=True)
        if file.with_suffix('.meta.json').exists():st.json(report(str(file.with_suffix('.meta.json'))),expanded=False)
    else:st.info('No real auction cohort imported. The example export is a schema fixture, not market evidence.')
with performance:
    st.subheader('Report card. The model cannot mark its own homework.')
    if location and (ROOT/location/'evaluation.json').exists():
        e=report(str(ROOT/location/'evaluation.json'));scores=e.get('test',e.get('stacked',{}))
        a,b,c,d=st.columns(4);a.metric('Test MAE',f"${scores['mae_usd']:,.0f}");b.metric('Test RMSE',f"${scores['rmse_usd']:,.0f}");c.metric('Median APE',f"{scores['median_ape_pct']:.1f}%");d.metric('R²',f"{scores['r2']:.3f}")
        st.write('Measured band coverage:',f"{e['band80_test_coverage']:.1%}")
        st.write(e.get('note'));st.json(e,expanded=False)
        st.download_button('Download evaluation',json.dumps(e,indent=2),'evaluation.json','application/json')
    else:st.info('No completed evaluation. Synthetic tests do not count as pricing performance.')
with training:
    st.subheader('Real sales in. Wishful thinking out.')
    st.write('The importer accepts explicit sold statuses, USD sale amounts and source evidence. It drops invalid VINs, retains one sale per vehicle, removes duplicate rich descriptions and splits whole dates chronologically. Unsold bids, reserves, listing asks, MMR estimates and currencies other than USD are rejected as targets.')
    upload=st.file_uploader('Import a source-audited auction export',type=['csv','jsonl'],key='import')
    st.caption('Use data/example_auction_export.csv as a schema guide. Rich text requires text_scope=pre_sale and text_as_of no later than sale_date. More than 30 distinct eligible vehicles and multiple sale dates are required.')
    if upload is not None and st.button('Validate and prepare export'):
        try:
            content=upload.getvalue();token=hashlib.sha256(content).hexdigest()[:12]
            dest=ROOT/'data/imports'/token;dest.mkdir(parents=True,exist_ok=True)
            original=dest/('export'+Path(upload.name).suffix);original.write_bytes(content)
            result=prepare(original,dest/'cohort.jsonl','auction_export')
            st.json(result);st.success(f'Prepared isolated cohort: data/imports/{token}/cohort.jsonl')
        except Exception as e:st.error(str(e))
    st.code('python -m carbids.data --download\npython -m carbids.baseline --threads 4\npython -m streamlit run app.py',language='bash')
    st.write('**Rich-description GPU branch:** use train_kaggle.ipynb. Attach an audited rich-text cohort as Kaggle input; the wholesale dataset has no descriptions, so its enhanced run is intentionally refused. Training is never launched secretly from this GUI.')
    st.code('python -m carbids.train --data data/rich-cohort.jsonl --output artifacts/enhanced --device cuda --folds 3 --epochs 2',language='bash')
    st.write('Optimizer checkpoints, completed-fold caches, provenance fingerprints and evaluations are saved. Resume with identical source, data, output path and settings. Do not apply post-sale bid statistics as pre-sale inputs.')
with bridge:
    st.subheader('Two markets. One bridge. No invented discount.')
    st.warning('Auction sale estimate is available. Calibrated retail-sale estimate is unavailable: matched target-domain retail outcomes have not been collected or fitted.')
    st.write('The safe first step is to supply Pricing with a frozen, vehicle-disjoint auction-price feature. Do not train on the same VIN and then call its auxiliary prediction out-of-fold. The export index conservatively blocks possible overlaps.')
    transfer=st.file_uploader('Pricing vehicle CSV with VIN and pre-sale structured fields',type=['csv'],key='bridge')
    if transfer is not None and location and st.button('Export independent auction signals'):
        try:
            from carbids.bridge import auxiliary_features
            frame=pd.read_csv(transfer)
            if len(frame)>2000:raise ValueError('Please export at most 2,000 rows at a time')
            signals=auxiliary_features(frame,ROOT/location)
            st.dataframe(signals.head(30));st.download_button('Download auction auxiliary features',signals.to_csv(index=False),'auction_features.csv','text/csv')
        except Exception as e:st.error(str(e))
    st.caption('The asking-to-auction gap is excluded: observed asking price would leak the target in an asking-price model. Features remain explicitly auction-domain signals, not retail conversions.')
    with st.expander('Advanced statistical design · partial pooling, transport and conformal intervals'):
        document=ROOT/'STATISTICAL_BRIDGE.md'
        if document.exists():st.markdown(document.read_text())
        else:st.info('See STATISTICAL_BRIDGE.md in the repository for the full design and assumptions.')
with census:
    from carbids.census import RELEASES,join_asof
    st.subheader('Census context · the neighbourhood, not the neighbour')
    st.write('American Community Survey five-year estimate vintages 2016–2022 supply regional income, population, unemployment, vehicle access, homeownership and working-from-home context. These are area averages, not attributes inferred about a buyer.')
    st.warning('The shipped wholesale model uses 2015 sales and no Census features. Vintages 2016–2022 were released later, so attaching them to that benchmark would introduce future information.')
    st.dataframe(pd.DataFrame([{'ACS vintage':y,'Five-year period':f'{y-4}–{y}','Released':d} for y,d in RELEASES.items()]),hide_index=True,width='stretch')
    st.caption('Joins conservatively wait until the day after release. The delayed 2020 vintage is unavailable until March 2022. A vintage year is not a publication year.')
    panel=ROOT/'data/census/acs5-2016-2022-state.csv'
    if panel.exists():
        context=pd.read_csv(panel,dtype={'acs_geo_id':str})
        state=st.text_input('Context state','ca',max_chars=2,key='census-state')
        from datetime import date
        date_value=st.date_input('Historical context as-of',value=date.today(),key='census-asof')
        selected=join_asof(pd.DataFrame([{'state':state,'date':date_value.isoformat()}]),context)
        st.dataframe(selected,width='stretch',hide_index=True)
    else:
        st.info('The download path is ready, but regional Census values are not yet cached. The API requires a Census key for this request. Set CENSUS_API_KEY privately; never paste it into this GUI, a notebook output, or source control.')
    st.code('python -m carbids.census fetch --geography state\npython -m carbids.census enrich --input data/rich-cohort.jsonl --panel data/census/acs5-2016-2022-state.csv --output data/rich-census.jsonl',language='bash')
    st.write('County joins require an explicit valid county FIPS and matching state; a postal ZIP is not a county FIPS. Auction location is not necessarily the buyer’s market. Future validation should compare with/without Census on the same time/VIN-disjoint target-sale test set.')
    st.caption('Margins of error are preserved. Adjacent five-year estimates overlap and are not independent annual measurements. Current API values may include revisions; strict historical replay requires archived releases or revision timestamps. No Census accuracy uplift is claimed before such evaluation.')
with learn:
    ordinary,technical=st.tabs(['Explain it over coffee','Explain it over tensor shapes'])
    with ordinary:
        st.markdown('''### Meet the pricing committee
**LightGBM:** many small decision trees, each correcting earlier mistakes. No single tree gets to value every Porsche at “a lot”.

**MiniLM:** reads pre-sale descriptions and learns a price signal. “Recent engine rebuild” and “engine recently left the vehicle” should not mean the same thing.

**Out-of-fold:** each training car gets a text estimate from a model that did not see its price. No peeking at the answer sheet.

**Huber loss:** takes ordinary errors seriously and stops huge errors from completely dominating training.

**SHAP:** divides credit for one prediction among its features. It explains what the model did, not what the universe must do.

**Asking vs sold:** an asking price is a seller’s opening monologue. A reported sale is the platform’s published outcome. Neither proves that the money cleared.

**Auction domain:** wholesale fleet cars, enthusiast auctions and salvage lots are different markets. Check the model’s sources and dates before applying it elsewhere.''')
    with technical:
        st.latex(r'\hat y=\exp(\mu+\sigma f_\theta(x))-1')
        st.write('For the log1p target, predictions are inverse-standardized and clipped before expm1. Raw-dollar targets are also compared. Validation USD MAE selects Huber alpha and iteration count; test MAE/RMSE/RMSLE/median APE/R² are measured before a full-data deployment refit.')
        st.write('MiniLM-L6: 22.7M parameters, 384-dimensional hidden states, 256-token windows across up to 8,000 cleaned characters, masked mean pooling, normalized chunk/document vectors and a trainable regression head. OOF dollar predictions supply one dynamic feature to LightGBM. No make/model embeddings, TF-IDF, ridge or Extra Trees.')
        st.write('Native TreeSHAP returns n_features+1 contributions, including expected value. After rescaling, reference plus contributions reconstructs the raw fitted target. Log contributions are multiplicative on price+1; gain importance is global, not local. Text-price SHAP is not token attribution.')
        st.write('Isolation Forest auto uses only training OOF text prices; validation/test are never filtered. Baseline mode has no text-price feature and no Isolation Forest. VIN deduplication and whole-date splits prevent obvious relisting and time-boundary leakage; text attestation does not independently establish the truth of source claims.')
        st.link_button('LightGBM contribution documentation','https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html#lightgbm.Booster.predict')
st.divider();st.caption('CarBids Studio · Reported sale prices, auditable sources, explanations with a sense of humour. An estimate is not an appraisal, a warranty or a winning bidding strategy.')
