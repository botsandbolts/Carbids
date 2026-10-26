import numpy as np
import pandas as pd
from carbids.census import parse_payload,join_asof,VARIABLES

def panel_fixture():
    header=['NAME']+[v+s for v in VARIABLES for s in ['E','M']]+['state']
    row=['California']+['100' if s=='E' else '10' for v in VARIABLES for s in ['E','M']]+['06']
    return parse_payload([header,row],2020,'state')

def test_2020_vintage_cannot_leak_into_2021_sales():
    panel=panel_fixture()
    frame=pd.DataFrame({'state':['ca']*4,'date':['2015-06-01','2021-05-01','2022-03-17T20:00:00Z','2022-03-19T00:00:00Z']})
    enriched=join_asof(frame,panel)
    assert enriched.acs_context_available.tolist()==[0,0,0,1]
    assert enriched.acs_household_income.iloc[:3].isna().all()
    assert enriched.acs_vintage.iloc[3]==2020

def test_unknown_state_not_filled_with_national_or_future_data():
    result=join_asof(pd.DataFrame({'state':['zz'],'date':['2026-01-01']}),panel_fixture())
    assert result.acs_context_available.iloc[0]==0

def test_negative_census_sentinels_become_missing():
    header=['NAME']+[v+s for v in VARIABLES for s in ['E','M']]+['state']
    row=['California']+['-666666666']*(len(header)-2)+['06']
    result=parse_payload([header,row],2022,'state')
    assert result.acs_household_income.isna().all()
    assert result.acs_unemployment_rate.isna().all()
