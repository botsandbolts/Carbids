from fastapi.testclient import TestClient
from carbids.api import app


def test_sale_price_api_and_domain_validation():
    client=TestClient(app)
    assert client.get('/health').json()['status']=='ready'
    row={'year':2014,'manufacturer':'toyota','model':'camry','odometer':65000,'condition':'4.0','source':'wholesale_2015'}
    response=client.post('/predict',json=row)
    assert response.status_code==200
    result=response.json()
    assert result['target_kind']=='platform_reported_sale_price' and result['settlement_verified'] is False
    assert result['explanation']['additivity_error']<1e-7
    assert client.post('/predict',json={**row,'source':'unseen_retail'}).status_code==422
    assert client.post('/predict',json={**row,'odometer':-1}).status_code==422
