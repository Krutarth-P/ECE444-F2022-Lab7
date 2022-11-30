import pytest
import requests, json
from time import time

API="http://fakenews-env.eba-fsscipzj.us-east-2.elasticbeanstalk.com/"

@pytest.mark.parametrize("input", [{"news": 'donlad trump was a president', "expected": 0},#real
                                   {"news": 'joe biden is the president', "expected": 0},#real
                                   {"news": 'Emperor of Mars has declared war on Earth', "expected": 1},#fake
                                   {"news": 'elephants can fly', "expected": 1}])#fake
def test_api_AWS(input):
    times=0
    pred=0
    for i in range(100):
        start = time()#start clock
        resp = requests.get(f"{API}predict?news={input['news']}")
        times += ((time() - start)*1000)#end clock
        pred = resp.json()["pred"]
    print()
    print("Test input news: ",input['news']) 
    print("Expected response: ", input['expected']), 
    print("Actual response: ", pred)
    print("Average latency over 100  calls(ms)",times/100)