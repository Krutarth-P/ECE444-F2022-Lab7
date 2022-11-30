import pytest
from project.application import application, Model
from time import time


#test predict funtion
@pytest.mark.parametrize("input", [{"news": 'donlad trump was a president', "expected": 0},#real
                                   {"news": 'joe biden is the president', "expected": 0},#real
                                   {"news": 'Emperor of Mars has declared war on Earth', "expected": 1},#fake
                                   {"news": 'elephants can fly', "expected": 1}])#fake
def test_detector(input):
    tester = application.test_client()
    pred = Model.predict(input["news"])
    assert pred == input["expected"]


#test the api on localhost
@pytest.mark.parametrize("input", [{"news": 'donlad trump was a president', "expected": 0},#real
                                   {"news": 'joe biden is the president', "expected": 0},#real
                                   {"news": 'Emperor of Mars has declared war on Earth', "expected": 1},#fake
                                   {"news": 'elephants can fly', "expected": 1}])#fake
def test_local(input):
    times=0
    pred=0
    for i in range(100):
        tester = application.test_client()
        start = time()#start clock
        resp = tester.get(f"/predict?news={input['news']}")
        times += ((time() - start)*1000)#end clock
        pred = resp.get_json()["pred"]
    print()
    print("Test input news: ",input['news']) 
    print("Expected response: ", input['expected']), 
    print("Actual response: ", pred)
    print("Average latency over 100  calls(ms)",times/100)
