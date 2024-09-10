import unittest
import pickle
import pandas as pd
import numpy as np


def load_model():
    with open('random_forest_t.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def test_prediction_case_1(model):
        
    input_data = pd.DataFrame([[41, 8, 9]])
    prediction = model.predict(input_data) 
    expected_output = [1]  
    #np.testing.assert_array_equal(prediction, expected_output)
    eq= 'False'
    if expected_output == prediction:
        eq= 'True'
    return eq
    


def test_process():
     model = load_model()
     print('model load fait')
     print(test_prediction_case_1(model))
     print('test 1 : fait')

test_process()
     
