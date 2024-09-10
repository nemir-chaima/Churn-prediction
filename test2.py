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
    np.testing.assert_array_equal(prediction, expected_output)


def test_prediction_case_2(model):

        input_data = pd.DataFrame([[51, 5.47, 10]])
        prediction = model.predict(input_data)
        
        expected_output = [0]  
        np.testing.assert_array_equal(prediction, expected_output)

def test_prediction_case_3(model):

    input_data = pd.DataFrame([[45, 5.46, 4]])
    prediction = model.predict(input_data)
        
    expected_output = [0]  
    np.testing.assert_array_equal(prediction, expected_output)

    
def test_process():
     model = load_model()
     print('model load fait')
     test_prediction_case_1(model)
     print('test 1 : fait')
     test_prediction_case_2(model)
     print('test 2 fait')
     test_prediction_case_2(model)
     print('test 3 fait')

test_process()
     
