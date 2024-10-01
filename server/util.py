import json
import pickle
import numpy as np

__locations = None
__data_colums = None
__model = None 

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_colums.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_colums))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)
  

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts....")
    global __data_colums
    global __locations
    global __model

    with open("./artifacts/columns.json","r") as f:
        __data_colums = json.load(f)['data_columns']
        __locations = __data_colums[3:]

    with open("./artifacts/banglore_home_prices_model.pickle",'rb') as f:
        __model = pickle.load(f)

if __name__ =='__main__':
    load_saved_artifacts()