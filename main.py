#------------------------------------------------------------------------------------
# Developed by Samuel Malkasian
#------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized. Please Fork Instead.
#------------------------------------------------------------------------------------

CURRENT_VERSION = "1.0"
CHANGES = """

"""

#------------------------------------NOTES-----------------------------------------

#------------------------------------IMPORTS-----------------------------------------

import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.data import apply_label, process_data
from ml.model import inference, load_model

# Errors and Logging
import logging
import traceback
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#--------------------------------------VAR DECS--------------------------------------
# Initialize FastAPI application
app =  FastAPI()

base_dir = os.path.dirname(os.path.abspath(__file__))

encoder_path = None # TODO: enter the path for the saved encoder 
data_path = "Deploying-a-Scalable-ML-Pipeline-with-FastAPI/data/census.csv"

encoder = load_model(encoder_path)
model = load_model(data_path)

#-------------------------------CLASSES/DB MODELS-------------------------------- # DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

#------------------------------------API ROUTES-----------------------------------

# TODO: create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    """ Say hello!"""
    return {"FatAPI Root. Welcome!"}
    pass


# TODO: create a POST on a different path that does model inference
@app.post("/data/")
async def post_inference(data: Data):
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens and Python does not allow those as variable names.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    data_processed, _, _, _ = process_data(
        # your code here
        # use data as data input
        # use training = False
        # do not need to pass lb as input
    )
    _inference = None # your code here to predict the result using data_processed
    return {"result": apply_label(_inference)}

# Example POST endpoint for model inference
@app.post("/data/")
async def post_inference(data: Data):
    data_dict = data.dict()
    
    # Clean up the dictionary to convert it into a pandas DataFrame
    data_cleaned = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data_df = pd.DataFrame.from_dict(data_cleaned)

    # Define categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Preprocess the data
    data_processed, _, _, _ = process_data(
        data_df, 
        categorical_features=cat_features, 
        label=None, 
        training=False
    )
    
    # Run inference on processed data
    prediction = inference(model, data_processed)
    
    # Apply the label
    result = apply_label(prediction)
    
    return {"result": result}