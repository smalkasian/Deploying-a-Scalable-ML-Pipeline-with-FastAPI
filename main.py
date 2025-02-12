#------------------------------------------------------------------------------------
# Developed by Samuel Malkasian
#------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized. Please Fork Instead.
#------------------------------------------------------------------------------------

CURRENT_VERSION = "1.0"
CHANGES = """
"""

#------------------------------------IMPORTS-----------------------------------------
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.data import apply_label, process_data
from ml.model import inference, load_model

# Errors and Logging
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#--------------------------------------INITIALIZE--------------------------------------
# Initialize FastAPI application
app = FastAPI()

base_dir = os.path.dirname(os.path.abspath(__file__))
encoder_path = os.path.join(base_dir, "model/encoder.pkl") 
model_path = os.path.join(base_dir, "model/model.pkl")

# Load model and encoder
encoder = load_model(encoder_path)
model = load_model(model_path)

#----------------------------------REQUEST MODEL-------------------------------- #
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

# Welcome endpoint
@app.get("/")
async def get_root():
    """ Say hello! """
    return {"message": "FastAPI Root. Welcome!"}

# Model inference endpoint
@app.post("/predict")
async def post_inference(data: Data):
    """ Runs model inference on input data """
    
    # Convert Pydantic model to dictionary
    data_dict = data.dict()
    
    # Convert dict to Pandas DataFrame
    data_df = pd.DataFrame([data_dict])

    data_df.rename(columns={
        "marital_status": "marital-status",
        "native_country": "native-country"
        }, inplace=True)

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

    # Preprocess input data using the loaded encoder
    data_processed, _, _, _ = process_data(
        data_df, 
        categorical_features=cat_features, 
        label=None, 
        training=False,
        encoder=encoder
    )
    
    # Run inference
    prediction = inference(model, data_processed)
    
    # Convert prediction to label
    result = apply_label(prediction)
    
    return {"result": result}