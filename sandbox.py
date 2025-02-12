#------------------------------------------------------------------------------------
# Developed by CSamuel Malkasian
#------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized. Please Fork Instead.
#------------------------------------------------------------------------------------

CURRENT_VERSION = "1.0"
CHANGES = """

"""

#------------------------------------NOTES-------------------------------------------
# Used to test code and run misc functions.


#------------------------------------IMPORTS-----------------------------------------
import pandas as pd
import requests


#-------------------------------------MAIN-------------------------------------------



# Test GET request
response = requests.get("http://127.0.0.1:8000/")
print("GET Response:", response.json())

# Test POST request with sample input
data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

response = requests.post("http://127.0.0.1:8000/predict", json=data)
print("POST Response:", response.json())
