import requests

# Send a GET request to the FastAPI server
r = requests.get("http://127.0.0.1:8000/")

# Print the status code
print("GET Status Code:", r.status_code)

# Print the welcome message from the GET response
print("GET Response:", r.json())

# Define the input data for the model
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST request to the FastAPI server
r = requests.post("http://127.0.0.1:8000/predict", json=data)

# Print the status code
print("POST Status Code:", r.status_code)

# Print the result from the POST response
print("POST Response:", r.json())
