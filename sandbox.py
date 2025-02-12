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



#-------------------------------------MAIN-------------------------------------------
df = pd.read_csv("data/census.csv") 

print(df.head())  # Show first 5 rows
print(df.info())  # Show column types and null values
print(df.describe())  # Show basic statistics
