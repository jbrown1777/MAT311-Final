import pandas as pd
import numpy as np

def load_data():
    # Convert excel file into DataFrame
    # If you have trouble reading this file, you may need to do this command in the terminal: pip install openpyxl 
    autism = pd.read_excel('./data/Autism_Screening_Adult.xlsx')
    autism = autism.drop(columns=['Ethnicity','Country_of_Res','Used_App_Before','Relation','Age_Desc'])

    # Convert "yes" values to 1 and "no" values to 0
    autism = autism.replace({'no':0,'yes':1,'NO':0,'YES':1})

    # Convert "f" to 0 and "m" to 1
    autism = autism.replace({'f':0,'m':1})

    # Convert '?' and the outlier to NaN
    autism = autism.replace('?',np.nan)
    autism = autism.replace(383.0,np.nan)

    # Drop missing values and outlier
    autism = autism.dropna()
    return autism
