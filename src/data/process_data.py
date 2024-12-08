import numpy as np
from sklearn.model_selection import train_test_split # make training & testing data

def process_data(autism):
    X = autism.drop(columns=['Age','Result','Class/ASD'])
    y = autism['Class/ASD']

    ### Separate Training, Validation, and Test Data ###
    seed=np.random.seed(123)

    # Set up the Test Data
    #global X_train, X_val, X_test,y_train,  y_val, y_test
    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=141) #140 rows is 20% of the autism dataframe; we added one to leave 560 rows left 

    # Set up Training and Validation Data
    X_train, X_val, y_train, y_val = train_test_split(X_val, y_val, test_size=140) # 140 rows is 20% of the autism dataframe

    return X_train, X_val, X_test, y_train, y_val, y_test
