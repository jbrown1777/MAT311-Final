from src.data.helper_functions import printConfusionMatrix,printResults

def validate_naive_bayes(y_val,y_pred_val_bayes):
    # Print values of the confusion matrix
    printConfusionMatrix(y_val, y_pred_val_bayes, 'Gaussian Naive Bayes', 'Validation')

    # Print results
    printResults(y_val, y_pred_val_bayes,'Gaussian Naive Bayes')
