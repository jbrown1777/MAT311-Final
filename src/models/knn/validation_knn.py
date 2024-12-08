from sklearn.neighbors import KNeighborsClassifier
from src.data.helper_functions import printConfusionMatrix,printResults

def validate_knn(best_k,X_train,y_train,X_val,y_val):
    # Generate new model with the best k-value
    knn = KNeighborsClassifier(n_neighbors=best_k)

    # Validate the new model
    knn.fit(X_train, y_train)
    y_pred_val_knn = knn.predict(X_val)

    # Print values of the confusion matrix
    printConfusionMatrix(y_val, y_pred_val_knn, 'KNN', 'Validation')

    # Print results
    printResults(y_val, y_pred_val_knn,'KNN')

    return knn, y_pred_val_knn
