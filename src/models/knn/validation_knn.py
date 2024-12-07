# Generate new model with the best k-value
knn = KNeighborsClassifier(n_neighbors=best_k)

# Validate the new model
knn.fit(X_train, y_train)
y_pred_val_knn = knn.predict(X_val)

# Print values of the confusion matrix
printConfusionMatrix(y_val, y_pred_val_knn, 'KNN', 'Validation')

# Print results
printResults(y_val, y_pred_val_knn)
