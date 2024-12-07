# KNN was the best model
y_pred_test_knn = knn.predict(X_test)

# Print values of the confusion matrix
printConfusionMatrix(y_test, y_pred_test_knn, 'KNN', 'Test')

# Print results
printResults(y_test, y_pred_test_knn)
