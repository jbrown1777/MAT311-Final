naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred_val_bayes = naive_bayes.predict(X_val)
