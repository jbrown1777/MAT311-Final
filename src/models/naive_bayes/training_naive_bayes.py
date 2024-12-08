from sklearn.naive_bayes import GaussianNB

def training_naive_bayes(X_train,y_train,X_val):
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)
    y_pred_val_bayes = naive_bayes.predict(X_val)

    return naive_bayes, y_pred_val_bayes
