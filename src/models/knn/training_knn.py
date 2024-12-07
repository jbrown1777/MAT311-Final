best_score = 0
best_k = 0

for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    #y_pred_val = knn.predict(X_val)
    score = knn.score(X_val, y_val)
    if score > best_score:
        best_score = score
        best_k = k
