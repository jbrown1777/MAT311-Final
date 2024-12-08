from sklearn.neighbors import KNeighborsClassifier

def training_knn(X_train,y_train,X_val,y_val):
    best_score = 0
    best_k = 0
    
    for k in range(1,21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        score = knn.score(X_val, y_val)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k
