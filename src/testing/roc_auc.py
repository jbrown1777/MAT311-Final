fig, ax = plt.subplots()

# K-Nearest Neighbors
y_knn_pred_val_proba = knn.predict_proba(X_val)
knn_fpr, knn_tpr, knn_threshold = roc_curve(y_val, y_knn_pred_val_proba[:,1])
knn_auc = roc_auc_score(y_val, y_pred_val_knn)
plt.plot(knn_fpr, knn_tpr, label='KNN')

# Bayes
y_bayes_pred_val_proba = naive_bayes.predict_proba(X_val)
bayes_fpr, bayes_tpr, bayes_threshold = roc_curve(y_val, y_bayes_pred_val_proba[:,1])
bayes_auc = roc_auc_score(y_val, y_pred_val_bayes)
plt.plot(bayes_fpr, bayes_tpr, label='Naive Bayes')

# Decision Tree
y_dt_pred_val_proba = decision_tree.predict_proba(X_val)
dt_fpr, dt_tpr, dt_threshold = roc_curve(y_val, y_dt_pred_val_proba[:,1])
dt_auc = roc_auc_score(y_val, y_pred_val_decision_tree)
plt.plot(dt_fpr, dt_tpr, label='Decision Tree')

ax.legend()
plt.show()
print()
print()
print('KNN AUC:          ',knn_auc)
print('Naive Bayes AUC:  ', bayes_auc)
print('Decision Tree AUC:', dt_auc)
