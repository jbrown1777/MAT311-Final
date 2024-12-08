from src.data import helper_functions, load_data, process_data
from src.models.knn.training_knn import training_knn
from src.models.knn.validation_knn import validate_knn
from src.models.naive_bayes.training_naive_bayes import training_naive_bayes
from src.models.naive_bayes.validation_naive_bayes import validate_naive_bayes
from src.models.decision_tree.training_decision_tree import train_tree
from src.models.decision_tree.validation_decision_tree import validate_tree
from src.testing import roc_auc, testing_knn


### LOAD AND PROCESS DATA ###
autism = load_data.load_data()
X_train, X_val, X_test, y_train, y_val, y_test = process_data.process_data(autism)


### MODELS ###

# KNN
best_k = training_knn(X_train, y_train, X_val, y_val)
knn, y_pred_val_knn = validate_knn(best_k, X_train, y_train, X_val, y_val)

# Naive Bayes
naive_bayes, y_pred_val_bayes = training_naive_bayes(X_train, y_train, X_val)
validate_naive_bayes(y_val,y_pred_val_bayes)

# Decision Tree
decision_tree, y_pred_val_decision_tree, features, depth_limit = train_tree(X_train, y_train, X_val)
validate_tree(decision_tree, y_val, y_pred_val_decision_tree, features, depth_limit)


### TESTING ###
roc_auc(knn, naive_bayes, decision_tree, y_pred_val_knn, y_pred_val_bayes, y_pred_val_decision_tree)
testing_knn(knn, X_test, y_test)
