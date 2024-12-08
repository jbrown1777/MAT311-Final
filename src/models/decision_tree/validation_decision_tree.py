import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from src.data.helper_functions import *

def validate_tree(decision_tree, y_val, y_pred_val_decision_tree,features,depth_limit):
    # Print values of the confusion matrix
    printConfusionMatrix(y_val, y_pred_val_decision_tree, 'Decision Tree', 'Validation')

    # Print results
    printResults(y_val, y_pred_val_decision_tree,'Decision Tree')
    print()

    # Print the tree
    plt.figure(figsize=(4,4))
    plot_tree(decision_tree, feature_names=features, class_names=['No ASD','Yes ASD'], filled=True)
    plt.title(f'Decision Tree (Features: {features}, Max Depth: {depth_limit})')
    plt.show()
