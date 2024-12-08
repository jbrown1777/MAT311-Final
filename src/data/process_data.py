from sklearn.tree import DecisionTreeClassifier, plot_tree

def train_tree(X_train,y_train,X_val):
    features = X_train.columns
    depth_limit = None

    decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth_limit)

    # Fit the model
    decision_tree.fit(X_train[features], y_train)

    # Predict on validation data
    y_pred_val_decision_tree = decision_tree.predict(X_val[features])

    return decision_tree,y_pred_val_decision_tree,features,depth_limit
