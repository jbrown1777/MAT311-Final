features = X_train.columns
depth_limit = None

decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth_limit)

# Fit the model
decision_tree.fit(X_train[features], y_train)

# Predict on validation data
y_pred_val_decision_tree = decision_tree.predict(X_val[features])
