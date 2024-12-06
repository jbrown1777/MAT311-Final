from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def specificity_score(y_true, y_pred_model):
    # Separate Confusion Matrix for Specificity Value
    TP = confusion_matrix(y_true,y_pred_model)[1][1]
    TN = confusion_matrix(y_true,y_pred_model)[0][0]
    FP = confusion_matrix(y_true,y_pred_model)[0][1]
    FN = confusion_matrix(y_true,y_pred_model)[1][0]
    return TN / (TN + FP)

def printConfusionMatrix(y_true, y_pred_model, model, status):
    confusion_matrix_model = confusion_matrix(y_true, y_pred_model)
    sns.heatmap(confusion_matrix_model, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {model} ({status})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    print()
    
def printResults(y_true, y_pred_model):
    print(f'Accuracy: {accuracy_score(y_true, y_pred_model)}')
    print(f'Precision: {precision_score(y_true, y_pred_model)}')
    print(f'Recall: {recall_score(y_true, y_pred_model)}')
    print(f'Specificity: {specificity_score(y_true, y_pred_model)}')
    print(f'F1-Score: {f1_score(y_true, y_pred_model)}')
