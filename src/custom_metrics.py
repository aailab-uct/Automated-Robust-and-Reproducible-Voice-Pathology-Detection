"""Custom metrics for binary classification tasks"""
from sklearn.metrics import confusion_matrix

def bookmakers_informedness(y_true, y_pred):
    """compute Bookmaker's Informedness"""
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Extract true negatives, false positives, false negatives, true positives
    tn, fp, fn, tp = cm.ravel()
    # Compute sensitivity (recall) and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    # Calculate Bookmaker's Informedness
    informedness = sensitivity + specificity - 1
    return informedness


def unweighted_average_recall_score(y_true, y_pred):
    """compute unweighted average recall"""
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Extract true negatives, false positives, false negatives, true positives
    tn, fp, fn, tp = cm.ravel()
    # Compute sensitivity (recall) and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    # Calculate unweighted average recall
    uar = (sensitivity + specificity) * 0.5
    return uar
