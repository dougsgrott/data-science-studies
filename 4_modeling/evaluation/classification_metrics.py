import pandas as pd
import numpy as np

# from scipy.stats import ks_2samp
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, log_loss, average_precision_score
) # roc_curve, precision_recall_curve

from ks_metric import calculate_ks

def calculate_model_metrics_datasets(dataset_dict):
    roc_auc, accuracy, f1, precision, recall, ks = [], [], [], [], [], []
    specificity, mcc, logloss, pr_auc, balanced_accuracy = [], [], [], [], []
    for set_name in dataset_dict:
        # x_ = dataset_dict[set_name]['x']
        y_ = dataset_dict[set_name]['y']
        y_pred = dataset_dict[set_name]['y_pred']
        y_prob = dataset_dict[set_name]['y_prob']

        # tn, fp, fn, tp = confusion_matrix(y_, y_pred).ravel()
        tp = int(((y_pred == 1) & (y_ == 1)).sum())
        fp = int(((y_pred == 1) & (y_ == 0)).sum())
        tn = int(((y_pred == 0) & (y_ == 0)).sum())
        fn = int(((y_pred == 0) & (y_ == 1)).sum())

        roc_auc.append(roc_auc_score(y_, y_pred))
        accuracy.append(accuracy_score(y_, y_pred))
        f1.append(f1_score(y_, y_pred))
        precision.append(precision_score(y_, y_pred))
        recall.append(recall_score(y_, y_pred))
        # ks.append(calculate_ks(y_, y_prob)[0])

        specificity.append(tn / (tn + fp))  # Specificity (True Negative Rate)
        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc_ = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0
        mcc.append(mcc_)
        try:
            logloss.append(log_loss(y_, y_prob))
        except ValueError:
            logloss.append(np.nan)
        pr_auc.append(average_precision_score(y_, y_prob))  # Precision-Recall AUC
        balanced_accuracy.append((recall[-1] + specificity[-1]) / 2)  # Balanced Accuracy

    metrics = pd.DataFrame({
        'ROC AUC': roc_auc,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        # 'KS': ks,
        'Specificity': specificity,
        'MCC': mcc,
        'Log Loss': logloss,
        'PR AUC': pr_auc,
        'Balanced Accuracy': balanced_accuracy,
    }, index=dataset_dict.keys())

    return metrics
