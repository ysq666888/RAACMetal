import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os


def plot_auc_roc(y_true, y_pred_proba, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    save_path = os.path.join(save_dir, 'auc_roc.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return roc_auc




def plot_auc_pr(y_true, y_pred_proba, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    average_precision = average_precision_score(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AP = {average_precision:.4f}, AUC = {pr_auc:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.show()
    save_path = os.path.join(save_dir, 'auc_pr.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return average_precision, pr_auc
