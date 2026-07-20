import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

y_true = np.array([1]*50 + [2]*50 + [3]*60)
y_prob = np.random.rand(160, 4)
y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

try:
    present_classes = np.unique(y_true)
    y_prob_present = y_prob[:, present_classes]
    y_prob_present = y_prob_present / y_prob_present.sum(axis=1, keepdims=True)
    auc_val = float(roc_auc_score(y_true, y_prob_present, multi_class='ovr', average='macro'))
    print("ROC AUC works:", auc_val)
except Exception as e:
    print(f"roc_auc_score error: {e}")
