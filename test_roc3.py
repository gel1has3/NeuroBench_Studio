import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

y_true = np.array([1]*50 + [2]*50 + [3]*60)
y_prob = np.random.rand(160, 4)
y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

try:
    auc_val = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro', labels=[0, 1, 2, 3]))
    print("ROC AUC works:", auc_val)
except Exception as e:
    print(f"roc_auc_score error: {e}")
