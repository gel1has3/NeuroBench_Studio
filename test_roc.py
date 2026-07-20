import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize

n_classes = 3
n_samples = 160
y_true = np.array([0]*80 + [1]*80)

# Mock y_prob for n_classes = 2
y_prob = np.random.rand(160, 2)
y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

try:
    auc_val = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
except Exception as e:
    print(f"roc_auc_score error: {e}")

try:
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
except Exception as e:
    print(f"roc_curve error: {e}")
