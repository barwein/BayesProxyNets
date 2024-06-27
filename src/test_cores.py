import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=100, random_state=42)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a logistic regression classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict probabilities on the test set
probs = clf.predict_proba(X_test)[:, 1]

# Calibrate the classifier using Platt Scaling
calibrated_clf_platt = CalibratedClassifierCV(estimator=clf, method='sigmoid', cv='prefit')
calibrated_clf_platt.fit(X_train, y_train)
probs_calibrated_platt = calibrated_clf_platt.predict_proba(X_test)[:, 1]

# Calibrate the classifier using Isotonic Regression
calibrated_clf_iso = CalibratedClassifierCV(estimator=clf, method='isotonic', cv='prefit')
calibrated_clf_iso.fit(X_train, y_train)
probs_calibrated_iso = calibrated_clf_iso.predict_proba(X_test)[:, 1]

# Plot the calibration curves
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')

prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', label='Uncalibrated')

prob_true_platt, prob_pred_platt = calibration_curve(y_test, probs_calibrated_platt, n_bins=10)
plt.plot(prob_pred_platt, prob_true_platt, marker='o', label='Platt Scaling')

prob_true_iso, prob_pred_iso = calibration_curve(y_test, probs_calibrated_iso, n_bins=10)
plt.plot(prob_pred_iso, prob_true_iso, marker='o', label='Isotonic Regression')

plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curves')
plt.legend()
plt.show()
