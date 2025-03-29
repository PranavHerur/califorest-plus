from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

try:
    from venn_abers import VennAbersCalibrator
except ImportError:
    raise ImportError(
        "Please install the venn_abers package using: pip install venn_abers"
    )

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_classes=3, n_informative=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Initialize and train the base classifier
clf = GaussianNB()
clf.fit(X_train, y_train)

# Define Venn-ABERS calibrator
va = VennAbersCalibrator(estimator=clf, inductive=True, cal_size=0.2, random_state=42)

# Fit on the training set
va.fit(X_train, y_train)

# Generate probabilities and class predictions on the test set
p_prime = va.predict_proba(X_test)
y_pred = va.predict(X_test)

# accuracy of multiclass classifer
print(f"Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.4f}")
