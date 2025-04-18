from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

try:
    from venn_abers import VennAbersCalibrator
except ImportError:
    raise ImportError(
        "Please install the venn_abers package using: pip install venn_abers"
    )

# Generate data
poly = PolynomialFeatures()
X, y = load_diabetes(return_X_y=True)
X = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Initialize and train the base classifier
clf = RandomForestClassifier()

# Define Venn-ABERS calibrator
va = VennAbersCalibrator(
    estimator=clf,
    inductive=False,
    n_splits=2,
    precision=4,
    random_state=42,
)

# Fit on the training set
va.fit(X_train, y_train)

# Generate probabilities and class predictions on the test set
p_prime = va.predict_proba(X_test)
y_pred = va.predict(X_test, one_hot=False)
print(y_pred)

# accuracy of multiclass classifer
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
