import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Create dataset
X = np.array([
    [10, 9],
    [5, 6],
    [2, 3],
    [8, 8],
    [1, 2],
    [7, 7],
    [3, 4],
    [9, 9]
])

y = np.array([0, 0, 1, 0, 1, 0, 1, 0])

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 3: Create model
model = LogisticRegression()

# Step 4: Train
model.fit(X_train, y_train)

print("Weights:", model.coef_)
print("Bias:", model.intercept_)

# Step 5: Predict
predictions = model.predict(X_test)

# Step 6: Evaluate
print("Predictions:", predictions)
print("Actual:", y_test)
print("Accuracy:", accuracy_score(y_test, predictions))