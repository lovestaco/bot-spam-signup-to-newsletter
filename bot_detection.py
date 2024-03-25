import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Load your dataset
df = pd.read_csv('signup_samples.csv')

# Select features and target
X = df[["email", "name"]]
y = df["type"]

# One-Hot Encoding for categorical features
ohe = OneHotEncoder(handle_unknown='ignore')
encoded_features = ohe.fit_transform(X)
X = pd.DataFrame(encoded_features.toarray(), columns=ohe.get_feature_names_out())

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
