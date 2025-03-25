import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from load_data import load_iris_data
df = load_iris_data()
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
ensemble_models = {
    "Bagging (Decision Tree)": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
    "Voting Classifier": VotingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('dt', DecisionTreeClassifier()),
        ('svm', SVC(probability=True))
    ], voting='soft')
}
results = {}
for name, model in ensemble_models.items():
    model.fit(X_train, y_train)  
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) 
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
print("\nEnsemble Model Performance Comparison:")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")
