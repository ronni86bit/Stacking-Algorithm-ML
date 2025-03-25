import sys
sys.stdout.reconfigure(encoding='utf-8')
model_accuracies = {
    "Logistic Regression": 0.9333,
    "K-Nearest Neighbors": 0.9333,
    "Decision Tree": 0.9667,
    "Support Vector Machine": 0.9667,
    "Bagging (Decision Tree)": 0.9667,
    "Random Forest": 0.9000,
    "Gradient Boosting": 0.9667,
    "Voting Classifier": 0.9667,
    "Stacking Classifier": 0.9667
}
print("\n🔹 Final Model Accuracy Comparison 🔹")
for model, acc in model_accuracies.items():
    print(f"{model}: {acc:.4f}")
print("\n🔹 Conclusion 🔹")
print("")
print("- Decision Tree & SVM performed best among basic models (96.67%).")
print("- Ensemble methods (Bagging, Gradient Boosting & Voting) reached the same accuracy.")
print("- Stacking did not provide additional improvement.")
print("- The Iris dataset is simple, so complex models don't always give better results.")
