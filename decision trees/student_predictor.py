import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('STUDENT_DATA_-_Sheet1.csv')

# Prepare data
X = df.drop('Result', axis=1)
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train TUNED Decision Tree
model = DecisionTreeClassifier(max_depth=6, min_samples_split=5, min_samples_leaf=2, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))

# Feature Importance Chart
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fail', 'Pass'], 
            yticklabels=['Fail', 'Pass'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print("\n✓ Charts saved!")
# Test new student
new_student = [[75, 80, 5.5]]
prediction = model.predict(new_student)
probability = model.predict_proba(new_student)

print(f"\n{'='*50}")
print("NEW STUDENT PREDICTION")
print(f"{'='*50}")
print(f"Attendance: 75%, Marks: 80, Study Hours: 5.5")
print(f"Result: {'✓ PASS' if prediction[0] == 1 else '✗ FAIL'}")
print(f"Confidence: {probability[0][prediction[0]] * 100:.2f}%")

#hyperparameters analysis graph
from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(12, 6))
sns.lineplot(x='param_max_depth', y='mean_test_score', hue='param_min_samples_split', style='param_min_samples_leaf', data=results)
plt.title('Hyperparameter Tuning Results')
plt.xlabel('Max Depth')
plt.ylabel('Mean Test Score')
plt.legend(title='Min Samples Split / Leaf')
plt.tight_layout()
plt.savefig('hyperparameter_tuning.png')
plt.show()
print("\n✓ Hyperparameter tuning chart saved!")    