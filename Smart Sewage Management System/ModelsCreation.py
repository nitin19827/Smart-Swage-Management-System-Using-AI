import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pandas.core.arrays.interval import le
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv("newdataset.csv")

# Handling missing values
imputer = SimpleImputer(strategy='mean')
df["ph"] = imputer.fit_transform(df[["ph"]]).ravel()
df["Sulfate"] = imputer.fit_transform(df[["Sulfate"]]).ravel()

# Checking for remaining missing values
print(df.isnull().sum())

# Split the data into features (x) and target variable (y)
x = df.drop(['Potability'], axis=1)
y = df['Potability']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Logistic Regression Model Creation
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
lr_accuracy = lr_model.score(x_test, y_test)
print(f"Logistic Regression Model Accuracy: {lr_accuracy}")

# Save the Logistic Regression model
with open('logistic_regression_model.pkl', 'wb') as model_file:
    pickle.dump(lr_model, model_file)
print("Logistic Regression Model saved as logistic_regression_model.pkl")

# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)
dt_accuracy = dt_model.score(x_test, y_test)
print(f"Decision Tree Model Accuracy: {dt_accuracy}")

# Save the Decision Tree model
with open('decision_tree_model.pkl', 'wb') as model_file:
    pickle.dump(dt_model, model_file)
print("Decision Tree Model saved as decision_tree_model.pkl")

# K Nearest Neighbours Model
knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
knn_accuracy = knn_model.score(x_test, y_test)
print(f"K Nearest Neighbours Model Accuracy: {knn_accuracy}")

# Save the K Nearest Neighbours model
with open('knn_model.pkl', 'wb') as model_file:
    pickle.dump(knn_model, model_file)
print("K Nearest Neighbours Model saved as knn_model.pkl")

# Random Forest Model
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
rf_accuracy = rf_model.score(x_test, y_test)
print(f"Random Forest Model Accuracy: {rf_accuracy}")

# Save the Random Forest model
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)
print("Random Forest Model saved as random_forest_model.pkl")


#SVM
svm_model = SVC()
svm_model.fit(x_train, y_train)
svm_accuracy = svm_model.score(x_test, y_test)
print(f"Support Vector Machine Model Accuracy: {svm_accuracy}")

# Save the SVM model
with open('support_vector_machine.pkl', 'wb') as model_file:
    pickle.dump(svm_model, model_file)
print("SVM Model saved as support_vector_machine.pkl")

#AdaBoost Classifier
ada_model = AdaBoostClassifier()
ada_model.fit(x_train, y_train)
ada_accuracy = ada_model.score(x_test, y_test)
print(f"AdaBoost Classifier Model Accuracy: {ada_accuracy}")

# Save the AdaBoost Classifier model
with open('adaboost_classifier.pkl', 'wb') as model_file:
    pickle.dump(ada_model, model_file)
print("AdaBoost Classifier Model saved as adaboost_classifier.pkl")

#XGBoost
xg_model = XGBClassifier()
xg_model.fit(x_train, y_train)
xg_accuracy = xg_model.score(x_test, y_test)
print(f"XGBoost Classifier Model Accuracy: {xg_accuracy}")

# Save the XGBoost Classifier model
with open('xgboost_classifier.pkl', 'wb') as model_file:
    pickle.dump(xg_model, model_file)
print("XGBoost Classifier Model saved as xgboost_classifier.pkl")

