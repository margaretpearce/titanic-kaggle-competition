import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV

# Read in cleaned train and test data sets
train_df = pd.read_csv("./input/train_cleaned.csv", dtype={"Age": np.float64})
test_df = pd.read_csv("./input/test_cleaned.csv", dtype={"Age": np.float64})

# Choose features from the data to use for the models
list_features = ["Pclass",
                 "Sex",
                 "Age",
                 "SibSp",
                 "Parch",
                 "EmbarkedS",
                 "EmbarkedC",
                 "EmbarkedQ",
                 "CabinStartsWithA",
                 "CabinStartsWithB",
                 "CabinStartsWithC",
                 "CabinStartsWithD",
                 "CabinStartsWithE",
                 "CabinStartsWithG",
                 "CabinStartsWithF",
                 "Family",
                 "Child",
                 "OnStarboardSide",
                 "OnPortSide",
                 "FamilySize",
                 "IsMr",
                 "IsMrs",
                 "IsMiss",
                 "IsMaster"]

# Get train and test data set features
X_train = train_df[list_features].values
X_train_df = train_df[list_features]
Y_train = train_df[["Survived"]].values.ravel()

X_test = test_df[list_features].values
X_test_df = test_df[list_features]

# Baseline: assume everyone survives
baseline_score = 1 - train_df.Survived.mean()
print("Baseline score: ", baseline_score)

# Random Forests with different ranges
# 10, 20, 30, 40, ..
ranges = range(10, 100, 10)
param_grid = dict(n_estimators=list(ranges), criterion=["gini", "entropy"])

random_forest = RandomForestClassifier(n_estimators=20)
grid = GridSearchCV(random_forest, param_grid, cv=10, scoring='accuracy')
grid.fit(X_train_df, train_df["Survived"])
print("Random Forest: ", grid.best_score_, grid.best_params_)
rf_classifier = RandomForestClassifier(n_estimators=20, criterion='gini')
rf_classifier.fit(X_train_df, train_df["Survived"])
Y_pred_rf = rf_classifier.predict(X_test_df)

# K Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
print("KNN: ", knn.score(X_train, Y_train))

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_nb = gaussian.predict(X_test)
print("Gaussian Naive Bayes: ", gaussian.score(X_train, Y_train))

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svm = svc.predict(X_test)
print("SVM: ", svc.score(X_train, Y_train))

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_log = logreg.predict(X_test)
print("Logistic Regression: ", logreg.score(X_train, Y_train))

# Linear SVC
linreg = LinearSVC()
linreg.fit(X_train, Y_train)
Y_pred_lin = linreg.predict(X_test)
print("Linear SVC: ", linreg.score(X_train, Y_train))


# Prepare to score test set
pid = np.array(test_df["PassengerId"]).astype(int)
solution_rf = pd.DataFrame(Y_pred_rf, pid, columns=["Survived"])
solution_knn = pd.DataFrame(Y_pred_knn, pid, columns=["Survived"])
solution_nb = pd.DataFrame(Y_pred_nb, pid, columns=["Survived"])
solution_svm = pd.DataFrame(Y_pred_svm, pid, columns=["Survived"])
solution_logreg = pd.DataFrame(Y_pred_log, pid, columns=["Survived"])
solution_linsvm = pd.DataFrame(Y_pred_lin, pid, columns=["Survived"])

# Write solutions to a csv file to submit to Kaggle
solution_rf.to_csv("./output/titanic_solution_rf.csv", index_label=["PassengerId"]) # 0.75598
solution_knn.to_csv("./output/titanic_solution_knn.csv", index_label=["PassengerId"]) # 0.70
solution_nb.to_csv("./output/titanic_solution_nb.csv", index_label=["PassengerId"]) # 0.76077
solution_svm.to_csv("./output/titanic_solution_svm.csv", index_label=["PassengerId"]) # 0.77512
solution_logreg.to_csv("./output/titanic_solution_logreg.csv", index_label=["PassengerId"]) # 0.78947
solution_linsvm.to_csv("./output/titanic_solution_linsvm.csv", index_label=["PassengerId"]) # 0.77033
