import numpy as np
import pandas as pd

# Ignore pandas warnings
pd.options.mode.chained_assignment = None  # default='warn'

# Read in train and test data sets
train = pd.read_csv("./input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("./input/test.csv", dtype={"Age": np.float64}, )

# Merge them for data cleaning + feature engineering
data_df = pd.concat([train, test])
frames = [train, test, data_df]

# Peek at the training data
print("-----Training data peek-----")
print(train.head())

# Compare stats of train and test
print("\n-----Training data stats-----")
print(train.info())
print("\n-----Test data stats-----")
print(test.info())

# How should "Embarked" be filled in?
print("\n-----Embarked values-----")
print(data_df.Embarked.value_counts())

# Takeaway - Embarked: set default as "S"

# How about age?
print("\n-----Age values-----")
print(data_df.Age.value_counts())
print(data_df.Age.median())

# Impute missing data and create new features in each data set
for df in frames:
    # Set Embarked default to 'S'
    df["Embarked"] = df["Embarked"].fillna("S")

    # Mark missing entries for Cabin
    df["Cabin"] = df["Cabin"].fillna("Missing")

    # Change Sex to numeric
    df["Sex"][df["Sex"] == "male"] = 0
    df["Sex"][df["Sex"] == "female"] = 1

    # Change Embarked to numeric
    df["EmbarkedS"] = 0
    df["EmbarkedS"][df["Embarked"] == "S"] = 1
    df["EmbarkedC"] = 0
    df["EmbarkedC"][df["Embarked"] == "C"] = 1
    df["EmbarkedQ"] = 0
    df["EmbarkedQ"][df["Embarked"] == "Q"] = 1

    # Add an indicator to say which cabin someone group someone was in (A-G)
    df["CabinStartsWithLetter"] = ""
    df["CabinStartsWithLetter"][df["Cabin"] != "Missing"] = df["Cabin"].str[0]

    df["CabinStartsWithA"] = 0
    df["CabinStartsWithA"][df["CabinStartsWithLetter"] == "A"] = 1
    df["CabinStartsWithB"] = 0
    df["CabinStartsWithB"][df["CabinStartsWithLetter"] == "B"] = 1
    df["CabinStartsWithC"] = 0
    df["CabinStartsWithC"][df["CabinStartsWithLetter"] == "C"] = 1
    df["CabinStartsWithD"] = 0
    df["CabinStartsWithD"][df["CabinStartsWithLetter"] == "D"] = 1
    df["CabinStartsWithE"] = 0
    df["CabinStartsWithE"][df["CabinStartsWithLetter"] == "E"] = 1
    df["CabinStartsWithG"] = 0
    df["CabinStartsWithG"][df["CabinStartsWithLetter"] == "F"] = 1
    df["CabinStartsWithF"] = 0
    df["CabinStartsWithF"][df["CabinStartsWithLetter"] == "G"] = 1

    # Add a boolean indicator to show if someone was with their family or not
    df["Family"] = df["Parch"] + df["SibSp"]
    df['Family'].loc[df['Family'] > 0] = 1
    df['Family'].loc[df['Family'] == 0] = 0

    # Is this passenger a child?
    df["Child"] = float(0)
    df["Child"][df["Age"] < 18] = 1
    df["Child"][df["Age"] >= 18] = 0

    # Starboard or port side? Starboard == odd numbers, port == even numbers
    df["OnStarboardSide"] = 0
    df["OnPortSide"] = 0
    df.loc[(df["Cabin"] != "Missing") & (df["Cabin"].str[-1].isin(["1", "3", "5", "7", "9"])), "OnStarboardSide"] = 1
    df.loc[(df["Cabin"] != "Missing") & (df["Cabin"].str[-1].isin(["2", "4", "6", "8", "0"])), "OnPortSide"] = 1

    # Family size
    df["FamilySize"] = df.Parch + df.SibSp + 1

    # Add a feature for title
    df["Title"] = df.Name.str.replace('(.*, )|(\\..*)', '')

    # Create binary features based on title
    df["IsMr"] = 0
    df.loc[(df["Title"].isin(['Mr', 'Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir'])), "IsMr"] = 1
    df["IsMrs"] = 0
    df.loc[(df["Title"].isin(['Mrs', 'the Countess', 'Mme', 'Lady'])), "IsMrs"] = 1
    df["IsMiss"] = 0
    df.loc[(df["Title"].isin(['Miss', 'Mlle', 'Ms'])), "IsMiss"] = 1
    df["IsMaster"] = 0
    df.loc[(df["Title"] == "Master"), "IsMaster"] = 1
    df.loc[(df["Title"] == "Dr") & (df["Sex"] == 0), "IsMr"] = 1
    df.loc[(df["Title"] == "Dr") & (df["Sex"] == 1), "IsMrs"] = 1

# Post-processing with the new features

# Handle missing fare with the median price paid by people in similar embarkments/ class
test.loc[test["PassengerId"] == 1044, "Fare"] = \
    data_df[(data_df["Embarked"] == 'S') & (data_df["Pclass"] == 3)].Fare.median()
data_df.loc[data_df["PassengerId"] == 1044, "Fare"] = \
    data_df[(data_df["Embarked"] == 'S') & (data_df["Pclass"] == 3)].Fare.median()

# Set age to the median value for that title
for title_value in data_df[np.isnan(data_df["Age"])].Title.unique():
    for df in frames:
        df.loc[(df["Title"] == title_value) & np.isnan(df["Age"]), "Age"] = \
            data_df[data_df["Title"] == title_value].Age.median()

# Preview the cleaned up data
print("-----Training data peek-----")
print(train.head())

print("\n-----Training data stats-----")
print(train.info())


# Save the transformed data
train.to_csv('./input/train_cleaned.csv', index=False)
test.to_csv('./input/test_cleaned.csv', index=False)