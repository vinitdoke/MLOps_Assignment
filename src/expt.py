# import stuff
import pandas as pd
import dvc.api
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import f1_score
import json

# Read Data using DVC into Pandas
with dvc.api.open(repo="https://github.com/vinitdoke/MLOps_Assignment.git", path="data/creditcard.csv", mode="r") as fd:
    df = pd.read_csv(fd)


# Perform EDA
std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)
scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)


# # Split Data and Store
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]
new_df = pd.concat([fraud_df, non_fraud_df])

final_df = new_df.sample(frac=1, random_state=42)

train_df, test_df = train_test_split(final_df, test_size=0.2)

train_df.to_csv(r"data\processed\train.csv")
test_df.to_csv(r"data\processed\test.csv")



# # Train Decision Tree using Scikit

train_df = pd.read_csv(r"data\processed\train.csv")
train_df = train_df[train_df.columns[1:]]
X_train = train_df.drop("Class", axis = 1)
y_train = train_df["Class"]
# print(final_df.columns)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf = clf.fit(X_train, y_train)

# # Save Trained Model
joblib.dump(clf, 'models/model.pkl')

# # Save Accuracy and F1 Score into JSON in Metrics
test_df = pd.read_csv(r"data\processed\train.csv")
test_df = test_df[test_df.columns[1:]]
# print(test_df.columns)
X_test = test_df.drop("Class", axis = 1)
y_test = test_df["Class"]
y_pred = clf.predict(X_test)
f1 = f1_score(y_test, y_pred, average = 'weighted')
accuracy = clf.score(X_test, y_test)
Jfile = {}
Jfile['Accuracy'] = accuracy
Jfile['weighted F1 Score'] = f1
with open(r'metrics\acc_f1.json', 'w') as f:
    json.dump(Jfile, f)