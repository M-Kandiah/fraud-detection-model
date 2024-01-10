import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



credit_card_data = pd.read_csv('creditcard.csv')

credit_card_data.head()

credit_card_data.isnull().sum() # no empty cells

credit_card_data['Class'].value_counts()

# Hghly unbaalnced dataset
# We have more normal transaction than fraudulent so the model we build  will be more likely to be classify as normal transaction
# than a fradulent one, therefore we need to create a balanced dataset with the same number of entries
# as our fradulent set of entries

normal = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


print(normal.Amount.describe())
# Normal
# count    284315.000000
# mean         88.291022
# std         250.105092
# min           0.000000
# 25%           5.650000
# 50%          22.000000
# 75%          77.050000
# max       25691.160000
# Name: Amount, dtype: float64

print(fraud.Amount.describe())
# Fraud
# count     492.000000
# mean      122.211321
# std       256.683288
# min         0.000000
# 25%         1.000000
# 50%         9.250000
# 75%       105.890000
# max      2125.870000
# Name: Amount, dtype: float64

# We can use these values to validate that the sample of the normal data we take
# has the same distribution and values that te parent dataset contained,
# so its a faithful representation of the main dataset

# So we need to get 492 normal transactions from the set of all normal transactions


normal_sample = normal.sample(n=492)
# print(normal_sample)
# print(normal_sample.Amount.describe())

# Normal sample
# count     492.000000
# mean       78.910955
# std       188.253643
# min         0.000000
# 25%         6.985000
# 50%        21.400000
# 75%        80.330000
# max      2150.000000
# Name: Amount, dtype: float64

# Normal
# count    284315.000000
# mean         88.291022
# std         250.105092
# min           0.000000
# 25%           5.650000
# 50%          22.000000
# 75%          77.050000
# max       25691.160000
# Name: Amount, dtype: float64


# The normal and normal sample have similar distribution enough to
# be happy that the sample set is a faithful representation of
# the parent set

# Fraud
# count     492.000000
# mean      122.211321
# std       256.683288
# min         0.000000
# 25%         1.000000
# 50%         9.250000
# 75%       105.890000
# max      2125.870000
# Name: Amount, dtype: float64


# Join the two equal sets of normal and fradulent transaction to form a new dataset
# with equal number of entries for each (n=492)

new_dataset = pd.concat([normal_sample, fraud], axis=0)

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model = LogisticRegression()

model.fit(X_train, Y_train)

#----------

#Evaluating the model

X_train_prediction = model.predict(X_train)
training_data_acc = accuracy_score(X_train_prediction, Y_train)

print(training_data_acc) # About 90% + accuracy score, which is good

X_test_prediction =model.predict(X_test)
training_data_acc_test = accuracy_score(X_test_prediction, Y_test)
print(training_data_acc_test) # Agan, about 90% + accuracy score



