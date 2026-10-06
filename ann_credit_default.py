# Artificial Neural Network on real bank-customer data
#
# Same ANN as the Churn_Modelling version, retrained on the real
# "Default of Credit Card Clients" dataset (UCI, CC BY 4.0):
#   Yeh, I. (2009). Default of Credit Card Clients [Dataset].
#   UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
# 30,000 real customers of a Taiwanese bank (April-September 2005).
# Target: will the customer default on next month's payment? (1 = yes)
#
# A CSV copy of the data is in data/ (see data/README.md for attribution).

# Importing the libraries
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)

SEED = 0
tf.keras.utils.set_random_seed(SEED)

# Part 1 - Data Preprocessing

# Importing the dataset
data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/default_of_credit_card_clients.csv'
dataset = pd.read_csv(data_path)
if dataset.columns[0] != 'ID':  # the .xls export has an extra title row
    dataset = pd.read_csv(data_path, header=1)
dataset = dataset.drop(columns=['ID'])
target = dataset.columns[-1]

# Cleaning undocumented category codes
# EDUCATION: 1 = graduate school, 2 = university, 3 = high school, 4 = others
# (0, 5 and 6 appear in the data but are not documented, so they become "others")
dataset['EDUCATION'] = dataset['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
# MARRIAGE: 1 = married, 2 = single, 3 = others (0 is undocumented)
dataset['MARRIAGE'] = dataset['MARRIAGE'].replace({0: 3})

# Encoding categorical data
# SEX is binary (1 = male, 2 = female), like Gender in the churn data
dataset['SEX'] = (dataset['SEX'] == 2).astype(int)
# One Hot Encoding EDUCATION and MARRIAGE, like Geography in the churn data
dataset = pd.get_dummies(dataset, columns=['EDUCATION', 'MARRIAGE'], dtype=int)

# Bill and payment amounts are heavily skewed (a few customers owe millions),
# so a signed log keeps them from dominating after scaling
money_cols = [c for c in dataset.columns if c.startswith(('LIMIT_BAL', 'BILL_AMT', 'PAY_AMT'))]
dataset[money_cols] = np.sign(dataset[money_cols]) * np.log1p(np.abs(dataset[money_cols]))

X = dataset.drop(columns=[target]).values.astype(float)
y = dataset[target].values
feature_names = list(dataset.drop(columns=[target]).columns)
print(X.shape, 'default rate:', y.mean())

# Splitting the dataset into the Training set and Test set
# (stratified, because only ~22% of customers default)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.Input(shape=(X_train.shape[1],)))

# Adding the input layer and the first hidden layer
# (more inputs than the churn data, so the hidden layers are a bit wider)
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))
ann.add(tf.keras.layers.Dropout(0.2))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=8, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# Training the ANN on the Training set
# 10% of the training set is held out to stop training when it stops improving
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=10,
                                              restore_best_weights=True)
history = ann.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.1,
                  callbacks=[early_stop], verbose=2)

# Part 4 - Making the predictions and evaluating the model

# Predicting a single customer (raw values, in the same column order as the CSV):
# limit 50,000 NT$, female, university, single, age 30, paid duly for the last
# two months but was 2 months late before that, with matching bills and payments
customer = pd.DataFrame([{
    'LIMIT_BAL': 50000, 'SEX': 2, 'EDUCATION': 2, 'MARRIAGE': 2, 'AGE': 30,
    'PAY_0': 0, 'PAY_2': 0, 'PAY_3': 2, 'PAY_4': 2, 'PAY_5': 0, 'PAY_6': 0,
    'BILL_AMT1': 40000, 'BILL_AMT2': 38000, 'BILL_AMT3': 39000,
    'BILL_AMT4': 30000, 'BILL_AMT5': 28000, 'BILL_AMT6': 25000,
    'PAY_AMT1': 2000, 'PAY_AMT2': 1500, 'PAY_AMT3': 0,
    'PAY_AMT4': 0, 'PAY_AMT5': 1500, 'PAY_AMT6': 1500,
}])
customer['SEX'] = (customer['SEX'] == 2).astype(int)
customer[money_cols] = np.sign(customer[money_cols]) * np.log1p(np.abs(customer[money_cols]))
for col in feature_names:
    if col.startswith(('EDUCATION_', 'MARRIAGE_')):
        prefix, value = col.rsplit('_', 1)
        customer[col] = int(customer[prefix].iloc[0] == int(value))
customer = customer[feature_names].values.astype(float)
print('Default probability for the example customer:', ann.predict(sc.transform(customer), verbose=0)[0, 0])

# Predicting the Test set results
y_prob = ann.predict(X_test, verbose=0).ravel()
y_pred = (y_prob > 0.5)

# Making the Confusion Matrix
# Accuracy alone is misleading here: always predicting "no default" scores ~78%,
# so ROC AUC and recall on defaulters are reported too
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Baseline (always "no default"):', 1 - y_test.mean())
print('ROC AUC:', roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred, target_names=['no default', 'default'], digits=3))

# Saving the test predictions and training history (used by make_figures.py)
import os
os.makedirs('results', exist_ok=True)
pd.DataFrame({'y_true': y_test, 'y_prob': y_prob}).to_csv('results/credit_default_predictions.csv', index=False)
pd.DataFrame(history.history).to_csv('results/credit_default_history.csv', index_label='epoch')
