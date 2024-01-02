import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE

data = pd.read_csv('creditcard.csv')
data.head()

kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42)
features = data.drop("Class", axis=1)

kmeans.fit(features)

cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Time', y='V2', data=features, hue=cluster_labels, palette='viridis', s=50, alpha=0.7)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', s=200, c='red', label='Cluster Centers')
plt.title('K-means Clustering')
plt.xlabel('Time')
plt.ylabel('V2')
plt.legend()
plt.show()


fraud_count = [0, 0]
for i in range(data.shape[0]):
    if data['Class'][i] == 1:
        fraud_count[cluster_labels[i]] += 1


print("Fraud count in cluster 0:", fraud_count[0])
print("Fraud count in cluster 1:", fraud_count[1])


count = [sum(cluster_labels == i) for i in range(2)]
print("Total count in cluster 0:", count[0])
print("Total count in cluster 1:", count[1])


percentage_fraud_cluster_0 = fraud_count[0] / count[0] * 100
percentage_fraud_cluster_1 = fraud_count[1] / count[1] * 100

print('Percentage of fraud in cluster 0:', percentage_fraud_cluster_0)
print('Percentage of fraud in cluster 1:', percentage_fraud_cluster_1)


x = data.drop(columns='Class', axis=1)
y = data['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=123)

oversampler = SMOTE()
x_train, y_train = oversampler.fit_resample(x_train, y_train)

sns.histplot(y_train)
plt.show()


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(f'x_train shape: {x_train.shape}\n, x_test shape: {x_test.shape}\n, y_train shape: {y_train.shape}\n, y_test shape: {y_test.shape}')


svm = SVC()
svm.fit(x_train, y_train)

y_train_pred = svm.predict(x_train)
y_train_cl_report = classification_report(y_train, y_train_pred, target_names=['No Fraud', 'Fraud'])
print("_" * 100)
print("TRAIN MODEL CLASSIFICATION REPORT")
print("_" * 100)
print(y_train_cl_report)

y_test_pred = svm.predict(x_test)
y_test_cl_report = classification_report(y_test, y_test_pred, target_names=['No Fraud', 'Fraud'])
print("_" * 100)
print("TEST MODEL CLASSIFICATION REPORT")
print("_" * 100)
print(y_test_cl_report)

print("_" * 100)


def train_logistic_regression(x_train, y_train):
    lr = LogisticRegression(penalty='l2', C=1.0, random_state=42, solver='liblinear')
    lr.fit(x_train, y_train)
    return lr

def generate_classification_report(y_true, y_pred, target_names):
    return classification_report(y_true, y_pred, target_names=target_names)

def logistic_regression(x_train, y_train, x_test, y_test):
    # Train the logistic regression model
    lr = train_logistic_regression(x_train, y_train)

    # Generate classification reports for the training set
    y_train_pred = lr.predict(x_train)
    y_train_cl_report = generate_classification_report(y_train, y_train_pred, ['No Fraud', 'Fraud'])
    print("_" * 100)
    print("TRAIN MODEL CLASSIFICATION REPORT")
    print("_" * 100)
    print(y_train_cl_report)

    # Generate classification reports for the test set
    y_test_pred = lr.predict(x_test)
    y_test_cl_report = generate_classification_report(y_test, y_test_pred, ['No Fraud', 'Fraud'])
    print("_" * 100)
    print("TEST MODEL CLASSIFICATION REPORT")
    print("_" * 100)
    print(y_test_cl_report)
    print("_" * 100)

    return y_test_pred, lr

y_test_pred, lr = logistic_regression(x_train, y_train, x_test, y_test)