import warnings
warnings.filterwarnings('ignore')
import pandas  as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import timeit
%matplotlib inline

#function to plot confusion matrix
def CMLABEL(cm,title):
    f,ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(cm,annot=True,fmt="d")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)

# Load in dataset
data = pd.read_csv('UCI_Credit_Card.csv')
print(data.head())

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.4, random_state = 0)
# Ensure the test data doesn't have the answer
test_solution = test['default.payment.next.month']
test = test.drop('default.payment.next.month', axis = 1)

train.describe()
plt.figure(figsize=(15,12))
cor = round(train.corr(),2)
sns.heatmap(cor, cmap = sns.color_palette('BuGn'), annot = True)
train[['PAY_0', 'default.payment.next.month']].groupby(['PAY_0'], as_index = False).mean()

# Function to get default payment means 
def get_pay_mean(PAY_NUM):
    temp = train[[PAY_NUM, 'default.payment.next.month']].groupby([PAY_NUM], as_index = True).mean()
    pay_mean = temp['default.payment.next.month']
    return pay_mean
pay_means = {}
for i in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
    pay_means[i] = get_pay_mean(i)
pay_means_results = pd.DataFrame(pay_means)
pay_means_results

pay_means_results.plot(kind = 'bar', title = 'PAY_# Mean Results', figsize=(15, 7), legend=True, rot = 0, colormap = 'Set2')
# Limit Balance and Default Rate Distribution
age_survival_dist = sns.FacetGrid(train, hue = 'default.payment.next.month', aspect = 2.5, size = 5, palette = 'BuGn')
age_survival_dist.map(sns.kdeplot, 'LIMIT_BAL', shade = True)
age_survival_dist.add_legend()
plt.suptitle('Limit Balance and Default Rate Distribution', fontsize = 20, y = 1.05)

# Age and Default Rate Distribution
age_survival_dist = sns.FacetGrid(train, hue = 'default.payment.next.month', aspect = 2.5, size = 5, palette = 'BuGn')
age_survival_dist.map(sns.kdeplot, 'AGE', shade = True)
age_survival_dist.add_legend()
plt.suptitle('Age and Default Rate Distribution', fontsize = 20, y = 1.05)

train[['SEX', 'default.payment.next.month']].groupby(['SEX'], as_index = False).mean()
train[['MARRIAGE', 'default.payment.next.month']].groupby(['MARRIAGE'], as_index = False).mean()
train[['EDUCATION', 'default.payment.next.month']].groupby(['EDUCATION'], as_index = False).mean()

credit_card = train.append(test, ignore_index = True)

credit_card['MARRIAGE'].replace(0, 3, inplace = True)
credit_card['EDUCATION'].replace([0, 5, 6], 4, inplace = True)
credit_card = credit_card.drop(['ID'], axis = 1)
credit_card.shape
train_cleaned = credit_card[0:18000]
test_cleaned = credit_card[18000:30000]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

features = train_cleaned.drop('default.payment.next.month', axis=1)
target = train_cleaned['default.payment.next.month']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=100)

#Logistic Regression Model
logr = LogisticRegression()
logr_parameters = {'penalty': ['l1', 'l2'], 
                   'C' : [10, 20, 30, 40, 50, 60]
                  }

acc_scorer = make_scorer(accuracy_score)

# Running the 10-fold grid search
grid_obj = GridSearchCV(logr, logr_parameters, cv = 10, scoring = acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Setting the algorithm to find the best combination of parameters
logr = grid_obj.best_estimator_

# Fitting the algorithm to the data. 
logr.fit(X_train, y_train)
start = timeit.default_timer()
y_pred=logr.predict(X_test)
stop = timeit.default_timer()
conf_mat = confusion_matrix(y_test,y_pred)
CMLABEL(conf_mat,"CONFUSION MATRIX-LOGISTIC REGRESSION")
print(round(logr.score(X_train, y_train) * 100, 2))
print((stop-start)*100)


#k-Nearest Neighbor Classifier
knn = KNeighborsClassifier()
knn_parameters = {'n_neighbors': range(2,6),
                  'leaf_size': [3, 5, 7, 10]
                 }

acc_scorer = make_scorer(accuracy_score)

# Running the 10-fold grid search
grid_obj = GridSearchCV(knn, knn_parameters, cv = 10, scoring = acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Setting the algorithm to find the best combination of parameters
knn = grid_obj.best_estimator_

# Fitting the best algorithm to the data. 
knn.fit(X_train, y_train)
start = timeit.default_timer()
y_pred=knn.predict(X_test)
stop = timeit.default_timer()
conf_mat= confusion_matrix(y_test,y_pred)
CMLABEL(conf_mat,"CONFUSION MATRIX- k-Nearest Neighbors")
print(round(knn.score(X_train, y_train) * 100, 2))
print((stop-start)*100)

#Naive-Bayes Classifier
nbc = GaussianNB()
nbc_parameters = {}

acc_scorer = make_scorer(accuracy_score)

# Running the 10-fold grid search
grid_obj = GridSearchCV(nbc, nbc_parameters, cv = 10, scoring = acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Setting the algorithm to the find best combination of parameters
nbc = grid_obj.best_estimator_

# Fitting the best algorithm to the data. 
nbc.fit(X_train, y_train)
start = timeit.default_timer()
y_pred=nbc.predict(X_test)
stop = timeit.default_timer()
conf_mat=confusion_matrix(y_test,y_pred)
CMLABEL(conf_mat,"CONFUSION MATRIX-NAIVE_BAYES")
print(round(nbc.score(X_train, y_train) * 100, 2))
print((stop-start)*100)
