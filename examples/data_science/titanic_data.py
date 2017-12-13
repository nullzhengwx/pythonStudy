# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv("/home/zhenmie/Documents/ml/dataset/titanic/train.csv")
test_df = pd.read_csv("/home/zhenmie/Documents/ml/dataset/titanic/test.csv")
combine = [train_df, test_df]
print(train_df.columns.values)

# What are the data types for various features
print(train_df.info())
print("-"*40)

# What is the distribution of numerical feature values across the samples
print(train_df.describe())

# What is the distribution of categorical features?
print(train_df.describe(include=['O']))

# The upper-class passengers(Pclass=1) were more like to have survived
# train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()\
#     .sort_values(by='Survived', ascending=False)

"""
# Sex=female had very high survival rate at 74%
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()\
    .sort_values(by='Survived', ascending=False)

# SibSp and Parch These features have zero correlation for certain values.
# It may be best to derive a feature or a set of features from these individual features.
train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()\
    .sort_values(by='Survived', ascending=False)
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()\
    .sort_values(by='Survived', ascending=False)

# Correlating numerical features
# We should consider Age in our model training
# Complete the Age feature for null values
# We should band age groups
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# Correlating numerical and ordinal features
# Consider Pclass for model trainning
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=6.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)    # alpha参数可以设置颜色
grid.add_legend()

# Correlating categorical features
# Add Sex feature to model training
# Complete and add Embarked feature to model training.
grid = sns.FacetGrid(train_df, row='Embarked',size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived','Sex', palette='deep', hue_order=["female", "male"])
grid.add_legend()
"""
# Correlating categorical and numerical features
# Consider baning Fare feature
grid = sns.FacetGrid(train_df, row="Embarked", col='Survived', size=2.2,aspect=1.6)
grid.map(sns.barplot,'Sex', 'Fare', alpha=.6, ci=None, order=['female', 'male'])
grid.add_legend()

plt.show()


# Correcting by dropping features
# parameter 'axis' uses to classify dropping by rows (0) or by column (1)
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

test_crosstab = pd.crosstab(train_df['Title'], train_df['Sex'])
print(test_crosstab)

for dataset in combine:
    dataset['Title']  = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# We can replace many titles with a more common name or classify them as Rare
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# We can convert the categorical tiltes to ordinal
title_mapping = {'Mr': 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# we can safely drop the Name feature from training and testing dataset.
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
print(train_df.shape, test_df.shape)

# converting a categorical feature
# convert Sex feature to a new feature called Gender where female=1 and male=0
if __name__ == '__main__':
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male':0}).astype(int)

# completing a numerical continuous feature
# more accurate way of guessing missing values is to use other correlated features;
# so guess Age values using median values for Age across sets of Pclass and Gender
# feature combinations

# view the Age and Pclass, Gender combinations
"""
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()
"""

# prepare an empty array to contain guessed Age values based on Pclass x Gender combinations
guess_ages = np.zeros((2, 3))
print(guess_ages)

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            # index supports .dropna() to exclude missing value
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # use random numbers between mean and standard deviation,
            # based on sets of Pclass and Gender combinations
            """
            age_mean = guess_df.mean()
            age_std = guess_df.std()
            age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)
            """

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess/0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & \
                        (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] \
            = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

# create Age bands and determine correlations with Survived

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean()\
    .sort_values(by='AgeBand', ascending=True)

# replace Age with ordinals based on these bands
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 64), 'Age'] = 4

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

# create a new feature for familySize which combines Parch and SibSp.
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()\
    .sort_values(by='Survived', ascending=False)

# create another feature called IsAlone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()\
    .sort_values(by='Survived', ascending=False)

# drop Parch, SibSp, and FamilySize features in favor of IsAlone
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

# create an artificial feature combining Pclass and Age
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

# fill the most common occurance of Embarked to missing value
# mode funciton : get the most frequently occurring value(s) of the values in a Series or DataFrame
freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()\
    .sort_values(by='Survived', ascending=False)

# convert the EmbarkedFill feature by creating a new numeric port fearture.
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# Quick completing and converting a numeric feature
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean()\
      .sort_values(by='FareBand', ascending=True)

for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

print(train_df.head())
print(test_df.head())

"""
With these two critiria - Supervised Learning plus Classification and Regression,
we can narrow down our choise of models to a few.
These include:
Logistic Regression
KNN or k-Nearest Neighbors
Support Vector machines
Naive Bayes classifier
Decision Tree
Random Forrest
Perceptron
Artificial neural network
RVM or Relevance Vector Machine
"""
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)

""" Logistic Regression
Logistic Regression measures the relationship between
the categorical dependent variable (feature, 分类因变量)
and one or more independent varialbes (features, 自变量)
by estimating probabilities using a logistic funciton,
which is the cumulative logistic distribution.
简单来说,就是用概率来表示相关度
"""
""" 实现过程
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)

# Positive coefficients increase the log-odds of the response (and thus increase the probability)
# Negative coefficients decrease the log-odds of the response (and thus decrease the probability)
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])
print(coeff_df.sort_index(by='Correlation', ascending=False))
"""

""" Support Vector Machines
Given a set of training samples, each marked as belonging to one
or the other of two categories, an SVM training algorithm builds
a model that assigns new test samples to one category or the other,
making it a non-probabilistic binary linear classifier.
"""
""" 实现过程
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print(acc_svc)
"""

""" k-Nearest Neighbors algorithm
a non-parametric method used for classification and regression.
A sample is classified by a majority vote of its neighbors,
with the sample being assigned to the class most common among
its k nearest neighbors(k is a positive integer, typically small).
If k = 1, then the ojbect is simply assigned to the class of that
single nearest neighbor.
"""
""" 实现过程
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn =round(knn.score(X_train, Y_train) * 100, 2)
print(acc_knn)
"""

""" naive Bayes classifiers (Gaussian Naive Bayes)
a family of simple probabilistic classifiers based on applying Bayes theorem
with strong(naive) independence assumptions between the features.
Naive Bayes classifiers are highly scalable, requiring a number of parameters
linear in the number of variables (features) in a linear problem.
"""
""" 实现过程
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print(acc_gaussian)
"""

""" Perceptron
functions that can decide whether an input represented by a vector of number,
belongs to some specific class or not.
It's a type of linear classifier.i.e, a classification algorithm thar makes
its predictions based on a linear predictor function combining a set of weight
with the feature vector.
"""
""" 实现过程　
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print(acc_perceptron)
"""

""" Linear SVC """
""" 实现过程　
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(acc_linear_svc)
"""

""" Stochastic Gradient Descent"""
""" 实现过程
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print(acc_sgd)
"""

""" Decision Tree
uses a decision tree as a predictive model which maps features (tree branches)
to conclusions about the target value (tree leaves).
"""
""" 实现过程
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(acc_decision_tree)
"""

""" Random Forest
Random forests or random decision forests are an ensemble learning method
for classification, regression and other tasks, that operate by
constructing a multitude of decision trees (n_estimators=100) at training time
 and outputting the class that is the mode of the classes (classification)
 or mean prediction (regression) of the individual trees.
"""
""" 实现过程
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)
"""