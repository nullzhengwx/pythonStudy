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
grid.map(sns.pointplot, 'Pclass', 'Survived','Sex', palette='deep')
grid.add_legend()

# Correlating categorical and numerical features
# Consider baning Fare feature
grid = sns.FacetGrid(train_df, row="Embarked", col='Survived', size=2.2,aspect=1.6)
grid.map(sns.barplot,'Sex', 'Fare', alpha=.6, ci=None)
grid.add_legend()

#plt.show()
"""

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