import functools
from random import shuffle

import numpy as np
import pandas as pd

"""
bias = 4
X = np.array([[2., 3.],
              [4., 5.],
              [6., 7.]])
w = np.array([bias, 2., 3.])

ones = np.ones((X.shape[0], 1))
X_with1 = np.hstack((ones, X))
print(X_with1)
print(np.dot(X_with1, w))

Y = np.array([[3., 5.],
              [7., 20.]])

X_with_Y = np.vstack((X, Y))
print(X_with_Y)
"""

X = np.array([[2., 3., 4.],
              [4., 5., 6.],
              [6., 7., 8.]])

dim = X.shape[1]
indics = tuple(range(dim))
print(dim)
print(indics)

from itertools import combinations
c = combinations(indics, dim)
print(c)

# 会优先匹配*之外的变量
a, *b, c = [1, 2]
print(a, b, c)

print([a if a > 0 else 0 for a in [-1, 0, 1]])

# 两个列表同时解析, 使用zip函数
for teama, teamb in zip(['packers', '34dfe'], ['raverns', '14gbdf45']):
    print(teama , "vs", teamb)

# 带索引的列表解析: 使用enumerate函数
for index, team in enumerate(['packers', '34dfe','raverns', '14gbdf45']):
    print(index, team)

M = [[1,2,3], [4,5,6], [7,8,9 ]]
G = (sum(row) for row in M)
for i in range(1,4):
    print(next(G))

start = 100
def tester(start):
    def nested(label):
        nonlocal start          # 定义start为域内变量,而且下面加法操作不会改变域外的值
        print("test before added", label, start)
        start += 3
        print("test after added", label, start)
    return nested

def tester2(start):
    def nested(label):
        global  start           # 指定start为域外的变量,而且下面的加法操作会改变域外的值
        print("test before added", label, start)
        start += 3
        print("test after added", label, start)
    return nested

start1 = 120
t = tester(start1)
t(1)
t = tester2(start1)
t(2)
print(start, start1)

def func(a:'spam', b:(1,10)=2, c:float=3.0) -> int:
    print(a, b, c)

print(func.__annotations__)

# lambda
print(list(map((lambda x: x + 1), [1,2,3])))
print(list(filter((lambda x: x > 0), range(-4,5))))
print(functools.reduce((lambda x,y: x + y), [1,2,3]))
print(functools.reduce((lambda x,y: x * y), [1,2,3]))

print("hello world")

df = pd.read_csv("/home/zhenmie/Documents/python/datas/top250_f1.csv", sep="#", encoding='utf8')
df = df.iloc[99:, :]
print(df)
