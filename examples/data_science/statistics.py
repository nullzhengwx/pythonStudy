from collections import Counter

import math
from matplotlib import pyplot as plt
from numpy.ma import dot


def sum_of_squares(v) :
    """ v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

def mean(x) :
    """
    均值,mean or average
    :param x:
    :return:
    """
    return sum(x) / len(x)

def median(v) :
    """
    中位值 median,奇数个就中间点的值,偶数个就中间两个点的平均值, middle-most value
    :param v:
    :return:
    """
    n = len(v)
    sorted_v = sorted(v)
    midpoint = n // 2

    if n % 2 == 0 :
        return sorted_v[midpoint]
    else :
        lo = midpoint -1
        hi = midpoint
        return (sorted_v[lo] + sorted_v[hi]) / 2

def quantile(x, p) :
    """
    分位数(quantile), 表示少于数据中特定百分比的一个值
    :param x:
    :param p:
    :return:  the pth-percentile value in x
    """
    p_index = int(p * len(x))
    return sorted(x)[p_index]

def mode(x) :
    """
    总数(mode),它是指出现次数最多的一个或多个值
    :param x:
    :return:
    """
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == max_count]

def data_range(x):
    """
    极差(range), 最大元素与最小元素的差
    :return:
    """
    return max(x) - min(x)

def de_mean(x) :
    """
    translate x by substracting its mean ( so the result has mean 0)
    :param x:
    :return:
    """
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def variance(x) :
    """
    方差(variance), assumes x has at least two elements
    :param x:
    :return:
    """
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)

def standard_deviations(x) :
    """
    标准差(standard deviation)
    :param x:
    :return:
    """
    return math.sqrt(variance(x))

def covariance(x, y) :
    """
    协方差(covariance), 衡量两个变量对均值的串联偏离程度
    :param x:
    :param y:
    :return:
    """
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)

def correlation(x, y):
    """
    相关,有协方差除以两个变量的标准差
    :param x:
    :param y:
    :return:
    """
    stdev_x = standard_deviations(x)
    stdev_y = standard_deviations(y)
    if stdev_x > 0 and stdev_y > 0 :
        return covariance(x, y) / stdev_x / stdev_y
    else :
        return 0        # 如果没有变动, 相关系数为零

num_friends = [100, 43, 5, 4, 5,12,33,45,64,32,3,41,23, 44,3,2,32,2,3,12,23,23,21,34,32,24,53,23,43,37,37,23]

friend_counts = Counter(num_friends)

num_points = len(num_friends)
xs = range(101)
ys = [friend_counts[x] for x in xs]     # height刚好是朋友的个数

plt.bar(xs, ys)
plt.axis([0, 101, 0, 8])
plt.title("朋友数的直方图")
plt.xlabel("朋友个数")
plt.ylabel("人数")
#plt.show()

print(median(num_friends))
print(quantile(num_friends, 0.25))
print(mode(num_friends))
print(de_mean(num_friends))
print(variance(num_friends))
print(standard_deviations(num_friends))