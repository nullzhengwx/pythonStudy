import math
import random
from collections import Counter

from matplotlib import pyplot as plt

def uniform_pdf(x):
    """
    uniform probability density function
    :param x:
    :return:
    """
    return 1 if x >= 0 and x < 1 else 0

def uniform_cdf(x) :
    """
    returns the probability that a uniform random variable is <= x
    :param x:
    :return:
    """
    if x < 0 : return 0     # 均匀分布的随机变量不会小于0
    elif x < 1 : return x   # e.g. P(X <= 0.4) = 0.4
    else :  return 1        # 均匀分布的随机变量总是小于1

def normal_pdf(x, mu=0, sigma=1) :
    """
    正太分布
    :param x:
    :param mu:  均值
    :param sigma:   标准差
    :return:
    """
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x - mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))

def normal_cdf(x , mu=0, sigma=1) :
    """
    normal cumulative density function, 标准正太分布的累积分布函数
    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001) :
    """
    对normal_cdf取逆. find approximate inverse using binary search
    :param p:
    :param mu:
    :param sigma:
    :param tolerance:
    :return:
    """
    # 如果非标准型,先调整单位使之服从标准型
    if mu != 0  or sigma != 1 :
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z, low_p = -10.0, 0     # mornal_cdf(-10)是(非常接近)0
    hi_z, hi_p = 10, 1          # mornal_cdf(10)是(非常接近)1

    while hi_z - low_z > tolerance :
        mid_z = (low_z + hi_z) / 2  # 考虑中点
        mid_p = normal_cdf(mid_z)   # 和cdf在那里的值

        if mid_p < p :
            # midpoint 仍然太低, 搜索比它大的值
            low_z, low_p = mid_z, mid_p
        elif mid_p > p :
            # midpoint 仍然太高, 搜索比它小的值
            hi_z, hi_p = mid_z, mid_p
        else :
            break

    return mid_z

def bernoulli_trial(p) :
    """
    伯努利随机变量
    :param p:
    :return:
    """
    return 1 if random.random() < p else 0

def biomial(n, p) :
    """
    二项式随机变量, n个伯努利随机变量之和
    :param n:
    :param p:
    :return:
    """
    return sum(bernoulli_trial(p) for _ in range(n))

def make_hist(p, n, num_points) :
    """
    二项分布与正态近似
    :param p:
    :param n:
    :param num_points:
    :return:
    """
    data = [biomial(n, p) for _ in range(num_points)]

    # 用条形图绘出实际的二项式样本
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v /num_points for v in histogram.values()],
            0.8, color='0.75')

    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))

    # 用线性图绘出正态近似
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)
          for i in xs]
    plt.plot(xs, ys)
    plt.title("二项分布与正态近似")
    plt.show()

""" examples """
xs = [x / 10.0 for x in range(-50, 50)]

"""
plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-', label="mu=0, sigma=1")
plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '--', label="mu=0, sigma=2")
plt.plot(xs, [normal_pdf(x, sigma=0.5) for x in xs], ':', label="mu=0, sigma=0.5")
plt.plot(xs, [normal_pdf(x, mu=1) for x in xs], '-.', label="mu=1, sigma=1")
plt.legend()    # 左上角会有图示
plt.title("多个正太分布的概率密度函数")
plt.show()

plt.plot(xs, [normal_cdf(x, sigma=1) for x in xs], '-', label="mu=0, sigma=1")
plt.plot(xs, [normal_cdf(x, sigma=2) for x in xs], '--', label="mu=0, sigma=2")
plt.plot(xs, [normal_cdf(x, sigma=0.5) for x in xs], ':', label="mu=0, sigma=0.5")
plt.plot(xs, [normal_cdf(x, mu=1) for x in xs], '-.', label="mu=1, sigma=1")
plt.legend(loc=4)   # 底部右边
plt.title("多个正太分布的累积分布函数")
plt.show()
"""

# print(normal_cdf(0.5))
# print(inverse_normal_cdf(0.69146))

# make_hist(0.75, 100, 10000)
