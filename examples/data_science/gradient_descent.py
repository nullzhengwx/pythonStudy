import random
from functools import partial

import math
from matplotlib import pyplot as plt

def vector_substract(v, w):
    """ substracts corresponding elements """
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def squard_distance(v, w):
    """ (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2 """
    return sum_of_squares(vector_substract(v, w))

def distance(v, w) :
    return math.sqrt(squard_distance(v, w))


def sum_of_squares(v):
    """ computes the sum of squared elements in v """
    return sum(v_i ** 2 for v_i in v)

def difference_quotient(f, x, h) :
    """ 商差,是通过点(x,f(x)) 和点(x+h, f(x+h))的割线的斜率 """
    return (f(x + h) - f(x)) / h

def square(x):
    return x * x

def derivative(x) :
    """ square 的导数 """
    return 2 * x

def partial_difference_quotient(f, v, i, h):
    """ compute the ith partial difference quotient of f at v """
    w = [v_j + (h if j == i else 0)     # 只对v的第i个元素增加h
         for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h

def estimate_gradient(f, v, h=0.00001) :
    return [partial_difference_quotient(f, v, i, h)
            for i, _ in enumerate(v)]

def square_derivative_estimate(x, h) :
    return (square(x + h) - square(x)) / h

def step(v, direction, step_size):
    """ move step_size in the direction from v """
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]

def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]

def safe(f) :
    """
    return a new function that's the same as f,
    except that it outputs infinity whenever f produces an error
    :param f:
    :return:
    """
    def safe_f(*args, **kwargs):
        try:
            return f(args, kwargs)
        except:
            return float('inf')     # 意思是python中的"无限值"
    return safe_f

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """ use gradient descent to find theta that minimize target function """
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    theta = theta_0                 # 设定theta为初始值
    target_fn = safe(target_fn)     # target_fn的安全版
    value = target_fn(theta)        # 我们试图最小化的值

    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
                       for step_size in step_sizes]

        # 选择一个使残差函数最小的值
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)

        # 当"收敛"时停止
        if abs(value - next_value) < tolerance :
            return theta
        else :
            thera, value = next_theta, next_value

""" testing """

derivative_estimate = partial(difference_quotient, square, h=0.00001)

# 绘出导入matplotlib.pyplot作为plt的基本相同的形态
"""
x = range(-10, 10)
plt.title("精确的导数值与估计值")
plt.plot(x, map(derivative, x), 'rx', lable='Actual')
plt.plot(x, map(derivative_estimate, x), 'b+', label='Estimate')
plt.legend(loc=9)
plt.show()
"""

# 选取一个随机初始值
v = [random.randint(-10, 10) for i in range(3)]

tolerance = 0.0000001

while True:
    gradient = sum_of_squares_gradient(v)   # 计算v的梯度
    next_v = step(v, gradient, -0.01)       # 取负的梯度步长
    if distance(next_v, v) < tolerance:     # 如果收敛了就停止
        break
    v = next_v                              # 如果没汇合就继续