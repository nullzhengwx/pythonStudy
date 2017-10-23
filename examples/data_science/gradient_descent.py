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
    """ use gradient descent to find theta that minimize target function
        梯度下降法
    """
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    theta = theta_0                 # 设定theta为初始值
    target_fn = safe(target_fn)     # target_fn的安全版
    value = target_fn(theta)        # 我们试图最小化的值

    while True:
        gradient = gradient_fn(theta)
        # 参数向斜率的方法行了特定的步长
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

def negate(f):
    """ return a function that for any input x returns -f(x) """
    return lambda *args, **kwargs: -f(*args, **kwargs)

def negate_all(f):
    """ the same when f returns a list of numbers """
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """ 最大化某个函数,只需要最小化这个函数的负数, 当然梯度函数也需取负"""
    return minimize_batch(negate(target_fn), negate_all(gradient_fn),
                          theta_0, tolerance)

def in_random_order(data):
    """ generator that returns the elements of data in random order """
    indexes = [i for i, _ in enumerate(data)]   # 生成索引列表
    random.shuffle(indexes)                     # 随机打乱数据
    for i in indexes:                           # 返回序列中的数据
        yield data[i]

def scalar_multiply(c, v):
    """ c is a number, v is a vector """
    return [c * v_i for v_i in v]

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    """ 随机梯度下降法, (stochastic gradient descent), 它每次仅计算一个点的梯度(并向前跨一步)
        这种方法留有一种可能性, 即也许会在最小值附近一直循环下去, 所以, 每当停止获得改进,都会减小步长并最终退出
    """

    data = zip(x,y)
    theta = theta_0                             # 初始值猜测
    alpha = alpha_0                             # 初始步长
    min_theta, min_value = None, float("inf")   # 迄今为止的最小值
    iterations_with_no_improvement = 0

    # 如果循环超过100次仍无改进,停止
    while iterations_with_no_improvement < 100:
        value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in data)

        if value < min_value:
            # 如果找到新的最小值, 记住它
            # 并返回到最初的步长
            min_value, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # 尝试缩小步长,否则没有改进
            iterations_with_no_improvement += 1
            alpha *= 0.9

        # 在每个数据点上向梯度方向前进一步:
        for x_i, y_i in in_random_order(data) :
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_substract(theta, scalar_multiply(alpha, gradient_i))

def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    return minimize_stochastic(negate(target_fn), negate_all(gradient_fn),
                               x, y, theta_0, alpha_0)

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