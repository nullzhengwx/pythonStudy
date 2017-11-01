import datetime
import math
import random
from collections import Counter
from collections import defaultdict
from functools import partial

from matplotlib import pyplot as plt
from numpy import shape, mean, dot

from examples.data_science.gradient_descent import distance, maximize_batch, maximize_stochastic, scalar_multiply
from examples.data_science.propobility import inverse_normal_cdf
from examples.data_science.statistics import standard_deviation, magnitude, vector_sum, vector_subtract


def get_row(A, i):
    return A[i]     # A[i]是第i行

def get_column(A, j):
    return [A_i[j]              # 第A_i行的第j个元素
            for A_i in A]       # 对每个A_i行

def bucketize(point, bucket_size):
    """ floor the point to the next lower multiple of bucket_size"""
    return bucket_size * math.floor(point / bucket_size)

def make_histogram(points, bucket_size) :
    """ buckets the points and counts how many in each bucket """
    return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points, bucket_size, title="") :
    histogram = make_histogram(points, bucket_size)
    # python3改变了dict.keys,返回的是dict_keys对象,支持iterable,但不支持indexable.所以要加上list转换
    plt.bar(list(histogram.keys()), histogram.values(), width=bucket_size)
    plt.title(title)
    plt.show()

def parse_row(input_row, parsers):
    """ given a list of parsers (some of which may be None)
    apply the appropriate one of each element of the input_row"""

    return [try_or_none(parser)(value) if parser is not None else value
            for value, parser in zip(input_row, parsers)]

def parse_rows_with(reader, parsers):
    """wrap a reader to apply the parsers to each of its rows"""
    for row in reader:
        yield parse_row(row, parsers)

def try_or_none(f) :
    """ wraps f to return None if f raises an exception
    assumes f takes only one input"""
    def f_or_none(x):
        try: return f(x)
        except: return None

    return f_or_none

def try_parse_field(field_name, value, parser_dict):
    """ try to parse value using the appropriate funciton from parser_dict"""
    parser = parser_dict.get(field_name)    # 如果没有此条目, 则为None
    if parser is not None:
        return try_or_none(parser)(value)
    else:
        return value

def parse_dict(input_dict, parser_dict):
    return {field_name : try_parse_field(field_name, value, parser_dict)
            for field_name, value in input_dict.iteritiems() }

def picker(field_name):
    """ returns a function that picks a field out fo dict """
    return lambda row : row[field_name]

def pluck(field_name, rows):
    """ turn a list of dicts into the list of field_name values """
    return map(picker(field_name), rows)

def group_by(grouper, rows, value_transform=None):
    # 键是分组情况的输出, 值是行的列表
    grouped = defaultdict(list)
    for row in rows:
        grouped[grouper(row)].append(row)

    if value_transform is None:
        return grouped
    else:
        return {key : value_transform(rows)
                for key, rows in grouped.items() }

def percent_price_change(yesterday, today):
    return today["closing_price"] / yesterday["closing_price"] - 1

def day_over_day_changes(grouped_rows):
    # 按日期进行排序
    ordered = sorted(grouped_rows, key=picker("date"))

    # 对偏移量应用zip函数得到连续两天的成对表示
    return [{ "symbol" : today["symbol"],
              "date" : today["date"],
              "change" : percent_price_change(yesterday, today) }
            for yesterday, today in zip(ordered, ordered[1:])]

def scale(data_matrix):
    """ returns the means and standard deviation of each column"""
    num_rows, num_cols = shape(data_matrix)
    means = [mean(get_column(data_matrix, j))
             for j in range(num_cols)]
    stdevs = [standard_deviation(get_column(data_matrix, j))
              for j in range(num_cols)]

def rescale(data_matrix):
    """ rescales the input data so that each column
    has mean 0 and standard deviation 1
    leaves alone columns with no deviation"""

    means, stdevs = scale(data_matrix)

    def rescaled(i, j):
        if stdevs[j] > 0:
            return (data_matrix[i][j] -means[j]) / stdevs[j]
        else:
            return data_matrix[i][j]

    num_rows, num_cols = shape(data_matrix)
    return make_matrix(num_rows, num_cols, rescale)

def make_matrix(num_rows, num_cols, entry_fn):
    """ return a num_rows x num_cols matrix
    whose (i,j)th entry is entry_fn(i,j)"""
    return [[entry_fn(i,j)                  # 根据i创建一个列表
             for j in range(num_cols)]      # [entry_fn(i,0), ....]
            for i in range(num_rows)]       # 为每一个i创建一个列表

# 下面是降维的一个例子
""" 主成分分析: principal component analysis (PCA)
以下是scikit-learn 包的api注解地址:
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition"""
def de_mean_matrix(A):
    """ returns the result of subtracting from every value in A the mean
    value of its column. the resulting matrix has mean 0 in every column"""
    nr, nc = shape(A)
    column_means, _ = scale(A)
    return make_matrix(nr, nc, lambda i, j: A[i][j] - column_means[j])

def direction(w):
    """ 获得绝对值为1 的向量"""
    mag = magnitude(w)
    return [w_i /mag for w_i in w]

def directional_variance_i(x_i, w):
    """某一行的方差, 矩阵每行x在方向w的扩展是点积dot(x,d)
    the variance of the row x_i in the direction determined by w"""
    return dot(x_i, direction(w)) ** 2

def directional_variance(X, w):
    """ the variance of the data in the direction determined by w"""
    return sum(directional_variance_i(x_i, w)
               for x_i in X)

def directional_variance_gradient_i(x_i, w):
    """ the contribution of row x_i to the gradient of
    the direction-w variance
    f(x)**2 的遍导为 2 * f(x) * x """
    projection_length = dot(x_i, direction(w))
    return [2 * projection_length * x_ij for x_ij in x_i]

def directional_variance_gradient(X, w):
    return vector_sum(directional_variance_gradient_i(x_i, w)
                      for x_i in X)

# 下面两个是获取第一主成分方向
def first_principal_component(X):
    """ 用 梯度下降法 获取 到最大化的方向"""
    guess = [1 for _ in X[0]]
    unscaled_maximizer =  maximize_batch(
        partial(directional_variance, X),               # 现在是X的一个函数
        partial(directional_variance_gradient, X),      # 现在是X的一个函数
        guess)
    return direction(unscaled_maximizer)

def first_principal_component_sgd(X):
    """ 用随机梯度下降法
    这里没有"y",所以仅仅是传递一个Nones的向量和忽略这个输入的函数"""
    guess = [1 for _ in X[0]]
    unscaled_maximizer = maximize_stochastic(
        lambda x, _, w: directional_variance_i(x, w),
        lambda x, _, w: directional_variance_gradient_i(x, w),
        X,
        [None for _ in X],      # 假的"y"
        guess)
    return direction(unscaled_maximizer)

# 一旦找到第一主成分的方向,就可以将数据在这个方向上投影得到这个成分的值
def project(v, w):
    """ return the projection of v onto the direction w"""
    projection_length = dot(v, w)
    return scalar_multiply(projection_length, w)

# 如果还想得到其他的成分, 就要先从数据中移除投影
def remove_projection_from_vector(v, w):
    """ projects v onto w and subtracts the result from v"""
    return vector_subtract(v, project(v, w))

def remove_projection(X, w):
    """ for each row of X
    projects the row onto w, and subtracts the result from the row"""
    return [remove_projection_from_vector(x_i, w) for x_i in X]

def principal_component_analysis(X, num_components):
    components = []
    for _ in range(num_components):
        component = first_principal_component(X)
        components.append(component)
        X = remove_projection(X, component)

    return components

# 然后再将原数据转换为由主成分生成的低维空间中的点
def transform_vector(v, components):
    return [dot(v, w) for w in components]

def transform(X, components):
    return [transform_vector(x_i, components) for x_i in X]

""" trainning and testing """
random.seed(0)

# -100到100之间均匀抽取
uniform = [200 * random.random() - 100 for _ in range(10000)]

# 均值为0标准差为57的正态分布
normal = [57 * inverse_normal_cdf(random.random())
          for _ in range(10000)]

#plot_histogram(uniform, 10, "均匀分布的直方图")

#plot_histogram(normal, 10, "正态分布的直方图")

data = [
    {'closing_price': 102.25,
     'data': datetime.datetime(2014, 8, 29, 0, 0),
     'symbol': 'AAPL'},
    {'closing_price': 92.29,
     'data': datetime.datetime(2014, 8, 27, 0, 0),
     'symbol': 'CCPL'},
    {'closing_price': 122.25,
     'data': datetime.datetime(2014, 8, 16, 0, 0),
     'symbol': 'AAPL'},
    {'closing_price': 10,
     'data': datetime.datetime(2014, 8, 29, 0, 0),
     'symbol': 'BBPL'}
]

by_symbol = defaultdict(list)
for row in data:
    by_symbol[row["symbol"]].append(row)

max_price_by_symbol = { symbol : max(row["closing_price"]
                                     for row in grouped_rows)
                        for symbol, grouped_rows in by_symbol.items()}
print(max_price_by_symbol)

# 函数方式来创建.
max_price_by_symbol_def = group_by(picker("symbol"), data,
                                   lambda rows: max(pluck("closing_price", rows)))
print(max_price_by_symbol_def)

"""
# 键是股票代码, 值是一个"change"字典的列表
changes_by_symbol = group_by(picker("symbol"), data, day_over_day_changes)

# 收集所有"change"字典放入一个大列表中
all_changes = [change
               for changes in changes_by_symbol.values()
               for change in changes]

max(all_changes,key=picker("change"))
"""

list1 = [1,2,3,45,65,4,34,7]

print([{first :second}
       for first, second in zip(list1, list1[1:])])

print(distance([63, 150], [67, 160]))
print(distance([63, 150], [70, 171]))
print(distance([67, 160], [70, 171]))
