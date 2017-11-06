import random
from collections import Counter
from matplotlib import pyplot as plt

from examples.data_science.gradient_descent import distance
from examples.data_science.plot_state_borders import plot_state_borders


def majority_vote(labels):
    """ assumes that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner                       # 唯一的获胜者,返回它的值
    else:
        return majority_vote(labels[:-1])   # 去掉最远元素,再次尝试

# 创建一个分类器
def knn_classify(k, labeled_points, new_point):
    """ each labeled point should be a pair (point, label) """

    # 把标记好的点按从最近到最远的顺序排序
    by_distance = sorted(labeled_points,
                         # key=lambda (point, _): distance(point, new_point))
                         # 上面的python2 的tuple拆箱方式,以下是python3的使用方式.请看"PEP 3113 -- Removal of Tuple Parameter Unpacking"
                         key=lambda labeled_point: distance([labeled_point[0], labeled_point[1]], new_point))

    # 寻找K个最近邻的标签. 因为 cities是 三个原书的list
    k_nearest_labels = [label for _, _, label in by_distance[:k]]

    # 然后让他们投票
    return majority_vote(k_nearest_labels)

def random_point(dim):
    """ 生成 随机点."""
    return [random.random() for _ in range(dim)]

def random_distances(dim, num_pairs):
    """ 随机点的生成距离 """
    return [distance(random_point(dim), random_point(dim))
            for _ in range(num_pairs)]

""" 维数灾难 : 高维空间过于巨大.
当有去多维度时,看上去最邻近的两个点的距离并不比点和点的平局距离小,
这说明两个点邻近并不特别意味着什么.
因此, 要在高维中使用最近邻法,不妨先做一些降维工作."""

""" training and testing """
# datas get from github
cities = [(-86.75,33.5666666666667,'Python'),(-88.25,30.6833333333333,'Python'),(-112.016666666667,33.4333333333333,'Java'),
         (-110.933333333333,32.1166666666667,'Java'),(-92.2333333333333,34.7333333333333,'R'),(-121.95,37.7,'R'),
         (-118.15,33.8166666666667,'Python'),(-118.233333333333,34.05,'Java'),(-122.316666666667,37.8166666666667,'R'),
         (-117.6,34.05,'Python'),(-116.533333333333,33.8166666666667,'Python'),(-121.5,38.5166666666667,'R'),
         (-117.166666666667,32.7333333333333,'R'),(-122.383333333333,37.6166666666667,'R'),(-121.933333333333,37.3666666666667,'R'),
         (-122.016666666667,36.9833333333333,'Python'),(-104.716666666667,38.8166666666667,'Python'),(-104.866666666667,39.75,'Python'),
         (-72.65,41.7333333333333,'R'),(-75.6,39.6666666666667,'Python'),(-77.0333333333333,38.85,'Python'),(-80.2666666666667,25.8,'Java'),
         (-81.3833333333333,28.55,'Java'),(-82.5333333333333,27.9666666666667,'Java'),(-84.4333333333333,33.65,'Python'),
         (-116.216666666667,43.5666666666667,'Python'),(-87.75,41.7833333333333,'Java'),(-86.2833333333333,39.7333333333333,'Java'),
         (-93.65,41.5333333333333,'Java'),(-97.4166666666667,37.65,'Java'),(-85.7333333333333,38.1833333333333,'Python'),
         (-90.25,29.9833333333333,'Java'),(-70.3166666666667,43.65,'R'),(-76.6666666666667,39.1833333333333,'R'),
         (-71.0333333333333,42.3666666666667,'R'),(-72.5333333333333,42.2,'R'),(-83.0166666666667,42.4166666666667,'Python'),
         (-84.6,42.7833333333333,'Python'),(-93.2166666666667,44.8833333333333,'Python'),(-90.0833333333333,32.3166666666667,'Java'),
         (-94.5833333333333,39.1166666666667,'Java'),(-90.3833333333333,38.75,'Python'),(-108.533333333333,45.8,'Python'),
         (-95.9,41.3,'Python'),(-115.166666666667,36.0833333333333,'Java'),(-71.4333333333333,42.9333333333333,'R'),
         (-74.1666666666667,40.7,'R'),(-106.616666666667,35.05,'Python'),(-78.7333333333333,42.9333333333333,'R'),
         (-73.9666666666667,40.7833333333333,'R'),(-80.9333333333333,35.2166666666667,'Python'),(-78.7833333333333,35.8666666666667,'Python'),
         (-100.75,46.7666666666667,'Java'),(-84.5166666666667,39.15,'Java'),(-81.85,41.4,'Java'),(-82.8833333333333,40,'Java'),
         (-97.6,35.4,'Python'),(-122.666666666667,45.5333333333333,'Python'),(-75.25,39.8833333333333,'Python'),(-80.2166666666667,40.5,'Python'),
         (-71.4333333333333,41.7333333333333,'R'),(-81.1166666666667,33.95,'R'),(-96.7333333333333,43.5666666666667,'Python'),(-90,35.05,'R'),
         (-86.6833333333333,36.1166666666667,'R'),(-97.7,30.3,'Python'),(-96.85,32.85,'Java'),(-95.35,29.9666666666667,'Java'),
         (-98.4666666666667,29.5333333333333,'Java'),(-111.966666666667,40.7666666666667,'Python'),(-73.15,44.4666666666667,'R'),
         (-77.3333333333333,37.5,'Python'),(-122.3,47.5333333333333,'Python'),(-89.3333333333333,43.1333333333333,'R'),(-104.816666666667,41.15,'Java')]

# 键是语言, 值是成对数据(longitudes, latitudes)
plots = {"Java" : ([], []), "Python" : ([], []), "R" : ([], []) }

# 每种语言都能有不同的记号和颜色
markers = {"Java" : "o", "Python" : "s", "R" : "^" }
colors = {"Java" : "red", "Python" : "blue", "R" : "lightgreen"}

"""
for longitude, latitude, language in cities:
    plots[language][0].append(longitude)
    plots[language][1].append(latitude)
"""

# 试试多个不同的k值
for k in [1, 3, 5, 7]:
    num_correct = 0

    for city in cities:
        lang_temp, lat_temp, actual_language = city
        other_cities = [other_city
                        for other_city in cities
                        if other_city != city]

        location = list([lang_temp, lat_temp])

        predicted_language = knn_classify(k, other_cities, location)

        if predicted_language == actual_language:
            num_correct += 1

    print(k, "neighbor[s]:", num_correct, "correct out of", len(cities))

k = 5 # 或3, 或5, 或......

for longitude in range(-130, -60):
    for latitude in range(20, 55):
        predicted_language = knn_classify(k, cities, [longitude, latitude])
        plots[predicted_language][0].append(longitude)
        plots[predicted_language][1].append(latitude)

    # 对每种语言创建一个散点序列
    for language, (x, y) in plots.items():
        plt.scatter(x, y, color=colors[language], marker=markers[language],
                    label=language, zorder=10)

# 打印美国地图
plot_state_borders()

plt.legend(loc=0)       # 让matplotlib选择一个位置
plt.axis([-130, -60, 20, 55])   # 设置轴

plt.title("最受欢迎的编程语言")
plt.show()


