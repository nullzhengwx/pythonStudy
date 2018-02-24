import warnings

from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression

# added version check for recent scikit-learn 0.18 checks
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version

# SVM
from sklearn.svm import SVC

# DecisionTree
from sklearn.tree import DecisionTreeClassifier

def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    """ a small convenience function to visualize the
        decision boundaries for 2D datasets
    """
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # np.unique() Find the unique elements of an array
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolors='black',
                    marker=markers[idx],
                    label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[len(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to Numpy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolors='black',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')

def show_plt(plt, param='standardized'):
    plt.xlabel('petal length [%s]' % param)
    plt.ylabel('petal width [%s]' % param)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def cost_1(z):
    return - np.log(sigmoid(z))

def cost_0(z):
    return - np.log(1 - sigmoid(z))


""" training and testing """

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#print(X_train_std)

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

"""
plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=ppn,
                      test_idx=range(105, 150))
show_plt(plt)


z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
# y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)
plt.show()
"""

"""
z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)

c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label='J(w) if y=1')

c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')

plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc="best")
plt.tight_layout()
plt.show()
"""

"""
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
show_plt(plt)

if Version(sklearn_version) < "0.17":
    print(lr.predict_proba(X_test_std[0, :]))
else:
    print(lr.predict_proba(X_test_std[0, :].reshape(1, -1)))
"""

"""
weights, params = [], []
for c in np.arange(-5., 5.):
    lr = LogisticRegression(C=10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
plt.plot(params,
         weights[:, 0],
         label='petal length')
plt.plot(params,
         weights[:, 1],
         linestyle='--',
         label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()
"""

"""
svm = SVC(kernel='linear', C=0.01, random_state=0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx=range(105, 150))

show_plt(plt)
"""

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

"""
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            #c = 'b',
            marker='x',
            label='1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c = 'g',
            marker='s',
            label='-1')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.tight_layout()
plt.show()
"""

"""
gamma会增减受影响的训练样本的范围,这将导致决策边界的变形.
而C值是影响模型的严厉程度.
两者都是越大越往过拟合方面发展.
"""
"""
svm = SVC(kernel='rbf', random_state=0, gamma=10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
"""

"""
svm = SVC(kernel='rbf', random_state=0, gamma=100, C=10.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
show_plt(plt)
"""

'''
将特征空间进行矩形划分的方式来构建复杂的决赛边界.
criterion参数是决定不纯度量度标准,暂时找到两个:'gini'和'entropy',误分类率找不到相应的参数
max_depth表示树的最大深度,越大就越精确,但可能会过拟合.到一定的深度就不会变,可以理解为树已经定型了,不能再分了.
虽然特征缩放是出于可视化的目的, 但在决策树算法中,这不是必须的,就是说不用将特征standardize
'''
'''
tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
plot_decision_regions(X_combined, y_combined,
                      classifier=tree, test_idx=range(105, 150))

show_plt(plt, 'cm')
'''

from sklearn.tree import export_graphviz
"""
export_graphviz(tree,
                out_file='/home/zhenmie/Documents/ml/pics/iris_tree_5.dot',
                feature_names=['petal length', 'petal width'])
"""

"""
from IPython.display import Image
from IPython.display import display

if Version(sklearn_version) >= '0.18':
    try:
        import pydotplus

        dot_data = export_graphviz(
        tree,
        out_file='/home/zhenmie/Documents/ml/pics/iris_tree_color.dot',
        # the parameters below are new in sklearn 0.18
        feature_names=['petal length', 'petal width'],
        class_names=['setosa', 'versicolor', 'virginica'],
        filled=True,
        rounded=True)

        graph = pydotplus.graph_from_dot_data(dot_data)
        display(Image(graph.create_png()))

    except ImportError:
        print('pydotplus is not installed.')
"""

from sklearn.ensemble import RandomForestClassifier

"""
n_estimators为bootstrap抽样的数量,一般与原始训练样本的数量相同.
这个数量越大,随机森林整体的分类表现就越好,但同时计算成本就越高.
"""

"""
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=50,
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
plot_decision_regions(X_combined, y_combined,
                      classifier=forest, test_idx=range(105, 150))

show_plt(plt, 'cm')
"""