from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from pprint import pprint

newsgroups_train = fetch_20newsgroups(subset='train')
pprint(list(newsgroups_train.target_names))

# 选取四个主题
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']

twenty_train = fetch_20newsgroups(subset='train', categories=categories)

# 文件内容在twenty_train.data里, 现在对内容进行分词和向量化操作
# X_train_counts是一个csr_matrix, 每行类似于 "(0, 230) 4",
# 其中0表示文章id, 230应该是词的号码, 4就是此文章的词频
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

# 接着对向量化之后的结果做TF-IDF转换
# 就是将 X_train_counts 归一化
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

"""
###### Rocchio 算法 start #####
from sklearn.neighbors.nearest_centroid import NearestCentroid

# 现在把TF-IDF转换后的结果和每条结果对应的主题编号twenty_train.target放入分类器中进行训练
clf = NearestCentroid().fit(X_train_tfidf, twenty_train.target)
###### Rocchio 算法 end #####

###### 朴素贝叶斯分类,其中的多项式贝叶斯分类 start #####
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
###### 朴素贝叶斯分类,其中的多项式贝叶斯分类 end #####

###### K-NN start #####
from sklearn.neighbors import KNeighborsClassifier

# 找出相似度最高的15篇文章, 计算成本高
clf = KNeighborsClassifier(15).fit(X_train_tfidf, twenty_train.target)
###### K-NN end #####
"""
###### SVM start #####
from sklearn import svm

# 使用线性支持向量分类linear, 对文章分类效果比较好, 计算开销大
clf = svm.SVC(kernel='linear').fit(X_train_tfidf, twenty_train.target)
###### SVM end #####

# 创建测试集合, 这里有两条数据,每条数据一行内容, 进行向量化和TF-IDF转换
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# 预测
predicted = clf.predict(X_new_tfidf)

# 打印结果
for doc, category in zip(docs_new, predicted) :
    print("%r => %s" % (doc, twenty_train.target_names[category]))
