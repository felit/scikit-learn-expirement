# -*- coding:utf8 -*-
from sklearn import datasets
import numpy as np
from sklearn import svm

iris = datasets.load_iris()
print iris

print iris.data.shape
print iris.target.shape

print np.unique(iris.target)

digits = datasets.load_digits()
print digits.images.shape
import pylab as pl

pl.imshow(digits.images[0], cmap=pl.cm.gray_r)
# pl.show()
data = digits.images.reshape((digits.images.shape[0], -1))
print data

# 学习与预测


clf = svm.LinearSVC()
clf.fit(iris.data, iris.target)
clf.predict([[5.0, 3.6, 1.3, 0.25]])
print clf.coef_
# 分类
# K最近邻(KNN)分类器
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data, iris.target)
print knn.predict([[0.1, 0.2, 0.3, 0.4]])
# 训练集和测试集
perm = np.random.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]
print knn.fit(iris.data[:100], iris.target[:100])
print knn.score(iris.data[:100], iris.target[:100])
# 分类支持向量机(SVMs)
# 线性支持向量机
svc = svm.SVC(kernel='linear')
print svc.fit(iris.data, iris.target)

from sklearn import cluster, datasets

# k_means = cluster.kmeans(3)
# k_means.fit(iris.data)
# print k_means.labels_[::10]
# print k_means.target[::10]



