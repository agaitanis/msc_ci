import scipy.io
import numpy as np
from sklearn import svm, metrics, utils, decomposition
import time

mat = scipy.io.loadmat('data/mnist_all.mat')

x_train = []
y_train = []
x_test = []
y_test = []

for key, data in mat.items():
    if 'train' in key:
        for x in data:
            x_train.append(x)
            num = int(key[-1])
            y_train.append(num % 2)
    if 'test' in key:
        for x in data:
            x_test.append(x)
            num = int(key[-1])
            y_test.append(num % 2)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train, y_train = utils.shuffle(x_train, y_train, random_state=0)

x_train = x_train/255.
y_train = y_train
x_test = x_test/255.
y_test = y_test


pca = decomposition.PCA(0.9)
x_train = pca.fit_transform(x_train)
print('x_train.shape =', x_train.shape)

svm = svm.SVC(C=1.0)

t1 = time.time()
svm.fit(x_train, y_train)
t2 = time.time()
print("Training time: {:.1f} min".format((t2 - t1)/60))

x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)
print('Accuracy =', metrics.accuracy_score(y_test, y_pred))


