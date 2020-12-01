import scipy.io
import numpy as np
import sklearn
import time

def load_mnist():
    mat = scipy.io.loadmat('data/mnist_all.mat')
    x_train, y_train, x_test, y_test = [], [], [], []
    
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
                
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


x_train, y_train, x_test, y_test = load_mnist()

scaler = sklearn.preprocessing.MinMaxScaler()
pca = sklearn.decomposition.PCA(0.9)
svm = sklearn.svm.SVC(C=1.0)

pipe = sklearn.pipeline.Pipeline([
    ('scaler', scaler),
    ('pca', pca),
    ('svm', svm)
    ])

t1 = time.time()
pipe.fit(x_train, y_train)
t2 = time.time()

print("Training time: {:.1f} min".format((t2 - t1)/60))

y_pred = pipe.predict(x_test)
print('Accuracy =', sklearn.metrics.accuracy_score(y_test, y_pred))


