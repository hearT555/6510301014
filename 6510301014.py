import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import numpy as np
X, y = make_blobs(n_samples=200, centers=[[2.0, 2.0], [3.0, 3.0]], 
                  cluster_std=0.75, n_features=2, random_state=42)
clf = LogisticRegression()
clf.fit(X, y)

def plot_decision_boundary(X, y, model):
    h = 0.02  # ความละเอียดของ grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.RdBu)
    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.title("Decision Plane")
    plt.legend(["Class 1", "Class 2"])
    plt.show()

plot_decision_boundary(X, y, clf)