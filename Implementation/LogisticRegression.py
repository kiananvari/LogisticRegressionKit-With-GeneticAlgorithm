import matplotlib.pyplot as plt
import numpy as np
from Data import normalize
from sklearn.preprocessing import PolynomialFeatures


class LogisticRegression:
    def __init__(self, in_dim, n_iter=1000, lr=0.1):
        self.w = np.zeros((in_dim))
        self.b = np.zeros((1))
        self.n_iter = n_iter
        self.lr = lr

    def CE_loss(self, z, y):
        # h = 1 / (1 + e^-z and
        # loss = (-y * log(h) - (1 - y) * log(1 - h)).sum() / y.shape[0]
        # better approach:
        # log(h) = log(1 / (1 + e^-z))
        #        = log(1) - log(e^0 + e^-z)
        # log(1 - h) = log(e^-z / (1 + e^-z))
        #            = -z - log(e^0 + e^-z)
        loss = (y * np.logaddexp(0, -z) + (1 - y) * (z + np.logaddexp(0, -z)))
        return loss.mean()

    def fit(self, X_train, y_train, X_test, y_test):
        losses_train = []
        losses_test = []
        for i in range(self.n_iter):
            h = self.predict(X_train)
            loss_der = h - y_train
            delta_w = self.lr * np.dot(X_train.T, loss_der) / X_train.shape[0]
            delta_b = self.lr * np.sum(loss_der) / X_train.shape[0]
            self.w -= delta_w
            self.b -= delta_b

            z_train = self.get_z(X_train)
            z_test = self.get_z(X_test)
            losses_train.append(self.CE_loss(z_train, y_train))
            losses_test.append(self.CE_loss(z_test, y_test))
        return losses_train, losses_test

    def get_z(self, X):
        z = np.dot(X, self.w) + self.b
        return z

    def predict(self, X):
        z = self.get_z(X)
        return 1 / (1 + np.exp(-z))

    def score(self, X, y):
        h = self.predict(X) > 0.5
        return (h == y).mean()

    def get_equation(self):
        if len(self.w) == 2:
            return f'{self.w[0]:.4f} x0 + {self.w[1]:.4f} x1 + {self.b[0]:.4f} = 0'
        else:
            return f'{self.w[0]:.4f} x0 + {self.w[1]:.4f} x1 + {self.w[2]:.4f} x2 + {self.b[0]:.4f} = 0'


class NonLinaerLogisticRegression(LogisticRegression):
    def __init__(self, in_dim, power=2, n_iter=1000, lr=0.1):
        self.pf = PolynomialFeatures(degree=power, include_bias=False)
        out_pf = self.pf.fit_transform(np.random.rand(1, in_dim)).shape[1]
        super().__init__(in_dim=out_pf, n_iter=n_iter, lr=lr)
        
    def predict(self, X):
        X = X.copy()
        if X.shape[1] != self.w.shape[0]:
            X = self.pf.transform(X)
        return super().predict(X)

    def fit(self, X_train, y_train, X_test, y_test):
        if X_train.shape[1] != self.w.shape[0]:
            X_train_new = self.pf.transform(X_train)
            X_test_new = self.pf.transform(X_test)
            return super().fit(X_train_new, y_train, X_test_new, y_test)
        else:
            return super().fit(X_train, y_train, X_test, y_test)

    def get_equation(self):
        names = self.pf.get_feature_names()
        out = ''
        for i in range(len(names)):
            if np.abs(self.w[i]) > 0.001:
                out = out + f'{self.w[i]:.4f} {names[i]} + '
        out = out + f'{self.b[0]:.4f} = 0'
        return out


if __name__ == '__main__':
    from Data import *
    for i in range(1, 8):
        dataset = load_data(i)
        dataset_train, dataset_test = train_test_split(dataset)

        X_train, y_train = getXY(dataset_train)
        X_test, y_test = getXY(dataset_test)

        X_train, X_test, (mean, std) = normalize(X_train, X_test)

        model = LogisticRegression(X_train.shape[1])
        losses_train, losses_test = model.fit(X_train, y_train, X_test, y_test)
        plt.plot(losses_train, 'bo-', label='Train')
        plt.plot(losses_train, 'r--', label='Test')
        plt.grid()
        plt.legend()
        plt.xlabel('Step')
        plt.ylabel('Cross Entropy')
        plt.show()
