import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(dataset_number=1):
    if dataset_number == 1:
        dataset = 'Datasets/Data1-blobs-2D.csv'
    elif dataset_number == 2:
        dataset = 'Datasets/Data2-circles-2D.csv'
    elif dataset_number == 3:
        dataset = 'Datasets/Data3-classification-2D.csv'
    elif dataset_number == 4:
        dataset = 'Datasets/Data4-gaussian-2D.csv'
    elif dataset_number == 5:
        dataset = 'Datasets/Data5-moon-2D.csv'
    elif dataset_number == 6:
        dataset = 'Datasets/Data6-classification-3D.csv'
    elif dataset_number == 7:
        dataset = 'Datasets/Data7-gaussian-3D.csv'
    else:
        return 0

    data = pd.read_csv(dataset)

    return data


def train_test_split(dataset, split=0.2):
    msk = np.random.rand(len(dataset)) < split
    dataset_test = dataset[msk]
    dataset_train = dataset[~msk]

    return dataset_train, dataset_test


def getXY(dataset):
    y = dataset['target'].to_numpy()
    X = dataset.drop('target', axis=1).to_numpy()
    return X, y


def normalize(X_train, X_test):
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std, (mean, std)
    # min, max = X_train.min(axis=0), X_train.max(axis=0)
    # return (X_train - min) / (max - min), (X_test - min) / (max - min), (min, max)


def visualize_data(X_train, y_train, X_test, y_test):
    if X_train.shape[1] == 2:
        # 2D plot:
        plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='#f39c36', alpha=0.7, label='Train 0')
        plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='#1f83c2', alpha=0.7, label='Train 1')

        plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='#f39c36', label='Test 0', edgecolors='k')
        plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='#1f83c2', label='Test 1', edgecolors='k')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid()
        plt.legend()
    else:
        # 3D plot
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], X_train[y_train == 0, 2], color='#f39c36', alpha=0.7, label='Train 0')
        ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], X_train[y_train == 1, 2], color='#1f83c2', alpha=0.7, label='Train 1')

        ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], X_test[y_test == 0, 2], color='#f39c36', label='Test 0', edgecolors='k')
        ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], X_test[y_test == 1, 2], color='#1f83c2', label='Test 1', edgecolors='k')

        ax.grid()
        ax.legend()

        return ax


def visualize_decision_boundary(model, max_lim=None, min_lim=None, resolution=200, ax=None):
    if min_lim is None:
        min_lim = [-1, -1]
    if max_lim is None:
        max_lim = [1, 1]

    if max_lim.shape[0] == 2:
        # 2D plot:
        x = np.linspace(min_lim[0], max_lim[0], resolution)
        y = np.linspace(min_lim[1], max_lim[1], resolution)
        x, y = np.meshgrid(x, y)
        plane = np.stack([x, y], axis=-1).reshape(-1, 2)
        h = (model.predict(plane) > 0.5).reshape(x.shape)
        plt.contour(x, y, h, 0)
    else:
        # 3D plot
        x = np.linspace(min_lim[0], max_lim[0], resolution)
        y = np.linspace(min_lim[1], max_lim[1], resolution)
        z = np.linspace(min_lim[2], max_lim[2], resolution)
        x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z)
        plane = np.stack([x_mesh, y_mesh, z_mesh], axis=-1).reshape(-1, 3)
        h = model.predict(plane).reshape(x_mesh.shape)
        z_abs = np.abs(h - 0.5)
        z_low_index = z_abs.argmin(axis=2)
        z_low = z_abs.min(axis=2)
        func = z[z_low_index]
        func[z_low > 0.01] = np.nan

        # ax.plot_surface(x_mesh[:, :, 0], y_mesh[:, :, 0], func)
        ax.scatter(x_mesh[:, :, 0], y_mesh[:, :, 0], func, color='k', marker='.')


if __name__ == '__main__':
    for i in range(1, 8):
        dataset = load_data(i)
        dataset_train, dataset_test = train_test_split(dataset)

        X_train, y_train = getXY(dataset_train)
        X_test, y_test = getXY(dataset_test)

        X_train, X_test, (mean, std) = normalize(X_train, X_test)

        visualize_data(X_train, y_train, X_test, y_test)
        plt.show()
