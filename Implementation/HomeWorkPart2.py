import matplotlib.pyplot as plt

from Data import load_data, train_test_split, getXY, normalize, visualize_data, visualize_decision_boundary
from LogisticRegression import NonLinaerLogisticRegression

for i in range(1, 8):
    dataset = load_data(i)
    dataset_train, dataset_test = train_test_split(dataset)

    X_train, y_train = getXY(dataset_train)
    X_test, y_test = getXY(dataset_test)

    X_train, X_test, (mean, std) = normalize(X_train, X_test)

    model = NonLinaerLogisticRegression(X_train.shape[1], power=3, lr=0.5)
    losses_train, losses_test = model.fit(X_train, y_train, X_test, y_test)
    print(f'Linear LogisticRegression on dataset {i}, train accuracy: {model.score(X_train, y_train) * 100:.2f}, test accuracy: {model.score(X_test, y_test) * 100:.2f}')
    print(f'Equation of decision boundary: {model.get_equation()}')

    plt.figure()
    plt.title(f'Training results for Dataset{i}')
    plt.plot(losses_train, 'bo-', label='Train')
    plt.plot(losses_test, 'r--', label='Test')
    plt.grid()
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('Cross Entropy')

    if i < 6:
    	plt.figure()
    plt.title(f'Data and Decision boundary for Dataset{i}')
    ax = visualize_data(X_train, y_train, X_test, y_test)
    visualize_decision_boundary(model, X_train.max(axis=0), X_train.min(axis=0), ax=ax)

    plt.show()
