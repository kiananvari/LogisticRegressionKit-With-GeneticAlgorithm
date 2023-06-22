# pip install pygad
import pygad
import matplotlib.pyplot as plt
import time

from Data import load_data, train_test_split, getXY, normalize, visualize_data, visualize_decision_boundary
from LogisticRegression import NonLinaerLogisticRegression


def decode_gene(gene):
    power = 1
    n_iter = 1
    log2lr = 0
    for i in range(3):
        power += gene[i] * (2 ** i)
    for i in range(10):
        n_iter += gene[i + 3] * (2 ** (i + 4))
    for i in range(4):
        log2lr += gene[i + 13] * (2 ** i)
    lr = 2 ** -log2lr

    return int(power), int(n_iter), lr


def fitness_function(ga_instance, solution, solution_idx):
    global X_train, y_train, X_test, y_test

    power, n_iter, lr = decode_gene(solution)
    model = NonLinaerLogisticRegression(X_train.shape[1], power=int(power), n_iter=int(n_iter), lr=lr)
    losses_train, losses_test = model.fit(X_train, y_train, X_test, y_test)

    losses_test[-1] = losses_test[-1]

    return 1 / losses_test[-1]


for i in range(1, 8):
    dataset = load_data(i)
    dataset_train, dataset_test = train_test_split(dataset)

    X_train, y_train = getXY(dataset_train)
    X_test, y_test = getXY(dataset_test)

    X_train, X_test, (mean, std) = normalize(X_train, X_test)

    # params = [power(0~7), n_iter(0~16,384), -log2lr(0~16)] 2^-16 ~= 1e-5
    # genes = [power4, power2, power1,
    #          niter8192, niter4096, niter2048, niter1024, niter512, niter256,
    #          niter128, niter64, niter32, niter16,
    #          log2lr8, log2lr4, log2lr2, log2lr1]

    ga_instance = pygad.GA(num_generations=5,  # Number of generations.
                           num_parents_mating=5,  # Number of parents in each generation.
                           sol_per_pop=10,  # total population size.
                           fitness_func=fitness_function,
                           num_genes=17,  # number of genes in each solution.
                           gene_space=[0, 1],
                           save_solutions=True
                           )
    ga_instance.run()
    ga_instance.plot_fitness()
    ga_instance.plot_result()
    ga_instance.plot_genes()
    ga_instance.plot_new_solution_rate()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    power, n_iter, lr = decode_gene(solution)
    print(f"Best chromosome: power={power}, n_iter={n_iter}, lr={lr}")
    # print(power, n_iter, lr)

    model = NonLinaerLogisticRegression(X_train.shape[1], power=int(power), n_iter=int(n_iter), lr=lr)
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
