import numpy as np
#from sklearn.model_selection import train_test_split
from NaiveBayesClassificator import NaiveBayes





def split_dataset(data, target):
    index_validate = int(0.7 * len(target))

    x_train = data[:index_validate]
    y_train = target[:index_validate]

    x_validate = data[index_validate:]
    y_validate = target[index_validate:]

    return x_train, y_train, x_validate, y_validate


def evaluate_standard_split(data, target):
    model = NaiveBayes()

    training_data, training_target,validation_data, validation_target = split_dataset(data, target)
    model.fit(training_data, training_target)

    predictions = model.predict(validation_data)
    return evaluate_model(predictions, validation_target)


def evaluate_k_cross_validation(data, target, k):
    model = NaiveBayes()
    accurancies = np.zeros(k)

    # split dataset on k subsets
    splitted_data = np.array_split(data, k)
    splitted_target = np.array_split(target, k)

    # each one of the subsets can be a validation set
    for i in range(len(splitted_data)):
        # split data
        validation_data = splitted_data[i]
        validation_target = splitted_target[i]
        training_data = [np.array(single_data) for indx, single_data in
                         enumerate(splitted_data) if indx != i]
        training_target = [np.array(single_data) for indx, single_data in
                           enumerate(splitted_target) if indx != i]
        training_data = np.concatenate(training_data)
        training_target = np.concatenate(training_target)

        # train model
        model.fit(training_data, training_target)

        # predict
        predictions = model.predict(validation_data)
        accurancies[i] = evaluate_model(predictions, validation_target)

    return np.mean(accurancies)


# def split_dataset(X: np.ndarray, Y: np.ndarray):
#     index_validate = int(0.7 * len(Y))
#     index_test = int(0.85 * len(Y))

#     x_train = X[:index_validate]
#     y_train = Y[:index_validate]

#     x_validate = X[index_validate:index_test]
#     y_validate = Y[index_validate:index_test]

#     x_test = X[index_test:]
#     y_test = Y[index_test:]

#     return x_train, y_train, x_validate, y_validate, x_test, y_test
