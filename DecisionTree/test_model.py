import matplotlib.pyplot as plt
import os

from models.decision_tree.Tree import DecisionTreeClassifier
from prepare_data import Preparator
from evaluate_model import get_preds_scoring


if __name__ == '__main__':
    max_depth = 20
    min_depth = 1

    # get datasets
    path = os.path.join('data', 'cardio_train.csv')
    no_negatives = ['ap_lo', 'ap_hi']
    preparator = Preparator(path, no_negatives, ';', 'cardio')
    splitted_data = preparator.split_dataset()
    x_train, y_train, x_validate, y_validate, x_test, y_test = splitted_data

    train_accurancies = []
    validate_accurancies = []

    for depth in range(min_depth, max_depth):
        model = DecisionTreeClassifier(max_depth=depth)
        # train model with given depth
        model.fit(x_train, y_train)

        # get predictions for training dataset and validating dataset
        train_dataset_predictions = model.predict(x_train)
        validate_dataset_predictions = model.predict(x_validate)

        train_results = get_preds_scoring(y_train,
                                          train_dataset_predictions)
        validate_results = get_preds_scoring(y_validate,
                                             validate_dataset_predictions)

        train_accurancies.append(train_results[1])
        validate_accurancies.append(validate_results[1])

    # Plot results
    plt.plot([x for x in range(min_depth, max_depth)],
             train_accurancies, label='Train Accuracy')
    plt.plot([x for x in range(min_depth, max_depth)],
             validate_accurancies, label='Validate Accuracy')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Depth')
    plt.legend()
    plt.show()
