import pandas as pd
import os
from evaluate_model import get_preds_scoring
from prepare_data import Preparator

from models.decision_tree.Tree import DecisionTreeClassifier


def decision_tree_classifier(X_tr: pd.DataFrame, X_val: pd.DataFrame,
                             y_tr: pd.Series, min_samples_leaf: int = 1,
                             max_tree_depth: int = 7) -> pd.Series:
    model = DecisionTreeClassifier(min_samples=min_samples_leaf,
                                   max_depth=max_tree_depth)
    model.fit(X_tr, y_tr)
    predictions = model.predict(X_val)
    return predictions


# test model
if __name__ == '__main__':
    path = os.path.join('data', 'cardio_train.csv')
    no_negatives = ['ap_lo', 'ap_hi']
    preparator = Preparator(path, no_negatives, ';', 'cardio')
    splitted_data = preparator.split_dataset()
    x_train, y_train, x_validate, y_validate, x_test, y_test = splitted_data
    depth = 7

    preds = decision_tree_classifier(x_train, x_test, y_train)
    conf_matrix, acc, prec, rec, f1, TNR, FPR = get_preds_scoring(y_test,
                                                                  preds)

    print(f'\n->->-> Model rating for depth {depth}<-<-<-')
    print(f'Confution matrix values (tp, fp, fn, tn): {conf_matrix}')
    print(f'Model accuracy: {acc}')
    print(f'Precision: {prec}')
    print(f'Recall: {rec}')
    print(f'F-measure: {f1}')
    print(f'FPR: {FPR}')
    print(f'TNR: {TNR}')
