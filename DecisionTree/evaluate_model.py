import pandas as pd


def confusion_matrix_values(y_true: pd.Series,
                            y_pred: pd.Series) -> tuple[int, int, int, int]:
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for index, value in y_pred.items():
        if value == y_true.iloc[index]:
            if value == 1:
                tp += 1
            else:
                tn += 1
        else:
            if value == 1:
                fp += 1
            else:
                fn += 1

    return tp, fp, fn, tn


def accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    tp, fp, fn, tn = confusion_matrix_values(y_true, y_pred)
    return (tp + tn) / (tp + fp + fn + tn)


def precision(y_true, y_pred) -> float:
    tp, fp, fn, tn = confusion_matrix_values(y_true, y_pred)
    return tp / (tp + fp)


def recall(y_true, y_pred):
    tp, fp, fn, tn = confusion_matrix_values(y_true, y_pred)
    return tp / (tp + fn)


def f_measure(y_true, y_pred):
    tp, fp, fn, tn = confusion_matrix_values(y_true, y_pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def tnr(y_true, y_pred):
    tp, fp, fn, tn = confusion_matrix_values(y_true, y_pred)
    return tn / (tn + fp)


def fpr(y_true, y_pred):
    tp, fp, fn, tn = confusion_matrix_values(y_true, y_pred)
    return fp / (fp + tn)


def get_preds_scoring(y: pd.Series, preds: pd.Series):
    conf_matrix = confusion_matrix_values(y, preds)
    acc = accuracy(y, preds)
    prec = precision(y, preds)
    rec = recall(y, preds)
    f1 = f_measure(y, preds)
    TNR = tnr(y, preds)
    FPR = fpr(y, preds)
    return conf_matrix, acc, prec, rec, f1, TNR, FPR
