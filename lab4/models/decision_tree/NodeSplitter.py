from typing import Dict, Optional
import pandas as pd

from .Node import Node
from .SplitEvaluator import Evaluator, GiniImpurityCalculator


criterion_mapping: Dict[str, Evaluator] = {
    'gini': GiniImpurityCalculator()
    # more criterions can be implemented
}


class NodeSplitter:
    '''
    Class that splitts the node by the best feature
    :samples_limit: minimum number of samples in the single node
    :evaluation_criterion: string representing type of
        split evaluator (gini/entropy)
    :values_limit: limit of rows in continuous columns to make splitting
        conditions from all
    :splitting_step: number to get step for splittig long continuous data
    '''
    def __init__(self, samples_limit: int, evaluation_criterion: str,
                 numeric_values_limit: int, numeric_split_step: int):
        self.evaluator: Evaluator = criterion_mapping[evaluation_criterion]
        self.numeric_split_step = numeric_split_step
        self.numeric_values_limit = numeric_values_limit
        self.samples_limit = samples_limit

    def split(self, X: pd.DataFrame, y: pd.Series,
              feature: str, value: Optional[float]) -> tuple[Node]:
        '''
        Returns tuple of nodes splitted by given feature
        :X: parent node to split
        :y: data to predict of parent node
        :feature: feature by which to split
        :value: if the feature is numric, it's needed for the condition
        '''
        # if the split is by true/false values
        if not value:
            # get indexes for the split
            values = X[feature].unique()
            left_node_indexes = X.index[X[feature] == values[1]]
            right_node_indexes = X.index[X[feature] == values[0]]
            # get splitted data frames
            left_node_data = X.loc[left_node_indexes]
            right_node_data = X.loc[right_node_indexes]
            # get splitted series of classification class
            left_node_to_predict = y.loc[left_node_indexes]
            right_node_to_predict = y.loc[right_node_indexes]

            # no need to drop used column, because next splits using it
            # will generate not enough either right or left data,
            # so they will be ommited

        # if the split is by numeric values
        else:
            # get indexes for the split
            left_node_indexes = X.index[X[feature] <= value]
            right_node_indexes = X.index[X[feature] > value]
            # get splitted data frames
            left_node_data = X.loc[left_node_indexes]
            right_node_data = X.loc[right_node_indexes]
            # get splitted series of classification class
            left_node_to_predict = y.loc[left_node_indexes]
            right_node_to_predict = y.loc[right_node_indexes]

            # no need to drop anything - in right node there is no
            # used value and in the left one, it will generate split with
            # not enough right data, so it will be ommited

        # create nodes on splitted data
        left_node = Node(data=(left_node_data, left_node_to_predict))
        right_node = Node(data=(right_node_data, right_node_to_predict))

        return left_node, right_node

    def find_split(self, X: pd.DataFrame,
                   y: pd.Series) -> tuple[str, Optional[float]]:
        '''
        Returns feature of the split with max information gain or
        min impurity and value of the split if feature's data is numeric
        :X: data in which it finds the best split feature
        :y: this data's series of classification feature
        '''
        # initialize needed value
        the_best_evaluation = None
        the_best_value = None
        the_best_feature = ''

        # look for the best split in every feature
        for feature in X.columns:
            if X[feature].nunique() == 2:
                # true/false data
                split_bool = self._get_split_bool(X, y,
                                                  the_best_evaluation,
                                                  feature)
                if split_bool:
                    the_best_evaluation, the_best_value = split_bool
                    the_best_feature = feature
            else:
                # numeric data
                split_numeric = self._get_split_numeric(X, y,
                                                        the_best_evaluation,
                                                        feature)
                if split_numeric:
                    the_best_evaluation, the_best_value = split_numeric
                    the_best_feature = feature

        # if not feature was found, no split is possible
        if not the_best_feature:
            return None
        else:
            return the_best_feature, the_best_value

    def _get_split_bool(self, X: pd.DataFrame,
                        y: pd.Series, the_best_evaluation: float,
                        feature: str) -> Optional[tuple[float, None]]:
        '''
        Splitts given node with true/false data.
        If length of data after the split is below the limit, returns None
        If this split is not better than the other, returns None
        '''
        feature_series = X[feature]

        # split data by given feature
        left_data = y[feature_series == 1]
        right_data = y[feature_series == 0]

        # check if length of any data is below the limit
        if len(left_data) < self.samples_limit or \
                len(right_data) < self.samples_limit:
            return None

        # get evaluation of this split
        left_data_counts = left_data.value_counts()
        right_data_counts = right_data.value_counts()
        evaluation = self.evaluator.evaluate_split(left_data_counts,
                                                   right_data_counts)
        # if it's the first evaluation or it's better than the previous,
        # return it (None because there is no value for try/false split)
        if not the_best_evaluation:
            return evaluation, None
        elif self.evaluator.check_upgrade(the_best_evaluation,
                                          evaluation):
            return evaluation, None
        else:
            return None

    def _get_split_numeric(self, X: pd.DataFrame, y: pd.Series,
                           the_best_evaluation: float,
                           feature: str) -> Optional[tuple[float, float]]:
        '''
        Splits node with continuous or discreet data, considering all the possible splits.
        Uses <= condition.
        If length of data every possible split is below the limit, returns None
        If this split is not better than the other, returns None
        '''
        # sort data
        feature_series = X[feature]
        sorted_ind = feature_series.argsort()
        data_column_sorted = feature_series.iloc[sorted_ind]
        y_sorted = y.iloc[sorted_ind]

        values = data_column_sorted.unique()

        # cut values number if there are to much of them
        if len(values) > self.numeric_values_limit:
            data_step = self.numeric_split_step
            condition_step = int(len(values) / data_step)
            values = values[::condition_step]

        the_best_value = None

        for value in values:
            # for each value consider possible split
            left_data = y_sorted[feature_series <= value]
            right_data = y_sorted[feature_series > value]

            # check if length of any data is below the limit
            if len(left_data) < self.samples_limit or \
                    len(right_data) < self.samples_limit:
                continue

            # get evaluation of this split
            left_data_counts = left_data.value_counts()
            right_data_counts = right_data.value_counts()
            evaluation = self.evaluator.evaluate_split(left_data_counts,
                                                       right_data_counts)
            # if it's the first evaluation or it's better than the previous,
            # update the best evaluation
            if not the_best_evaluation:
                the_best_evaluation = evaluation
                the_best_value = value
            elif self.evaluator.check_upgrade(the_best_evaluation,
                                              evaluation):
                the_best_evaluation = evaluation
                the_best_value = value

        if the_best_value is None:
            return None
        else:
            return the_best_evaluation, the_best_value
