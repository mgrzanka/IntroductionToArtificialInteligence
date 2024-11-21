from queue import Queue
import pandas as pd

from .Node import Node
from .NodeSplitter import NodeSplitter
from ..solver import Solver


class Tree:
    def __init__(self, min_samples: int, max_depth: int,
                 numeric_values_limit: int, numeric_split_step: int,
                 evaluation_criterion: str):
        '''
        Class representing a DecisionTree model
        :max_depth: number of child nodes that can be generated
        :evaluator: string representing type of split evaluator (gini/entropy)
        :values_limit: limit of rows in continuous columns to make
            splitting conditions from all
        :splitting_step: number to get step for splittig long continuous data
        :samples_limit: minimum number of samples in the single node
        '''
        self.max_depth = max_depth
        self.splitter = NodeSplitter(min_samples,
                                     evaluation_criterion,
                                     numeric_values_limit,
                                     numeric_split_step)

    def check_if_leaf(self, node: Node):
        '''
        Node is considered a leaf if it has all 1 or all 0
        in classification class
        '''
        return node.data[1].nunique() == 1

    def calculate_value(self, node: Node):
        '''
        Calculates value of the leaf.
        The value is the size corresponds to the class that has more samples
        '''
        flase_values = node.data[1].value_counts().get(0, 0)
        true_values = node.data[1].value_counts().get(1, 0)
        node.value = true_values > flase_values

    def build_tree(self, x_train: pd.DataFrame,
                   y_train: pd.DataFrame):
        '''
        Main method to build decision tree model.
        It uses queue to split nodes until all leafs are found
        '''
        # inicialize root with starting data
        data = x_train, y_train
        root = Node(data=data, depth=0)

        # create a queue for upcomming nodes
        q = Queue()
        q.put(root)

        # generate nodes until the queue is empty
        while not q.empty():
            node = q.get()

            if self.check_if_leaf(node) or node.depth == self.max_depth:
                # if taken node is a leaf, just calculate its value
                self.calculate_value(node)
                continue

            # new x_train and y_train are from this node's data
            x_train = node.data[0]
            y_train = node.data[1]

            the_best_split = self.splitter.find_split(x_train, y_train)
            if the_best_split is None:
                # if the best split is None, no split is possible
                # (consider this a leaf)
                self.calculate_value(node)
                continue

            # create left and right children
            feature, value = the_best_split
            node.split_condition = feature
            node.split_condition_value = value
            child_left, child_right = self.splitter.split(x_train,
                                                          y_train,
                                                          feature,
                                                          value)
            child_left.depth = child_right.depth = node.depth + 1
            node.left_child = child_left
            node.right_child = child_right

            # put new nodes to the queue
            q.put(child_left)
            q.put(child_right)

        return root


class DecisionTreeClassifier(Solver):
    def __init__(self, min_samples: int = 1, max_depth: int = float('inf'),
                 numeric_values_limit=300, numeric_split_step=100,
                 evaluation_criterion: str = 'gini'):
        '''
        Class representing a DecisionTree model
        :max_depth: number of child nodes that can be generated
        :evaluator: string representing type of split evaluator (gini/entropy)
        :values_limit: limit of rows in continuous columns to make
            splitting conditions from all
        :splitting_step: number to get step for splittig long continuous data
        :samples_limit: minimum number of samples in the single node
        '''
        self.max_depth = max_depth
        self.evaluator = evaluation_criterion
        self.values_limit = numeric_values_limit
        self.splitting_step = numeric_split_step
        self.samples_limit = min_samples
        self.tree = Tree(min_samples, max_depth, numeric_values_limit,
                         numeric_split_step, evaluation_criterion)
        self.root = None

    def get_parameters(self) -> dict:
        return {
            "max depth": self.max_depth,
            "evaluation criterion": self.evaluator,
            "values limit": self.values_limit,
            "splitting_step": self.splitting_step,
            "samples_limit": self.samples_limit
        }

    def fit(self, X, y) -> None:
        '''
        Method to train model
        :X: training DataFrame
        :y: values of class column for training DataFrame
        '''
        self.root = self.tree.build_tree(X, y)

    def predict(self, x_validate: pd.DataFrame) -> pd.Series:
        '''
        Returns pd.Series od predictions for given DataFrame
        :X: DataFrame to predict class column values
        '''
        preds = []
        for index, row in x_validate.iterrows():
            value = self._single_predict(row)
            preds.append(value)
        preds = pd.Series(preds)
        return preds

    def _single_predict(self, x: pd.Series) -> bool:
        '''
        Returns prediction for single sample
        :x: sample to predict
        '''
        # inicialize node with root node
        node = self.root

        # go down the tree until reaching the first leaf
        while node.value is None:
            if node.split_condition_value is None:
                # true/false split
                if x[node.split_condition] == 1:
                    node = node.left_child
                else:
                    node = node.right_child
            else:
                # numeric data split
                if x[node.split_condition] <= node.split_condition_value:
                    node = node.left_child
                else:
                    node = node.right_child

        # reached leaf's value is value that will be predicted
        return node.value
