from abc import ABC, abstractmethod


class Evaluator(ABC):
    '''
    Abstract class as a pattern for all of the possible split evaluators.
    '''
    @abstractmethod
    def evaluate_split(self, left_node_counts: tuple[int],
                       right_node_counts: tuple[int]) -> float:
        '''
        Evaluate whole split
        :left_node_counts: tuple of values in left node:
        - first values being number of samples with classification class 1
        - the second with classification class 0
        :right_node_counts: like in _left_node_counts, but for right node
        '''
        pass

    @abstractmethod
    def check_upgrade(self, the_best_split_value: float,
                      split_values: float) -> bool:
        '''
        Return true if split_value (new one) if better than
        the old one (the_best_split_value)
        '''
        pass

    @abstractmethod
    def _evaluate_single_node(self, data_couts: tuple[int]) -> float:
        '''
        Evaluates single node, used in evaluate_split method
        :data_count: tuple of first values being number of samples with
        classification class 1 and the second with classification class 0
        node that is being evaluated
        '''
        pass


class GiniImpurityCalculator(Evaluator):
    def evaluate_split(self, left_node_counts: dict[int],
                       right_node_counts: dict[int]) -> float:

        # impurities for both nodes
        left_node_impurity = self._evaluate_single_node(left_node_counts)
        right_node_impurity = self._evaluate_single_node(right_node_counts)

        # weights for both nodes
        left_weight = sum(tuple(left_node_counts))
        right_weight = sum(tuple(right_node_counts))
        total_weight = left_weight + right_weight
        left_node_weight = left_weight / total_weight
        right_node_weight = right_weight / total_weight

        # calculate gini impurity based on gien formula
        left_node_component = left_node_weight*left_node_impurity
        right_node_component = right_node_weight*right_node_impurity
        return left_node_component + right_node_component

    def check_upgrade(self, the_best_impurity: float, impurity: float) -> bool:
        # the new one better than the old one
        return impurity < the_best_impurity

    def _evaluate_single_node(self, data_couts: dict[int]) -> float:
        number_of_samples = sum(data_couts)
        probability_yes = data_couts.get(1, 0) / number_of_samples
        probability_no = data_couts.get(0, 0) / number_of_samples

        return 1 - probability_yes**2 - probability_no**2
