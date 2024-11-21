

class MinMax:
    '''a class containing Min-Max algorithm
    it uses recursion to evalueate given state (node)
    :depth: how many moves ahead will algorithm consider for evaluation
    '''
    def __init__(self, depth) -> None:
        '''define hiperparameters of the algorithm
        :depth: how many moves ahead will algorithm consider for evalution
        '''
        self._depth = depth

    @property
    def depth(self):
        return self._depth

    def solve(self, state, alfa=-float('inf'), beta=float('inf'), depth=None):
        '''
        solver using MinMax algorithm with alpha-beta pruning
        :state: state of the game that is being evaluated
        '''
        if depth is None:  # for stating states, depth is None at it sets it on the given in parameters value
            depth = self.depth
        children = state.generate_children()  # generates all of the possible next moves

        if len(children) == 0 or depth == 0:  # if the state is terminal or reached max depth, return state evaluation (recursion is going back)
            return state.evaluation

        if state.player == 1:  # if max is moving, look for the maximum evaluation
            for child in children:  # for every possible move get the evaluation and choose max of them
                alfa = max(alfa, self.solve(child, alfa, beta, depth-1))
                if alfa >= beta:  # pruning - if alfa > beta there is no point in examining this branch of moves (min won't go this way)
                    return beta
            return alfa
        else:  # simillary with min as with max
            for child in children:
                beta = min(beta, self.solve(child, alfa, beta, depth-1))
                if alfa >= beta:
                    return alfa
            return beta
