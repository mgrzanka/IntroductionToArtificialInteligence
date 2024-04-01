class MinMax:
    '''a class containing Min-Max algorithm
    it uses recursion to evalueate given state (node)'''
    def __init__(self, depth) -> None:
        '''define hiperparameters of the algorithm'''
        self._depth = depth

    @property
    def depth(self):
        return self._depth

    def solve(self, state, alfa=-float('inf'), beta=float('inf'), depth=None):
        '''solver using MinMax algorithm with alpha-beta pruning'''
        if depth is None:
            depth = self.depth
        children = state.generate_children()

        if len(children) == 0 or depth == 0:
            return state.evaluation

        if state.player == 1:  # jeśli rusza się max
            for child in children:
                if state.player == child.player:
                    alfa = max(alfa, self.solve(child, alfa, beta, depth))
                else:
                    alfa = max(alfa, self.solve(child, alfa, beta, depth-1))
                if alfa >= beta:
                    return beta
            return alfa
        else:  # jeśli rusza się min
            for child in children:
                if state.player == child.player:
                    beta = min(beta, self.solve(child, alfa, beta, depth))
                else:
                    beta = min(beta, self.solve(child, alfa, beta, depth-1))
                if alfa >= beta:
                    return alfa
            return beta
