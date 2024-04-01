from __future__ import annotations
import numpy as np
import copy


class DostsAndBoxesState:
    def __init__(self, player: bool, dimentions: int, state=None, evaluation=0) -> None:
        '''class representing a single Node
        ::player: 0 for Min, 1 for Max
        ::state: state of the game in the node
        ::dimentions: dims of the game board
        '''
        self.player = player
        self.dimentions = dimentions
        self.evaluation = evaluation
        if not state:
            self.state = self.generate_first_state()
        else:
            self.state = state

    def generate_first_state(self):
        state = []
        for i in range(2*self.dimentions - 1):
            if i % 2 == 1:
                state.append(np.full(self.dimentions, -1))
            else:
                state.append(np.full(self.dimentions - 1, -1))
        return state

    def generate_children(self) -> None:
        '''
        children: states one state under current state
        while generating, it makes the evaluation based on evaluation of the parent and potentialy found box
        '''
        childen = []
        for row_index, row in enumerate(self.state):
            for connection_index, connection in enumerate(row):
                if connection == -1:
                    child = copy.deepcopy(self.state)
                    new_row = copy.deepcopy(row)
                    new_row[connection_index] = self.player  # zamieniam -1 na gracza
                    child[row_index] = new_row
                    boxes = self.find_box(row_index, connection_index)
                    if boxes:
                        component = len(boxes) if self.player else -len(boxes)
                        evaluation = self.evaluation + component
                        childen.append(DostsAndBoxesState(self.player, self.dimentions, child, evaluation))
                    else:
                        childen.append(DostsAndBoxesState(not self.player, self.dimentions, child, self.evaluation))
        return childen

    def find_box(self, row_index, column_index) -> np.array[list[np.array]]:
        '''
        looks for the boxes from up, down, left and right
        return array of booleans up_down / left_right
        '''
        boxes = []
        if row_index % 2 == 0:
            up_down = [0 if row_index == 0 else 1,
                       0 if row_index == (2*self.dimentions - 1) - 1 else 1]
            for indx, direction in enumerate(up_down):
                if direction == 1:  # if it's possible to go either up or down
                    row1 = row_index+2*((-1)**(indx+1))
                    row2 = row_index+1*((-1)**(indx+1))
                    if self.state[row1][column_index] != -1 and \
                        self.state[row2][column_index] != -1 and \
                            self.state[row2][column_index+1] != -1:
                        boxes.append(True)
        else:
            left_right = [0 if column_index == 0 else 1,
                          0 if column_index == self.dimentions - 1 else 1]
            for indx, direction in enumerate(left_right):
                if direction == 1:  # if it's possible to go either left or right
                    column1 = column_index-1 if indx == 0 else column_index+1
                    column2 = column_index-1 if indx == 0 else column_index
                    if self.state[row_index][column1] != -1 and \
                        self.state[row_index-1][column2] != -1 and \
                            self.state[row_index+1][column2] != -1:
                        boxes.append(True)
        return boxes

    def __str__(self) -> str:
        '''
        prints out the state
        '''
        result = ""
        for index, row in enumerate(self.state):
            if index % 2 == 1:
                symbols = [' ' if connection == -1 else '|' for connection in row]
                line = [symbol+" " for symbol in symbols]
                str_line = "".join(line)
                result = result + str_line + "\n"
            else:
                symbols = [' ' if connection == -1 else '-' for connection in row]
                line = ["*"+symbol for symbol in symbols]
                str_line = "".join(line) + "*"
                result = result + str_line + "\n"
        return result
