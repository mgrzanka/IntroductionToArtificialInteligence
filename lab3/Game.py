from __future__ import annotations
from time import sleep
from copy import deepcopy
from random import random
import numpy as np

from State import DostsAndBoxesState
from MinMax import MinMax


class DotsAndBoxes:
    def __init__(self, dimentions=3, depth_bot1=4, depth_bot2=None, player_mode=1) -> None:
        self.dimentions = dimentions
        self.player_mode = player_mode
        self.first_state = DostsAndBoxesState(1, self.dimentions)
        if depth_bot2 is None:
            self.minmax_evaluator = MinMax(depth_bot1)
            self.with_player = True
        else:
            self.minmax_evaluator_max = MinMax(depth_bot1)
            self.minmax_evaluator_min = MinMax(depth_bot2)
            self.with_player = False

    def play_two_bots(self) -> bool:
        state = self.first_state
        score = [0, 0]
        while True:
            children = state.generate_children()
            if len(children) == 0:
                break
            previous_state = deepcopy(state)
            message = "Max scored!" if state.player else "Min scored!"
            state = self.bot_make_move(state, children)
            self.check_scored(state, previous_state, message, score)
            print(state)
            print('\n')
            sleep(1)
        # Who won?
        print(state)
        if score[0] == score[1]:
            print("Tie!")
        else:
            winner = "Max" if score.index(max(score)) == 1 else "Min"
            print(f"The winner is: {winner}")

    def bot_make_move(self, state, children):
        '''
        makes a move based on alpha-beta pruning from MinMax class
        '''
        if state.player:
            print("Max move:")
            the_best_move = -float('inf')
            for new_state in children:
                if not self.with_player:
                    evaluation = self.minmax_evaluator_max.solve(new_state)
                else:
                    evaluation = self.minmax_evaluator.solve(new_state)
                if evaluation == the_best_move:
                    draw = random()
                    state = new_state if draw < 0.5 else state
                elif evaluation > the_best_move:
                    the_best_move = evaluation
                    state = new_state
        else:
            print("Min move:")
            the_best_move = float('inf')
            for new_state in children:
                if not self.with_player:
                    evaluation = self.minmax_evaluator_max.solve(new_state)
                else:
                    evaluation = self.minmax_evaluator.solve(new_state)
                if evaluation == the_best_move:
                    draw = random()
                    state = new_state if draw < 0.5 else state
                elif evaluation < the_best_move:
                    the_best_move = evaluation
                    state = new_state
        return state

    def check_scored(self, state, previous_state, message, score):
        '''if the box has been just compleated'''
        for index, row in enumerate(state.state):
            difference = np.where(row != previous_state.state[index])[0]
            if len(difference) != 0:
                box = state.find_box(index, difference[0])
                if box:
                    print(message)
                    score[state.player] += len(box)
                    break

    def player_make_move(self, state):
        '''player's move'''
        player_move = [float('inf'), float('inf')]
        while True:
            player_move = input("Make your move: ").split(",")
            try:
                player_move = [int(player_move[0]), int(player_move[1])]
            except Exception:
                player_move = [float('inf'), float('inf')]
                continue
            try:
                if state.state[int(player_move[0])][int(player_move[1])] != -1:
                    player_move = [float('inf'), float('inf')]
                    continue
            except IndexError:
                player_move = [float('inf'), float('inf')]
                continue
            break
        state.state[int(player_move[0])][int(player_move[1])] = self.player_mode
        if not state.find_box(int(player_move[0]), int(player_move[1])):
            state.player = not state.player
        return state

    def play(self):
        '''
        the game itself
        '''
        state = self.first_state
        score = [0, 0]
        player_name = "Max" if self.player_mode else "Min"
        print(f"Welcome to the game, you are playing as {player_name}")
        while True:
            if state.player:
                # Max turn
                message = "Max scored!"
                children = state.generate_children()
                if len(children) == 0:
                    break
                print(state)
                print('\n')
                previous_state = deepcopy(state)
                if self.player_mode:
                    state = self.player_make_move(state)
                else:
                    state = self.bot_make_move(state, children)
                self.check_scored(state, previous_state, message, score)
            if not state.player:
                # Min turn
                message = "Min scored!"
                children = state.generate_children()
                if len(children) == 0:
                    break
                print(state)
                print('\n')
                previous_state = deepcopy(state)
                if not self.player_mode:
                    state = self.player_make_move(state)
                else:
                    state = self.bot_make_move(state, children)
                self.check_scored(state, previous_state, message, score)

        # Who won?
        print(state)
        if score[0] == score[1]:
            print("Tie!")
        else:
            winner = "Max" if score.index(max(score)) == 1 else "Min"
            print(f"The winner is: {winner}")
