from __future__ import annotations
from time import sleep
from copy import deepcopy
from random import choice
import numpy as np

from State import DostsAndBoxesState
from MinMax import MinMax


class DotsAndBoxes:
    def __init__(self, dimentions=3, depth_bot1=4, depth_bot2=None, player_mode=1, first_player=1) -> None:
        '''
        class representing single game in dots and boxes
        :dimentions: how big will be the board
        :depth_bot1: the depth of searching for the best move for bot1
        :depth_bot2: similarry, it can be None if the player is paying (no need for the second bot)
        :palyer_mode: if the player is playing as max or as min
        :first_player: who starts the game
        '''
        self.dimentions = dimentions
        self.player_mode = player_mode
        self.first_state = DostsAndBoxesState(first_player, self.dimentions)
        if depth_bot2 is None:
            self.minmax_evaluator = MinMax(depth_bot1)
            self.with_player = True
        else:
            self.minmax_evaluator_max = MinMax(depth_bot1)
            self.minmax_evaluator_min = MinMax(depth_bot2)
            self.with_player = False

    def play_two_bots(self) -> bool:
        '''
        method for simaluation of 2 bots playing with each other
        '''
        state = self.first_state
        score = [0, 0]
        while True:
            children = state.generate_children()
            if len(children) == 0:  # if the state is terminal, end the game loop
                break

            previous_state = deepcopy(state)
            message = "Max scored!" if state.player else "Min scored!"

            state = self.bot_make_move(state, children)  # state is change to the state picked by corresponding bot
            self.check_scored(state, previous_state, message, score)  # looks if a box was compleated by the player
            print(state)
            print('\n')
            sleep(1)

        # Who won?
        print(state)
        if score[0] == score[1]:
            print("Tie!")
        else:
            winner = "Max" if score.index(max(score)) == 1 else "Min"
            print(f"The winner is: {winner}. Max has {score[1]} pointed, Min has {score[0]} points.")

    def bot_make_move(self, state, children):
        '''
        makes a move based on alpha-beta pruning from MinMax class
        returns the best next state for min or max (depending whos turn is it)
        '''
        if state.player:
            print("Max move:")
            the_best_move = -float('inf')
            same_evaluation_moves = []
            for new_state in children:  # check evalution of every possible move to pick the best one
                # if two bots are playing with each, other, there is seperate solver for max and min, if with player, there's just one solver
                if not self.with_player:
                    evaluation = self.minmax_evaluator_max.solve(new_state)
                else:
                    evaluation = self.minmax_evaluator.solve(new_state)

                if evaluation == the_best_move:  # if this state has the same evaluation as the previous one, pick random of them
                    same_evaluation_moves.append(new_state)
                    state = choice(same_evaluation_moves)
                elif evaluation > the_best_move:  # if it found better move, clear same evalution list and change best move to founded one
                    same_evaluation_moves.clear()
                    the_best_move = evaluation
                    state = new_state
        else:
            # similary for min
            print("Min move:")
            the_best_move = float('inf')
            same_evaluation_moves = []
            for new_state in children:
                if not self.with_player:
                    evaluation = self.minmax_evaluator_min.solve(new_state)
                else:
                    evaluation = self.minmax_evaluator.solve(new_state)

                if evaluation == the_best_move:
                    same_evaluation_moves.append(new_state)
                    state = choice(same_evaluation_moves)
                elif evaluation < the_best_move:
                    same_evaluation_moves.clear()
                    the_best_move = evaluation
                    state = new_state
        return state

    def check_scored(self, state, previous_state, message, score):
        '''
        it look for the differences in previous state and current state and chekcs if the difference
        generated a box - if yes, it adds points for the current state player
        '''
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
                player_move = [float('inf'), float('inf')]  # reset if player input is invalid
                continue
            try:
                if state.state[int(player_move[0])][int(player_move[1])] != -1:  # if this place on  the board is already taken
                    player_move = [float('inf'), float('inf')]
                    continue
            except IndexError:
                player_move = [float('inf'), float('inf')]  # if player inut is invalid
                continue
            break
        state.state[int(player_move[0])][int(player_move[1])] = self.player_mode  # change the state of the place on the board
        if not state.find_box(int(player_move[0]), int(player_move[1])):  # if there was no score, changed state is now other player's turn
            state.player = not state.player
        return state

    def play(self):
        '''
        the game itself
        '''
        # simillar to play_two_bots
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
                if self.player_mode:  # if  player is max
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
                if not self.player_mode:  # if player is min
                    state = self.player_make_move(state)
                else:
                    state = self.bot_make_move(state, children)
                self.check_scored(state, previous_state, message, score)

        # Who won?
        print(state)
        if score[0] == score[1]:
            print(f"Tie! {score[0]} to {score[1]}!")
        else:
            winner = "Max" if score.index(max(score)) == 1 else "Min"
            print(f"The winner is: {winner}. Max has {score[1]} pointed, Min has {score[0]} points.")
