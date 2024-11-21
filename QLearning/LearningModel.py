import numpy as np
import gymnasium as gym
import random


class LearningModel:
    def __init__(self, discount_factor: float, T: float, episodes: int,
                 states_number: int, actions_number: int, learning_rate: float) -> None:
        """
        Class for training and executing Q-learning model on the Taxi problem (gymnasium library)
        :param discount_factor: discount for calculating q-value
        :param T: temperature for getting probabilities fo choosing each action
        :param episodes: how many training games should be played to train the model
        :param states_number: how many possible states the game has
        :param actions_number: how many possible actions the game has
        :param learning rate: step for gradient descend algorithm

        Attributes
        Q-table : table state x action to store q-values

        Run fit to train game bot and play to let him execute the game
        """
        self.discount_factor = discount_factor
        self.T = T
        self.learning_rate = learning_rate
        self.episodes = episodes
        self.actions_number = actions_number
        self.Q_table = np.random.uniform(low=0, high=0.1, size=(states_number, actions_number))

    def _get_probabilities_of_actions(self, observation: int):
        """
        get possibilities of each possible action with Boltzmann Strategy
        """
        Q_values = self.Q_table[observation]  # numpy array of q-values for every action in this state
        exp_Q_values = np.exp(Q_values / self.T)

        probabilities = exp_Q_values / np.sum(exp_Q_values)  # given formula

        return probabilities

    def _get_action(self, observation: int):
        """
        choose action with calculated by method _get_probabilities_of_actions
        probabilities
        """
        probabilities = self._get_probabilities_of_actions(observation)

        possible_actions = np.arange(self.actions_number)  # actions 0-5 for taxi problem
        chosen_action = random.choices(possible_actions, weights=probabilities, k=1)[0]

        return chosen_action

    def _get_q_value(self, next_state: int, reward: int):
        """
        Q(s, a) = r(s, a) + discount * max(Q(next_s, next_a))
        """
        return reward + self.discount_factor * np.max(self.Q_table[next_state])

    def _update_Q_table(self, state: int, new_state: int, action: int, reward: int):
        """
        update Q-table by gradient descend step
        Q(s,a) = Q(s,a) + alpha*(q_value - desired_q_value)
        """
        calculated_q_value = self._get_q_value(new_state, reward)
        desired_q_value = self.Q_table[state][action]

        # calculate error
        error = calculated_q_value - desired_q_value

        # update Q-table
        self.Q_table[state][action] = self.Q_table[state][action] + self.learning_rate * error

    def fit(self):
        # initialize environment without drawing output (training)
        env = gym.make("Taxi-v3")

        for episode in range(self.episodes):
            observation, info = env.reset()
            terminated = truncated = False  # reset termination

            while not terminated and not truncated:
                # make move based on Q-function
                action = self._get_action(observation)
                new_observation, reward, terminated, truncated, info = env.step(action)

                # update Q-function based on error in predicted reward that was made
                self._update_Q_table(observation, new_observation, action, reward)
                observation = new_observation

            print(f"Episode {episode} completed")

    def play(self):
        iterations = 0
        # initialize environment with output drawing
        env = gym.make("Taxi-v3")
        observation, info = env.reset()

        terminated = truncated = False
        print("Game playing...")
        while not terminated and not truncated:
            # env.render()
            # make move based on Q-function
            action = self._get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            # time.sleep(1)  # sleep to slow down the game for better visualization
            iterations += 1
        print(reward)
        print(iterations)
        return reward, iterations
