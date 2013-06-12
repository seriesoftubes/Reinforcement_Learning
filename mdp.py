import random

import util


class MDP(object):    
    def __init__(self, states, gamma=0.9, epsilon=0.001):
        """A Markov Decision Process, defined by:
        -an initial state
        -transition model
        -reward function
        @param states: all possible states in the MDP
        @param gamma: discount rate for future rewards
        @param epsilon: Value function convergence threshold
        
        """
        self.states = states 
        self.gamma = gamma
        self.epsilon = epsilon
    
    def get_reward(self, state):
        raise NotImplementedError
    
    def get_possible_actions(self, state):
        """What can you do in <state>?"""
        raise NotImplementedError
    
    def get_future_probability_and_state_pairs(self, state, action):
        """What can happen if you do <action> in <state>?"""
        raise NotImplementedError
    
    def get_expected_utility(self, state, action):
        return sum(probability * self.utility_by_state[new_state] for probability, new_state in 
                    self.get_future_probability_and_state_pairs(state, action))
    
    def update_utility_by_state(self, action_by_state):
        new_utility_by_state = self.utility_by_state.copy()
        there_was_some_change_in_utility_function = True
        while there_was_some_change_in_utility_function:
            self.utility_by_state.update(new_utility_by_state)
            max_delta = 0
            for state in self.states:
                action = action_by_state[state]
                new_utility_by_state[state] = self.get_reward(state) + self.gamma * self.get_expected_utility(state, action)
                max_delta = max(max_delta, abs(new_utility_by_state[state] - self.utility_by_state[state]))
            there_was_some_change_in_utility_function = max_delta >= self.epsilon * (1 - self.gamma) / self.gamma
    
    def get_optimal_action_by_state_via_policy_iteration(self):
        self.utility_by_state = dict.fromkeys(self.states, 0)
        action_by_state = dict((state, random.choice(self.get_possible_actions(state))) for state in self.states)
        while True:
            self.update_utility_by_state(action_by_state)
            policy_is_completely_unchanged = True
            for state in self.states:
                best_action = util.argmax(self.get_possible_actions(state), 
                                      lambda action: self.get_expected_utility(state, action))
                if best_action != action_by_state[state]:
                    action_by_state[state] = best_action
                    policy_is_completely_unchanged = False
            if policy_is_completely_unchanged:
                return action_by_state
    
    def get_optimal_action_by_state_via_value_iteration(self):
        self.utility_by_state = dict.fromkeys(self.states, 0)
        new_utility_by_state = dict.fromkeys(self.states, 0)
        there_was_some_change_in_utility_function = True
        while there_was_some_change_in_utility_function:
            self.utility_by_state.update(new_utility_by_state)
            max_delta = 0
            for state in self.states:
                max_possible_utility = max(self.get_expected_utility(state, action) for action in self.get_possible_actions(state))
                new_utility_by_state[state] = self.get_reward(state) + self.gamma * max_possible_utility
                max_delta = max(max_delta, abs(new_utility_by_state[state] - self.utility_by_state[state]))
            there_was_some_change_in_utility_function = max_delta >= self.epsilon * (1 - self.gamma) / self.gamma
        action_by_state = {}
        for state in self.states:
            possible_actions = self.get_possible_actions(state)
            action_by_state[state] = util.argmax(possible_actions, lambda action: self.get_expected_utility(state, action))
        return action_by_state