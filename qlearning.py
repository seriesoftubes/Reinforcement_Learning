import collections
import random

import util


class Environment(object):
    """Represents the world surrounding the agent(s)"""
    def __init__(self):
        self.state = None
    
    def reset(self):
        """Reset the simulation"""
        raise NotImplementedError
    
    def get_state(self):
        """From the Environment's sensors"""
        raise NotImplementedError
    
    def update_state(self, action):
        """This is used for simulation purposes only.
        Although an Action may change a real environment, the real world updates itself
        and is only reflected through sensor data.
        So in real life, the Action parameter would probably be ignored 
        and only sensor data would be used to determine the state.
        
        """
        raise NotImplementedError


class Agent(object):
    def __init__(self, environment, alpha=0.2, gamma=0.9, epsilon=0.1):
        """An autonomous agent without any predetermined knowledge of:
        -states
        -transition probabilities
        @param environment: the environment with which the Agent interacts
        @param alpha: learning rate
        @param gamma: discount rate for future rewards
        @param epsilon: Value function convergence threshold
        
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.environment = environment
        self.q_by_state_action = collections.defaultdict(int)
    
    def detect_state(self):
        return self.environment.get_state()
    
    def get_possible_actions(self, state):
        """What can you do in <state>?"""
        raise NotImplementedError
    
    def get_reward(self, state):
        raise NotImplementedError
    
    def reached_goal(self, state):
        raise NotImplementedError
    
    def get_best_action(self, state):
        possible_actions = self.get_possible_actions(state)
        get_q_value = lambda action: self.q_by_state_action.get((state, action), 0)
        return util.argmax(possible_actions, get_q_value)
    
    def update_q_value(self, state1, action1, reward, state2):
        """Q(s,a) <- Q(s,a) + a*(R(s) + y*max(Q(s',a')) - Q(s,a))"""
        old_q_value = self.q_by_state_action[state1, action1]
        best_future_q = max(self.q_by_state_action.get((state2, action), 0) for action in self.get_possible_actions(state2))
        self.q_by_state_action[state1, action1] = old_q_value + self.alpha * (reward + (self.gamma * best_future_q) - old_q_value)
    
    def get_epsilon_greedy_action(self, state):
        """Get a random action with probability: Epsilon. 
        With probability 1-Epsilon, go with the action that maximizes Q.
        If there are multiple actions with q == max(Q), choose one randomly.
        
        """
        possible_actions = self.get_possible_actions(state)
        if random.random() < self.epsilon:
            action = random.choice(possible_actions)
        else:
            q_values = [self.q_by_state_action.get((state, action), 0) for action in possible_actions]
            maxq = max(q_values)
            count = q_values.count(maxq)
            if count > 1:
                best = [i for i in xrange(len(possible_actions)) if q_values[i] == maxq]
                i = random.choice(best)
            else:
                i = q_values.index(maxq)
            action = possible_actions[i]
        return action
    
    def get_optimal_action_by_state_via_sarsa_q_learning(self, episodes=100):
        count = 0
        self.environment.reset()
        while count < episodes:
            state1 = self.detect_state()
            reward = self.get_reward(state1)
            action1 = self.get_epsilon_greedy_action(state1)
            self.environment.update_state(action1)
            state2 = self.detect_state()
            self.update_q_value(state1, action1, reward, state2)
            if self.reached_goal(state1):
                self.environment.reset()
                count += 1
        states = set(state for state, action in self.q_by_state_action.iterkeys())
        return dict((state, self.get_best_action(state)) for state in states)