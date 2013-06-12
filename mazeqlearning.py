import qlearning


class Maze(qlearning.Environment):
    def get_state(self):
        return self.state
    
    def update_state(self, action):
        row_number, column_number = self.state
        if action == '^^':
            new_state = (row_number - 1, column_number)
        elif action == 'vv':
            new_state = (row_number + 1, column_number)
        elif action == '<<':
            new_state = (row_number, column_number - 1)
        elif action == '>>':
            new_state = (row_number, column_number + 1)
        new_row_number, new_column_number = new_state
        in_bounds = (0 <= new_row_number <= 2) and (0 <= new_column_number <= 1)
        if in_bounds:
            self.state = new_state
    
    def reset(self):
        self.state = (2, 1)


class MazeRunner(qlearning.Agent):
    def get_possible_actions(self, state):
        return ('^^', 'vv', '>>', '<<')
    
    def get_reward(self, state):
        reward = -1
        if state == (0, 0): #Finishing line
            reward = 0
        elif state == (1, 1): #Danger zone
            reward = -99
        return reward
    
    def reached_goal(self, state):
        return state == (0, 0)


def main():
    maze = Maze()
    alpha = 0.2
    epsilon = 0.5
    agent = MazeRunner(maze, alpha, epsilon)
    final_action_by_state = agent.get_optimal_action_by_state_via_sarsa_q_learning(18908)
    final_q_by_state_action = agent.q_by_state_action
    cumulative_iterations = 0
    maze = Maze()
    agent = MazeRunner(maze, alpha, epsilon)
    headers = ['Cumulative Iterations', 'Policy Difference', 'Q Value Difference']
    print '\t'.join(headers)
    while cumulative_iterations <= 18908:
        action_by_state = agent.get_optimal_action_by_state_via_sarsa_q_learning(100)
        policy_difference = sum(1 for state, action in action_by_state.iteritems() if action != final_action_by_state[state])
        q_value_by_state_action = agent.q_by_state_action
        q_value_difference = sum(abs(q_value - final_q_by_state_action[state, action]) for (state, action), q_value 
                                 in q_value_by_state_action.iteritems())
        row = [cumulative_iterations, policy_difference, q_value_difference]
        print '\t'.join(map(str, row))
        cumulative_iterations += 100

if __name__ == '__main__':
    main()