from mdp import MDP


class MazeMDP(MDP):
    def get_reward(self, state):
        reward = -1
        if state == (0, 0): #Finishing line
            reward = 0
        elif state == (1, 1): #Danger zone
            reward = -99
        return reward
    
    def get_possible_actions(self, state):
        return ('^^', 'vv', '>>', '<<')
    
    def get_future_probability_and_state_pairs(self, state, action):
        row_number, column_number = state
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
            return [(1, new_state)]
        else:
            return [(1, state)]


def print_report(mdp):
    optimal_action_by_state_policy_iteration = mdp.get_optimal_action_by_state_via_policy_iteration()
    utility_by_state_policy_iteration = mdp.utility_by_state
    optimal_action_by_state_value_iteration = mdp.get_optimal_action_by_state_via_value_iteration()
    utility_by_state_value_iteration = mdp.utility_by_state
    assert optimal_action_by_state_value_iteration == optimal_action_by_state_policy_iteration
    headers = ['Coordinates', 'Action (Policy Iteration)', 'Action (Value Iteration)', 
               'Utility (Policy Iteration)', 'Utility (Value Iteration)']
    print '\t'.join(headers)
    for state in sorted(optimal_action_by_state_policy_iteration):
        action_PI = optimal_action_by_state_policy_iteration[state]
        action_VI = optimal_action_by_state_value_iteration[state]
        utility_PI = utility_by_state_policy_iteration[state]
        utility_VI = utility_by_state_value_iteration[state]
        row = [state, action_PI, action_VI, utility_PI, utility_VI]
        print '\t'.join(map(str, row))

def main():
    states = [ #(row, column)
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 0), (2, 1),
    ]
    gamma = 0.5
    mdp = MazeMDP(states, gamma)
    print_report(mdp)

if __name__ == '__main__':
    main()