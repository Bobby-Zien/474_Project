# q-learning

import numpy as np
import matplotlib.pyplot as plt

# Set up wind and environment
# (Row, Column)
# Top left = (0,0)
bounds = [28, 25]
# wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0] # Blowing North

# Kings flag
KINGS = False

# Current Position
current_position = [3, 0]

# Initialize Epislon, Alpha, & Gamma
epsilon = 0.05
alpha = 0.2
gamma = 0.9


def pacmap():
    # game map
    walls_pos = [(0, 0, 232, 8), (0, 8, 8, 104), (112, 8, 8, 32), (224, 8, 8, 104), (24, 24, 24, 16), (64, 24, 32, 16), (136, 24, 32, 16), (184, 24, 24, 16), (24, 56, 24, 8), (64, 56, 8, 56), (88, 56, 56, 8), (160, 56, 8, 56), (184, 56, 24, 8), (112, 64, 8, 24), (8, 80, 40, 32), (72, 80, 24, 8), (136, 80, 24, 8), (184, 80, 40, 32), (88, 104, 4, 32), (140,104,4,32), (92,128,48,8),(92,104,16,8),(124,104,16,8),(0, 128, 48, 32), (64, 128, 8, 32), (160, 128, 8, 32), (184, 128, 48, 32), (88, 152, 56, 8), (0, 160, 8, 96), (112, 160, 8, 24), (224, 160, 8, 96), (24, 176, 24, 8), (64, 176, 32, 8), (136, 176, 32, 8), (184, 176, 24, 8), (40, 184, 8, 24), (184, 184, 8, 24), (8, 200, 16, 8), (64, 200, 8, 32), (88, 200, 56, 8), (160, 200, 8, 32), (208, 200, 16, 8), (112, 208, 8, 24), (24, 224, 40, 8), (72, 224, 24, 8), (136, 224, 24, 8), (168, 224, 40, 8), (8, 248, 216, 8)] 
    cookies_pos = [(14, 14), (22, 14), (30, 14), (38, 14), (46, 14), (54, 14), (62, 14), (70, 14), (78, 14), (86, 14), (94, 14), (102, 14), (126, 14), (134, 14), (142, 14), (150, 14), (158, 14), (166, 14), (174, 14), (182, 14), (190, 14), (198, 14), (206, 14), (214, 14), (14, 22), (54, 22), (102, 22), (126, 22), (174, 22), (214, 22), (14, 190), (214,190), (14,30), (54, 30), (102, 30), (126, 30), (174, 30), (214, 30), (14, 38), (54, 38), (102, 38), (126, 38), (174, 38), (214, 38), (14, 46), (22, 46), (30, 46), (38, 46), (46, 46), (54, 46), (62, 46), (70, 46), (78, 46), (86, 46), (94, 46), (102, 46), (110, 46), (118, 46), (126, 46), (134, 46), (142, 46), (150, 46), (158, 46), (166, 46), (174, 46), (182, 46), (190, 46), (198, 46), (206, 46), (214, 46), (14, 54), (54, 54), (78, 54), (150, 54), (174, 54), (214, 54), (14, 62), (54, 62), (78, 62), (150, 62), (174, 62), (214, 62), (14, 70), (22, 70), (30, 70), (38, 70), (46, 70), (54, 70), (78, 70), (86, 70), (94, 70), (102, 70), (126, 70), (134, 70), (142, 70), (150, 70), (174, 70), (182, 70), (190, 70), (198, 70), (206, 70), (214, 70), (54, 78), (174, 78), (54, 86), (174, 86), (54, 94), (174, 94), (54, 102), (174, 102), (54, 110), (174, 110), (54, 118), (174, 118), (54, 126), (174, 126), (54, 134), (174, 134), (54, 142), (174, 142), (54, 150), (174, 150), (54, 158), (174, 158), (14, 166), (22, 166), (30, 166), (38, 166), (46, 166), (54, 166), (62, 166), (70, 166), (78, 166), (86, 166), (94, 166), (102, 166), (126, 166), (134, 166), (142, 166), (150, 166), (158, 166), (166, 166), (174, 166), (182, 166), (190, 166), (198, 166), (206, 166), (214, 166), (14, 174), (54, 174), (102, 174), (126, 174), (174, 174), (214, 174), (14, 182), (54, 182), (102, 182), (126, 182), (174, 182), (214, 182), (22, 190), (30, 190), (54, 190), (62, 190), (70, 190), (78, 190), (86, 190), (94, 190), (102, 190), (126, 190), (134, 190), (142, 190), (150, 190), (158, 190), (166, 190), (174, 190), (198, 190), (206, 190), (30, 198), (54, 198), (78, 198), (150, 198), (174, 198), (198, 198), (30, 206), (54, 206), (78, 206), (150, 206), (174, 206), (198, 206), (14, 214), (22, 214), (30, 214), (38, 214), (46, 214), (54, 214), (78, 214), (86, 214), (94, 214), (102, 214), (126, 214), (134, 214), (142, 214), (150, 214), (174, 214), (182, 214), (190, 214), (198, 214), (206, 214), (214, 214), (14, 222), (102, 222), (126, 222), (214, 222), (14, 230), (102, 230), (126, 230), (214, 230), (14, 238), (22, 238), (30, 238), (38, 238), (46, 238), (54, 238), (62, 238), (70, 238), (78, 238), (86, 238), (94, 238), (102, 238), (110, 238), (118, 238), (126, 238), (134, 238), (142, 238), (150, 238), (158, 238), (166, 238), (174, 238), (182, 238), (190, 238), (198, 238), (206, 238), (214, 238)] 
    walls_pos = np.array(walls_pos)
    cookies_pos = np.array(cookies_pos)
    value_table = np.zeros((29,26))

    for cookie in cookies_pos:
        if cookie[0] == 14: cookie[0] = 8
        if cookie[1] == 14: cookie[1] = 8
        
    for i in range(len(cookies_pos)):
        cookies_pos[i][0] = (cookies_pos[i][0] - 8)/8
        cookies_pos[i][1] = (cookies_pos[i][1] - 8)/8
        value_table[cookies_pos[i][1],cookies_pos[i][0]] = 1

    return value_table
    # print map
    ##for i in range(29):
    ##    for j in range(26):
    ##        if int(value_table[i][j]) == -1:
    ##            print('-',end = "")
    ##        else:
    ##            print(int(value_table[i][j]),end = "")
    ##    print()
    

"""
Resets policies and current position so SARSA and Q-learning can be run repeatedly
"""
def reset_world():
    global policy, q_table, sarsa_table, current_position, actions
    current_position = [3,0]
    if KINGS: # Stochastic and Kings move
        # policy = np.ones((7, 10, 8)) * 0.125
        q_table = np.zeros((29, 26, 8))
        sarsa_table = np.zeros((7, 10, 8))
        sarsa_lambda_table = np.zeros((7, 10, 8))
        q_lambda_table = np.zeros((7, 10, 8))
        actions = [[0, 1], [0, -1], [1, 0], [-1, 0], [1,1],[-1,-1],[1,-1],[-1,1]]
    else: # Regular wind and regular steps
    # policy = np.ones((7, 10, 4)) * 0.25
        q_table = np.zeros((29, 26, 2, 4))
        sarsa_table = np.zeros((7, 10, 4))
        sarsa_lambda_table = np.zeros((7, 10, 4))
        q_lambda_table = np.zeros((7, 10, 4))
        actions = [[0, 1], [0, -1], [1, 0], [-1, 0]]

"""
Finds the max value in an array and returns it's index (breaking ties randomly)
Input: An array of values
Output: Index of max value in array
"""
def max_with_broken_ties(array):
    maxes = []
    best = -np.inf
    for idx, val in enumerate(array):
        if val > best:
            best = val
            maxes = [idx]
        elif val == best:
            maxes.append(idx)
    return np.random.choice(maxes)



"""
Changes current position based on the action and wind values
Input: An int 0, 1, 2, or 3 corresponding to right, left, down, and up respectively
Output: reward of moving
"""
def act(action_num, value_table):
    global current_position

    # Get action values
    action = actions[action_num]

    # Calculate new position including wind
    new_position = [ # - because 0,0 is top left
    current_position[0] + action[0],
    current_position[1] + action[1]
    ]

    # Verify validity of value, and if valid set as new value
    if new_position[0] < 0:
        new_position[0] = 0
    elif new_position[0] > bounds[0]:
        new_position[0] = bounds[0]

    if new_position[1] < 0:
        new_position[1] = 0
    elif new_position[1] > bounds[1]:
        new_position[1] = bounds[1]
    
    # collide wall
    if value_table[new_position[0], new_position[1]] == 0:
        new_position = [current_position[0], current_position[1]]

    current_position = new_position

    # Return reward for action
    return -1


"""
Update the Q(s,a) table using Q-learning
Input: current state: S, action: A (never used), reward from taking A at state S: R, the next state (reached from S using A): S'
"""
def q_update(sars):
    state, action, reward, next_state, cookie, next_cookie = sars
    state_0, state_1 = state
    next_state_0, next_state_1 = next_state
    q_table[state_0, state_1, cookie, action] = q_table[state_0, state_1, cookie, action] + alpha*(reward + gamma*max(q_table[next_state_0, next_state_1, next_cookie]) - q_table[state_0, state_1, cookie, action])


def eps_greedy(table, cookie, current_position):
    if KINGS:
        moves = list(range(8))
    else:
        moves = list(range(4))
    if np.random.random() < epsilon:
        return np.random.choice(moves)
    return max_with_broken_ties(table[current_position[0], current_position[1], cookie])

"""
Display a graph of the episodic reward over time
"""
def graph_reward_over_time(rewards):
    plt.plot(rewards)
    plt.ylabel("Total Discounted Reward")
    plt.xlabel("Episode")
    plt.show()

"""
Display a graph containing both the q and sarsa episodic rewards over time
"""
def graph_two_rewards_over_time(q_rewards, sarsa_rewards):
    plt.plot(q_rewards, label='Q Learning')
    plt.plot(sarsa_rewards, label = 'SARSA')
    plt.ylabel("Total Discounted Reward")
    plt.xlabel("Episode")
    plt.show()


##def displayRecentPath(states):
##    path_taken = np.zeros((8, 10), dtype=str)
##    for ix, iy in np.ndindex(path_taken.shape):
##        path_taken[ix][iy] = ' '
##
##    for st in states:
##        path_taken[st[0]][st[1]] = "X"
##
##    goalR, goalC = goal
##    path_taken[goalR][goalC] = "G"
##
##    print("PATH\n{}".format(path_taken))


def run_q_simulation(num_episodes, display_graph=False):
    global current_position
    reset_world()
    final_rewards = np.zeros(num_episodes)
    states = [(3,0)]
    
    # Loop for each episode:
    for ep in range(num_episodes):
        # Initialize S
        value_table = pacmap()
        cookies = 244
        cookie = 0
        current_position = [3, 0]
        # reached_goal = False
        discounted_reward = 0
        state = current_position[0], current_position[1]
        # Loop for each step of episode:
        while cookies > 0:
            # Choose A from S using policy derived from Q
            action = eps_greedy(q_table, cookie, current_position)
            
            # Take action A, observe R, S'
            reward = act(action, value_table)
            discounted_reward += reward
            
            # cookies
            if (value_table[current_position[0], current_position[1]] == 1):
                cookies -= 1
                # print(cookies)
                value_table[current_position[0], current_position[1]] = 2
                next_state = current_position[0], current_position[1]
                # Q(S, A) <- Q(S, A) + α[r + γ max_a Q(S', a) - Q(S, A)]                
                q_update((state, action, reward, next_state, cookie, 0))
                cookie = 0
            else:
                next_state = current_position[0], current_position[1]                        
                # Q(S, A) <- Q(S, A) + α[r + γ max_a Q(S', a) - Q(S, A)]
                q_update((state, action, reward, next_state, cookie, 1))
                cookie = 1
                
            state = [next_state[0], next_state[1]]
            if ep == num_episodes-1:
                states.append(state)

        final_rewards[ep] = discounted_reward
        #print(discounted_reward)

        # Calculate delta
        avg = (sum(final_rewards[ep-50:ep]) / 50)
        delta = abs(avg - final_rewards[ep])
        #print(delta)
        if (delta < 0.02):
            print(ep, delta)

    if display_graph:
        graph_reward_over_time(final_rewards)

    # displayRecentPath(states)

# Run Q-Learning Simulation
run_q_simulation(1000, display_graph=True)
