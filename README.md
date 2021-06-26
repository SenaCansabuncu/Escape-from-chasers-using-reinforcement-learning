### Escape from chasers using reinforcement learning.

##### Solving an optimization problem using a MDP and TD learning. 

The environment for this problem is a game with walls and a single exit and two chasers. An agent (the learner and decision maker) is placed somewhere in the game. The agents goal is to reach the exit as quickly as possible. To get there the agent moves through the game in a succession of steps. For every step the agent must decide which action to take (move left/right/up/down). For this purpose the agent is trained; it learns a policy (Q) which tells what is the best next move to make. With every step the agent incurs a penalty or (when finally reaching the exit) a reward. These penalties and rewards are the input when training the policy. 

The policies (or models) used here are based on Sarsa and Q-learning. During training the learning algorithm updates the action-value function Q for each state which is visited. The highest value indicates the most preferable action. Updating the values is based on the reward or penalty incurred after the action was taken. With TD-learning a model learns at every step it takes, not only when the exit is reached. However learning does speed up once the exit has been reached for the first time. 

This project demonstrates different models which learn to move through a game. Class Maze in file *maze.py* in package *environment* defines the environment including the rules of the game (rewards, penalties). In file *main.py* an example of a game is defined (but you can create your own) as an np.array. By selecting a value for *test* from Enum Test a certain model is trained and can then be used to play a number of games from different starting positions in the game. When training or playing the agents moves can be plotted by setting respectively Maze.render (Render.TRAINING) or Maze.render (Render.MOVES).

2. *QTableModel* uses a table to record the value of each (state, action) pair. For a state the highest value indicates the most desirable action. These values are constantly refined during training. This is a fast way to learn a policy.
Requires matplotlib, numpy, keras and tensorflow.
