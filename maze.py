import logging
from enum import Enum, IntEnum
import random
import matplotlib.pyplot as plt
import numpy as np
from environment.utils import Utils
from environment.runner import Runner

class Cell(IntEnum):
    EMPTY = 0  # indicates empty cell where the agent can move to
    OCCUPIED = 99  # indicates cell which contains a wall and cannot be entered
    CURRENT = 1  # indicates current cell of the agent


class Action(IntEnum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3


class Render(Enum):
    NOTHING = 0
    TRAINING = 1
    MOVES = 2


class Status(Enum):
    WIN = 0
    LOSE = 1
    PLAYING = 2


class Maze:
    """ A maze with walls. An agent is placed at the start cell and must find the exit cell by moving through the maze.

        The layout of the maze and the rules how to move through it are called the environment. An agent is placed
        at start_cell. The agent chooses actions (move left/right/up/down) in order to reach the exit_cell. Every
        action results in a reward or penalty which are accumulated during the game. Every move gives a small
        penalty (-0.05), returning to a cell the agent visited earlier a bigger penalty (-0.25) and running into
        a wall a large penalty (-0.75). The reward (+10.0) is collected when the agent reaches the exit. The
        game always reaches a terminal state; the agent either wins or looses. Obviously reaching the exit means
        winning, but if the penalties the agent is collecting during play exceed a certain threshold the agent is
        assumed to wander around clueless and looses.

        A note on cell coordinates:
        The cells in the maze are stored as (col, row) or (x, y) tuples. (0, 0) is the upper left corner of the maze.
        This way of storing coordinates is in line with what matplotlib's plot() function expects as inputs. The maze
        itself is stored as a 2D numpy array so cells are accessed via [row, col]. To convert a (col, row) tuple
        to (row, col) use: (col, row)[::-1]
    """
    actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN]  # all possible actions

    reward_exit = 10.0  # reward for reaching the exit cell
    penalty_move = -0.05  # penalty for a move which did not result in finding the exit cell
    penalty_visited = -0.25  # penalty for returning to a cell which was visited earlier
    penalty_impossible_move = -0.75  # penalty for trying to enter an occupied cell or moving out of the maze

    def __init__(self, maze,runner, start_cell=(12, 7),start_cell2=(11, 7), exit_cell=None):
        """ Create a new maze game.

            :param numpy.array maze: 2D array containing empty cells (= 0) and cells occupied with walls (= 1)
            :param tuple start_cell: starting cell for the agent in the maze (optional, else upper left)
            :param tuple exit_cell: exit cell which the agent has to reach (optional, else lower right)
        """
        self.maze = maze
        self.runner=runner
        ddx=self.runner_pos().action[0]
        self.__minimum_reward =-0.5 * self.maze.board.size  # stop game if accumulated reward is below this threshold
        self.__minimum_reward2 =-0.5 * self.maze.board.size 
        nrows, ncols = self.maze.board.shape 
        self.cells = [(col, row) for col in range(ncols) for row in range(nrows)]
        self.empty = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze.board[row, col] == Cell.EMPTY]
        self.turn = 1
        self.__exit_cell =(ddx[1],ddx[0])
        self.__exit_previous_cell=self.__current_exit_cell=self.__exit_cell

        # Check for impossible maze layout
        if self.__exit_cell not in self.cells:
            raise Exception("Error: exit cell at {} is not inside maze".format(self.__exit_cell))
        if self.maze.board[self.__exit_cell[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: exit cell at {} is not free".format(self.__exit_cell))

        # Variables for rendering
        self.__render = Render.NOTHING  # what to render
        self.__ax1 = None  # axes for rendering the moves
        self.__ax2 = None  # axes for rendering the best action per cell
        self.reset2(start_cell2)
        self.reset(start_cell)
        
    def update_board( self ):
        ddd=self.runner_pos()
        prev=ddd.action[0]
        current=ddd.action[1]
        self.maze.board[prev[1],prev[0]] = 0
        self.maze.board[current[1],current[0]] = -1
        return self.runner_pos()

    def reset(self, start_cell=(12, 7)):
        """ Reset the maze to its initial state and place the agent at start_cell.

            :param tuple start_cell: here the agent starts its journey through the maze (optional, else upper left)
            :return: new state after reset
        """
        if start_cell not in self.cells:
            raise Exception("Error: start cell at {} is not inside maze".format(start_cell))
        if self.maze.board[start_cell[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: start cell at {} is not free".format(start_cell))
        if start_cell == self.__exit_cell:
            raise Exception("Error: start- and exit cell cannot be the same {}".format(start_cell))

        self.__previous_cell = self.__current_cell = start_cell

        self.__total_reward = 0.0  # accumulated reward
        self.__visited = set()  # a set() only stores unique values
        self.cizim()
        return self.__observe()

    def reset2(self, start_cell2):
            """ Reset the maze to its initial state and place the agent at start_cell.

                :param tuple start_cell: here the agent starts its journey through the maze (optional, else upper left)
                :return: new state after reset
            """
            if start_cell2 not in self.cells:
                raise Exception("Error: start cell at {} is not inside maze".format(start_cell2))
            if self.maze.board[start_cell2[::-1]] == Cell.OCCUPIED:
                raise Exception("Error: start cell at {} is not free".format(start_cell2))
            if start_cell2 == self.__exit_cell:
                raise Exception("Error: start- and exit cell cannot be the same {}".format(start_cell2))

            self.__previous_cell2 = self.__current_cell2 = start_cell2
            self.__total_reward2 = 0.0  # accumulated reward
            self.__visited2 = set()  # a set() only stores unique values      
            return self.__observe2()

    def cizim(self):
        if self.__render in (Render.TRAINING, Render.MOVES):
            # render the maze
            nrows, ncols = self.maze.board.shape
            self.__ax1.clear()
            self.__ax1.set_xticks(np.arange(0.5, ncols, step=1))#burayı degistirdim
            self.__ax1.set_xticklabels([])
            self.__ax1.set_yticks(np.arange(0.5, ncols, step=1))
            self.__ax1.set_yticklabels([])
            self.__ax1.grid(True)
            self.__ax1.plot(*self.__current_cell, "rs", markersize=30)  # start is a big red square
            self.__ax1.text(*self.__current_cell, "C1", ha="center", va="center", color="white")
            self.__ax1.plot(*self.__current_cell2, "rs", markersize=30)  # start is a big red square
            self.__ax1.text(*self.__current_cell2, "C2", ha="center", va="center", color="white")
            self.__ax1.plot(*self.__current_exit_cell, "gs", markersize=30)  # exit is a big green square
            self.__ax1.text(*self.__current_exit_cell, "Runner", ha="center", va="center", color="white")
            self.__ax1.imshow(self.maze.board, cmap="binary")
            self.__ax1.get_figure().canvas.draw()
            self.__ax1.get_figure().canvas.flush_events()
                

    def __draw(self):
        """ Draw a line from the agents previous cell to its current cell. """
        self.__ax1.plot(*zip(*[self.__previous_cell, self.__current_cell]), "bo-")  # previous cells are blue dots
        self.__ax1.plot(*self.__current_cell, "ro")  # current cell is a red dot
        self.__ax1.plot(*zip(*[self.__previous_cell2, self.__current_cell2]), "bo-")  # previous cells are blue dots
        self.__ax1.plot(*self.__current_cell2, "ro")  # current cell is a red dot
        self.__ax1.plot(*zip(*[self.__exit_previous_cell, self.__current_exit_cell]), "bo-")  # previous cells are blue dots
        self.__ax1.plot(*self.__current_exit_cell, "ro")  # current cell is a red dot
        self.__ax1.get_figure().canvas.draw()
        self.__ax1.get_figure().canvas.flush_events()

    def render(self, content=Render.NOTHING):
        """ Record what will be rendered during play and/or training.

            :param Render content: NOTHING, TRAINING, MOVES
        """
        self.__render = content

        if self.__render == Render.NOTHING:
            if self.__ax1:
                self.__ax1.get_figure().close()
                self.__ax1 = None
            if self.__ax2:
                self.__ax2.get_figure().close()
                self.__ax2 = None
        if self.__render == Render.TRAINING:
            if self.__ax2 is None:
                fig, self.__ax2 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Best move")
                self.__ax2.set_axis_off()
                self.render_q(None)
        if self.__render in (Render.MOVES, Render.TRAINING):
            if self.__ax1 is None:
                fig, self.__ax1 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Runner-Chaser Game")

        plt.show(block=False)

    def step(self, action):
        """ Move the agent according to 'action' and return the new state, reward and game status.

            :param Action action: the agent will move in this direction
            :return: state, reward, status
        """
        reward = self.__execute(action)
        #self.update_board()
        self.__total_reward += reward
        status = self.__status()
        state = self.__observe()
        logging.debug("action: {:10s} | reward: {: .2f} | status: {}".format(Action(action).name, reward, status))
        return state, reward, status
    def step2(self, action):
        """ Move the agent according to 'action' and return the new state, reward and game status.

            :param Action action: the agent will move in this direction
            :return: state, reward, status
        """
        reward2 = self.__execute2(action)
        
        #self.update_board()
        self.__total_reward2 += reward2
        status2 = self.__status2()
        state2 = self.__observe2()
        logging.debug("action: {:10s} | reward: {: .2f} | status: {}".format(Action(action).name, reward2, status2))
        return state2, reward2, status2    
    def __execute(self, action):
        """ Execute action and collect the reward or penalty.

            :param Action action: direction in which the agent will move
            :return float: reward or penalty which results from the action
        """
        possible_actions = self.__possible_actions(self.__current_cell)

        if not possible_actions:
            reward = self.__minimum_reward - 1  # cannot move anywhere, force end of game
        elif action in possible_actions:
            col, row = self.__current_cell
            if action == Action.MOVE_LEFT:
                col -= 1
            elif action == Action.MOVE_UP:
                row -= 1
            if action == Action.MOVE_RIGHT:
                col += 1
            elif action == Action.MOVE_DOWN:
                row += 1

            self.__previous_cell = self.__current_cell
            self.__current_cell = (col, row)
            if self.maze.board[self.__previous_cell[1],self.__previous_cell[0]]==2:
                self.maze.board[self.__previous_cell[1],self.__previous_cell[0]] = 0
            self.maze.board[self.__current_cell[1],self.__current_cell[0]] =2

            prev_exit=self.update_board().action[0]

            self.__exit_previous_cell =self.__current_exit_cell
            
            self.__current_exit_cell =(prev_exit[0],prev_exit[1])
            #print(" execute__exit_previous_cell" , self.__exit_previous_cell)
            #print(" execute__current_exit_cell" , self.__current_exit_cell)
            if self.__render != Render.NOTHING:
                self.__draw()

            if self.__current_cell == self.__current_exit_cell: 
                #reward = Maze.reward_exit  # maximum reward when reaching the exit cell
                reward=self.maze.utils.calc_points(self.__current_exit_cell,self.__current_cell)
                                                                       
            elif self.__current_cell in self.__visited:
                #reward = Maze.penalty_visited  # penalty when returning to a cell which was visited earlier
                reward=self.maze.utils.calc_points(self.__current_exit_cell,self.__current_cell)
            else:
                #reward = Maze.penalty_move  # penalty for a move which did not result in finding the exit cell
                reward=self.maze.utils.calc_points(self.__current_exit_cell,self.__current_cell)
            self.__visited.add(self.__current_cell)
        else:
            reward = Maze.penalty_impossible_move  # penalty for trying to enter an occupied cell or move out of the maze

        return reward
    def __execute2(self, action):
                """ Execute action and collect the reward or penalty.

                    :param Action action: direction in which the agent will move
                    :return float: reward or penalty which results from the action
                """
                possible_actions = self.__possible_actions(self.__current_cell2)

                if not possible_actions:
                    reward = self.__minimum_reward2 - 1  # cannot move anywhere, force end of game
                elif action in possible_actions:
                    col, row = self.__current_cell2
                    if action == Action.MOVE_LEFT:
                        col -= 1
                    elif action == Action.MOVE_UP:
                        row -= 1
                    if action == Action.MOVE_RIGHT:
                        col += 1
                    elif action == Action.MOVE_DOWN:
                        row += 1

                    self.__previous_cell2 = self.__current_cell2
                    self.__current_cell2 = (col, row)
                    #print('self.__previous_cell2',self.__previous_cell2  )
                    #print('self.__current_cell2',self.__current_cell2  )
                    if self.maze.board[self.__previous_cell[1],self.__previous_cell[0]]==1:
                        self.maze.board[self.__previous_cell[1],self.__previous_cell[0]] = 0
                    self.maze.board[self.__current_cell[1],self.__current_cell[0]] =1

                    if self.__render != Render.NOTHING:
                        self.__draw()

                    if self.__current_cell2 == self.__current_exit_cell: 
                        #reward = Maze.reward_exit  # maximum reward when reaching the exit cell
                        reward=self.maze.utils.calc_points(self.__current_exit_cell,self.__current_cell2)
                                                                            
                    elif self.__current_cell2 in self.__visited:
                        #reward = Maze.penalty_visited  # penalty when returning to a cell which was visited earlier
                        reward=self.maze.utils.calc_points(self.__current_exit_cell,self.__current_cell2)
                    else:
                        #reward = Maze.penalty_move  # penalty for a move which did not result in finding the exit cell
                        reward=self.maze.utils.calc_points(self.__current_exit_cell,self.__current_cell2)
                    self.__visited.add(self.__current_cell2)
                else:
                    reward = Maze.penalty_impossible_move  # penalty for trying to enter an occupied cell or move out of the maze

                return reward
    def __possible_actions(self, cell=None):
        """ Create a list with all possible actions from 'cell', avoiding the maze's edges and walls.

            :param tuple cell: location of the agent (optional, else use current cell)
            :return list: all possible actions
        """
        if cell is None:
            col, row = self.__current_cell
        else:
            col, row = cell

        possible_actions = Maze.actions.copy()  # initially allow all

        # now restrict the initial list by removing impossible actions
        nrows, ncols = self.maze.board.shape
        if row == 0 or (row > 0 and self.maze.board[row - 1, col] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_UP)
        if row == nrows - 1 or (row < nrows - 1 and self.maze.board[row + 1, col] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_DOWN)

        if col == 0 or (col > 0 and self.maze.board[row, col - 1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_LEFT)
        if col == ncols - 1 or (col < ncols - 1 and self.maze.board[row, col + 1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_RIGHT)

        return possible_actions

    def __status(self):
        """ Return the game status.

            :return Status: current game status (WIN, LOSE, PLAYING)
        """
        if self.__current_cell == self.__current_exit_cell:
            return Status.WIN

        if self.__total_reward < self.__minimum_reward:  # force end of game after to much loss
            return Status.LOSE

        return Status.PLAYING
    def __status2(self):
            """ Return the game status.

                :return Status: current game status (WIN, LOSE, PLAYING)
            """
            if self.__current_cell2 == self.__current_exit_cell:
                return Status.WIN

            if self.__total_reward2 < self.__minimum_reward2:  # force end of game after to much loss
                return Status.LOSE

            return Status.PLAYING   
    def __observe(self):
        """ Return the state of the maze - in this game the agents current location.

            :return numpy.array [1][2]: agents current location
        """
        return np.array([[*self.__current_cell]])
    def __observe2(self):
        """ Return the state of the maze - in this game the agents current location.

            :return numpy.array [1][2]: agents current location
        """
        return np.array([[*self.__current_cell2]])     
    def runner_pos(self):
        runner = Runner(play_type='f') 
        maze_trans = self.maze.board.T.copy()
        return runner.play(maze_trans)
       
         
    def play(self, model,model2, start_cell=(12, 7),start_cell2=(11, 7)):
        """ Play a single game, choosing the next move based a prediction from 'model'.

            :param class AbstractModel model: the prediction model to use
            :param tuple start_cell: agents initial cell (optional, else upper left)
            :return Status: WIN, LOSE
        """
        
        self.reset2(start_cell2)
        self.reset(start_cell)
       
        state = self.__observe()
        state2 = self.__observe2()
        while True:
            action2 = model2.predict(state=state2)
            #action2=random.choice(self.actions)
            state, reward2, status2 = self.step2(action2)

            if status2 in (Status.WIN, Status.LOSE):
                return status2   

            action = model.predict(state=state)
            state, reward, status = self.step(action)

            if status in (Status.WIN, Status.LOSE):
                return status
              

    def check_win_all(self, model):
        """ Check if the model wins from all possible starting cells. """
        previous = self.__render
        self.__render = Render.NOTHING  # avoid rendering anything during execution of the check games

        win = 0
        lose = 0

        for cell in self.empty:
            if self.play(model, cell) == Status.WIN:
                win += 1
                #self.turn+=1
            else:
                lose += 1
                #self.turn+=1
        
        self.__render = previous  # restore previous rendering setting

        logging.info("won: {} | lost: {} | win rate: {:.5f}".format(win, lose, win / (win + lose)))

        result = True if lose == 0 else False

        return result, win / (win + lose)
        
    def check_win_all2(self, model2):
            """ Check if the model wins from all possible starting cells. """
            previous = self.__render
            self.__render = Render.NOTHING  # avoid rendering anything during execution of the check games

            win = 0
            lose = 0

            for cell in self.empty:
                if self.play(model2, cell) == Status.WIN:
                    win += 1
                else:
                    lose += 1

            self.__render = previous  # restore previous rendering setting

            logging.info("won: {} | lost: {} | win rate: {:.5f}".format(win, lose, win / (win + lose)))

            result = True if lose == 0 else False

            return result, win / (win + lose)

    def render_q(self, model):
        """ Render the recommended action(s) for each cell as provided by 'model'.

        :param class AbstractModel model: the prediction model to use
        """
        def clip(n):
            return max(min(1, n), 0)

        if self.__render == Render.TRAINING:
            nrows, ncols = self.maze.board.shape

            self.__ax2.clear()
            self.__ax2.set_xticks(np.arange(0.5, nrows, step=1))
            self.__ax2.set_xticklabels([])
            self.__ax2.set_yticks(np.arange(0.5, ncols, step=1))
            self.__ax2.set_yticklabels([])
            self.__ax2.grid(True)
            self.__ax2.plot(*self.__current_exit_cell, "gs", markersize=30)  # exit is a big green square
            self.__ax2.text(*self.__current_exit_cell, "Runner", ha="center", va="center", color="white")

            for cell in self.empty:
                q = model.q(cell) if model is not None else [0, 0, 0, 0]
                a = np.nonzero(q == np.max(q))[0]

                for action in a:
                    dx = 0
                    dy = 0
                    if action == Action.MOVE_LEFT:
                        dx = -0.2
                    if action == Action.MOVE_RIGHT:
                        dx = +0.2
                    if action == Action.MOVE_UP:
                        dy = -0.2
                    if action == Action.MOVE_DOWN:
                        dy = 0.2

                    # color (red to green) represents the certainty
                    color = clip((q[action] - -1)/(1 - -1))

                    self.__ax2.arrow(*cell, dx, dy, color=(1 - color, color, 0), head_width=0.2, head_length=0.1)

            self.__ax2.imshow(self.maze.board, cmap="binary")
            self.__ax2.get_figure().canvas.draw()
