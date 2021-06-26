import logging
from enum import Enum, auto
import matplotlib.pyplot as plt
import numpy as np
import models
from environment.maze import Maze, Render
from environment.board import Board
from environment.utils import Utils
from environment.runner import Runner
from environment.maze import Status
import click
import joblib

logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)  # Only show messages *equal to or above* this level

class Test(Enum):
    SHOW_MAZE_ONLY = auto()
    RANDOM_MODEL = auto()
    Q_LEARNING = auto()


@click.command()
@click.option('--turn', prompt='Enter the number of turns', type=int)
@click.option('--black-cells', prompt='Enter the number of black cells on the board', type=int)
@click.option('--black-cell-type', type=click.Choice(['r', 'm'], case_sensitive=False)
    , prompt="Are the cells going to be determined randomly or manually ")
# , callback=bring_board, expose_value=True)
# Callback function can use for sending value to a function before next prompt
@click.option('--user', type=click.Choice(['c1', 'c2'], case_sensitive=False)
    , prompt="Which user are you ")
# @click.option('--user-control', type=click.Choice(['r','m', 'n'], case_sensitive=False)
#               , prompt="Enter the user control type ")
@click.option('--runner-type', prompt="Are the runner controlled by function, random or manually "
    , type=click.Choice(['f', 'r', 'm']))
@click.option('--random-game', type=int, default=0)
@click.option('--save-game-hist', type=bool, default=False)
@click.option('--board-size', type=tuple, default=(8, 13))



def main( turn, black_cells, black_cell_type, user, runner_type, random_game, save_game_hist, board_size ):
    board_size = board_size
    turn=turn
    if random_game == 0:
        # create instance for util class.
        utils = Utils(board_size=board_size)
        # create board class
        board = Board(utils=utils
                      , board_size=board_size
                      , block_assign_type=black_cell_type
                      , block_number=black_cells)
         # create runner class
        runner = Runner(play_type=runner_type) 
        

        test = Test.Q_LEARNING  # which test to run    
        # create runner class
        #runner = Runner(play_type=runner_type)
        # only show the maze
        if test == Test.SHOW_MAZE_ONLY:
            #degisken=runner.play(board)                   
            #print(degisken)
            #game = Maze(board,runner=degisken)   
            game = Maze(board) 
            game.render(Render.MOVES)
            game.reset()

        # play using random model
        if test == Test.RANDOM_MODEL:
            degisken=runner.play(board)                   
            print(degisken)
            game = Maze(board,runner=degisken)   
            game.render(Render.MOVES)
            model = models.RandomModel(game)
            game.play(model, start_cell=(12, 7))

        # train using tabular Q-learning
        if test == Test.Q_LEARNING:
            #degisken=runner.play(board)                   
            #print(degisken)
            game = Maze(board,runner=runner)  
           # game = Maze(board) 
            game.render(Render.TRAINING)
            model = models.QTableModel(game, name="QTableModel")
            #h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
             #                           stop_at_convergence=True) 

            game.render(Render.TRAINING)
            model2 = models.QTableModel(game, name="QTableModel")
            
            #h, w, _, _ = model2.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
            #                            stop_at_convergence=True)
            """
            h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                                        stop_at_convergence=True) 
            #model2 = models.QTableTraceModel(game)
            h1, w1, _, _ = model2.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                                       stop_at_convergence=True)                                            
          
            
                # draw graphs showing development of win rate and cumulative rewards
        try:
            h  # force a NameError exception if h does not 1st, and thus don't try to show win rate and cumulative reward
            fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
            fig.canvas.set_window_title(model.name)
            ax1.plot(*zip(*w))
            ax1.set_xlabel("episode")
            ax1.set_ylabel("win rate")
            ax2.plot(h)
                        
            ax2.set_xlabel("episode")
            ax2.set_ylabel("cumulative reward")
            plt.show()    
             
        except NameError:
            pass
        """
        game.render(Render.MOVES)
        game.play(model,model2, start_cell=(12, 7),start_cell2=(11, 7))

        plt.show()  # must be placed here else the image disappears immediately at the end of the program
    else:
        for ind in range(random_game):
            # create instance for util class.
            utils = Utils(board_size=board_size)
            # create board class
            board = Board(utils=utils
                        , board_size=board_size
                        , block_assign_type=black_cell_type
                        , block_number=black_cells)
            # create runner class
            runner = Runner(play_type=runner_type) 
            
            test = Test.Q_LEARNING  # which test to run    
            # create runner class
            #runner = Runner(play_type=runner_type)
            # only show the maze
            if test == Test.SHOW_MAZE_ONLY:
                #degisken=runner.play(board)                   
                #print(degisken)
                #game = Maze(board,runner=degisken)   
                game = Maze(board) 
                game.render(Render.MOVES)
                game.reset()

            # play using random model
            if test == Test.RANDOM_MODEL:
                degisken=runner.play(board)                   
                print(degisken)
                game = Maze(board,runner=degisken)   
                game.render(Render.MOVES)
                model = models.RandomModel(game)
                game.play(model, start_cell=(12, 7))

            # train using tabular Q-learning
            if test == Test.Q_LEARNING:
                #degisken=runner.play(board)                   
                #print(degisken)
                game = Maze(board,runner=runner)  
            # game = Maze(board) 
                game.render(Render.TRAINING)
                model = models.QTableModel(game, name="QTableModel")
                #h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                #                           stop_at_convergence=True) 

                game.render(Render.TRAINING)
                model2 = models.QTableModel(game, name="QTableModel")
                
                #h, w, _, _ = model2.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                #                            stop_at_convergence=True)
                """
                h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                                            stop_at_convergence=True) 
                #model2 = models.QTableTraceModel(game)
                h1, w1, _, _ = model2.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                                        stop_at_convergence=True)                                            
            
                
                    # draw graphs showing development of win rate and cumulative rewards
            try:
                h  # force a NameError exception if h does not 1st, and thus don't try to show win rate and cumulative reward
                fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
                fig.canvas.set_window_title(model.name)
                ax1.plot(*zip(*w))
                ax1.set_xlabel("episode")
                ax1.set_ylabel("win rate")
                ax2.plot(h)
                            
                ax2.set_xlabel("episode")
                ax2.set_ylabel("cumulative reward")
                plt.show()    
                
            except NameError:
                pass
            """
            game.render(Render.MOVES)
            game.play(model,model2, start_cell=(12, 7),start_cell2=(11, 7))

            plt.show()  # must be placed here else the image disappears immediately at the end of the program

       
if __name__ == "__main__":
    main()
 






