from environment.playermove import PlayerMove
import os
import string
from collections import namedtuple
import random
import uuid
import pickle
import datetime
import numpy as np
import pandas as pd
import click
from environment.utils import Utils

class Runner:
    global r_pos
    def __init__( self, play_type ):
        """
        Initialize function for Runner.
        :param play_type: f for function, r for random, m for manuel and n for neural network.
        """
        # create utils instance for helper functions.
        self.utils = Utils()
        self.play_type = play_type
        r_pos=[4,4]
    @property
    def get_player_name( self ):
        return "Runner"

    def random_play( self, pos_locations ):
        return random.choice(pos_locations)

    def manuel_play( self, pos_locations ):

        is_input_true = False
        df_locations = []
        for pos in pos_locations:
            df_locations.append(self.utils.convert_np_to_df_index(pos))

        while not is_input_true:
            #print(f"Possible Locations: {df_locations}")
            resp = input(f"Enter the next position for Runner: ")
            index = self.utils.convert_df_to_np_index(resp)
            if index in pos_locations:
                is_input_true = True
                return index
            else:
                print(f"{resp} is not legal move. Try again. ")
                is_input_true = False

    def function_play( self, c1, c2, possible_locations ):
        distances = { }

        for pos in possible_locations:
            distances[pos] = self.utils.distance(pos, c1) + self.utils.distance(pos, c2)

        # find the farthest point from Runner.
        # if there are two same distance then selects random one.
        # because python dict type has no order.
        return max(distances, key=distances.get)
    
    def play( self, board ):
        # get the runner and chasers positions.
        self.board=board
        maze_trans = self.board.T.copy()
        r_pos = self.utils.get_position(board, 'r') # according to maze
        c1_pos = self.utils.get_position(board, 'c1')
        #print(c1_pos)
        c2_pos = self.utils.get_position(board, 'c2')
        # find possible positions.
        board_rpos=(r_pos[0],r_pos[1])
        possible_locations = self.utils.get_neigbours(board, board_rpos)
  # return selected position.
        if self.play_type == 'f':
            aa=PlayerMove(player='r', action=(board_rpos, self.function_play(c1_pos, c2_pos, possible_locations)))
            #print(aa)
            return aa
        elif self.play_type == 'r':
            return PlayerMove(player='r', action=(board_rpos, self.random_play(possible_locations)))
            #return self.random_play(possible_locations)