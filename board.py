import os
import string
from collections import namedtuple
import random
import uuid
import pickle
import datetime
import numpy as np
import pandas as pd
#import click
from environment.utils import Utils
#from playermove import PlayerMove

class Board:
    """
    Holds all board information for game.
    """

    def __init__( self, utils
                  , board_size=(8, 13)
                  , block_number=20
                  , block_assign_type='r'
                  ):
        """
        Initialize function for board class.

        :param utils: Utils class instance. Utils class contains helper functions.
        :param board_size: Board size of game, default is (8, 13)
        :param block_assign_type: information for creating board random or custom, default is r (random).
        """

        self.utils = utils
        self.board_size = board_size
        self.board = self.__create_empty_board__

        self.block_number = block_number
        self.block_assign_type = block_assign_type

        # self.columns = list(string.ascii_lowercase[:13])
        # self.rows = list(range(8, 0, -1))
        self.columns = self.utils.columns
        self.rows = self.utils.rows
        
        if self.block_assign_type == 'r':
            self.board = self.assign_random_block
        else:
            self.board = self.assign_blocks_manually
        
        #self.board = self.assign_random_block
       

        self.board_df = self.utils.convert_arr_to_df(self.board)

    @property
    def __create_empty_board__( self ):
        """
        Creates empty initial board.
        """
        empty_board = np.full(self.board_size, 0, dtype=int)
        empty_board[4, 4] = -1
        empty_board[-1, -2] = 1
        empty_board[-1, -1] = 2

        return empty_board

    @property
    def assign_random_block( self ):

        index_array = np.arange(self.board.size)
        board = self.board.copy()
        np.random.shuffle(index_array)
         
        blocks = index_array[np.where(np.in1d(index_array, [0, 102, 103]), False, True)][:self.block_number] 

        for block in blocks:
            if board.reshape(-1)[0:][block] not in (-1, 1, 2):
                board.reshape(-1)[0:][block] = 99

        return board

    @property
    def assign_blocks_manually( self ):

        board = self.board.copy()

        # for i in range(self.block_number):
        i = 0
        while i < self.block_number:
            resp = input(f"Enter the position of the {i + 1}. black cell: (ex: 6d) ")
            if len(resp) == 2:
                if resp[0] in str(self.rows) and resp[1] in self.columns:
                    if board[self.rows.index(int(resp[0])), self.columns.index(resp[1])] not in (-1, 1, 2):
                        board[self.rows.index(int(resp[0])), self.columns.index(resp[1])] = 99
                        i += 1
                    else:
                        print(f"Try again, {resp} is wrong. Selected cell contains chaser or runner. ")
                        continue
                else:
                    print(f"Try again, {resp} is wrong. row must be in 1-8 and column must in a-m.")
                    continue
            else:
                print(f"Try again, {resp} must be 2 character, row and column.  ")
                continue

        return board