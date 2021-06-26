
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

class Utils:

    def __init__( self, board_size=(8, 13) ):
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.columns = list(string.ascii_lowercase[:board_size[1]])
        self.rows = list(range(board_size[0], 0, -1))

        self.players = { 'r': 'Runner', 'c1': 'Chaser1', 'c2': 'Chaser2' }

    def get_neigbours( self, board, position ):
        #assert isinstance(position, tuple)
        x, y = position[0], position[1]
        neighbours = []

        for d in self.directions:
            if (0 <= d[0] + x < board.shape[0]) and (0 <= d[1] + y < board.shape[1]):
                if board[(x + d[0], y + d[1])] < 99:
                    neighbours.extend([(x + d[0], y + d[1])])
        neighbours.append(position)
        #print(neighbours)
        return neighbours

    def get_position( self, board, player ):
        """
        Return tuple of current position of a player, ex (2, 3)
        :param board: numpy array of board.
        :param player: r, c1 or c2
        :return: tuple
        """
        assert player in ('r', 'c1', 'c2')
        # get the runner and chasers positions.
        # np.where is return tuple of array.
        # this is convert it to tuple of integer positions.
        if player == 'r':
            player = -1
        elif player == 'c1':
            player = 1
        elif player=='c2':
            player = 2
 
        # res = tuple(x[0] for x in np.where(board == player))
     
        res = np.where(board == player)
        # print(f"RES: {res}")

        # 12, 21, -12 or -11
        # if the cell has another player in board.
        # I need to research for it.
        if res[0].size == 0:
            if player == -1:
                res = np.where(np.in1d(board, [-11, -12]).reshape(board.shape[0], board.shape[1]))
            elif player == 1 or player == 2:
                res = np.where(np.in1d(board, [21, 12]).reshape(board.shape[0], board.shape[1]))
                if res[0].size == 0:
                    res = np.where(np.in1d(board, [-12, -11]).reshape(board.shape[0], board.shape[1]))
            else:
                res = np.where(np.in1d(board, [-121, -112]).reshape(board.shape[0], board.shape[1]))

        return tuple(x[0] for x in res)

    def convert_df_to_np_index( self, index ):
        return self.rows.index(int(index[0])), self.columns.index(index[1])

    def convert_np_to_df_index( self, index ):
        return self.rows[int(index[0])], self.columns[int(index[1])]

    def calc_points( self, runner_pos, chaser_pos ):
        dist = self.distance(runner_pos, chaser_pos)
        if dist == 2:
            return 1
        elif dist == 1:
            return 2
        else:
            return 0

    def distance( self, p1, p2 ):
        """
        Finds the manhattan distance between 2 points.

        :param p1: Tuple, like (1,2)
        :param p2: Tuple, like (3,4)
        :return: int, distance
        """
        assert isinstance(p1, tuple) and isinstance(p2, tuple)
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def convert_arr_to_df( self, data ):
        # noinspection PyCompatibility
        # rows = [*range(8, 0, -1)]
        # noinspection PyCompatibility
        # columns = [*string.ascii_lowercase[:8]]
        conditions = [data == 2, data == 1
            , data == 0, data == -1
            , data == 99, data == 12
            , data == 21, data == -12
            , data == -11]
        choices = ['C2', 'C1', '-', 'R', 'X', 'C1,C2', 'C2,C1', 'R,C2', 'R,C1']
        return pd.DataFrame(np.select(conditions, choices)
                            , index=self.rows
                            , columns=self.columns)