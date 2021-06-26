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



class PlayerMove(namedtuple("PlayerMove", ['player', 'action'])):
    """
    Player move type for all game actions.
    player is r, c1 or c2.
    action is tuple like (0, 1), (4, 5) numpy array slices.

    Ex:  player chaser1 is moving from (1,2) to (1,3)
         PlayerMove(player='c1', action=((1, 2), (1,3)))
    """

    def __str__( self ):
        return "Player: {}, Action: {}".format(self.player, self.action)

    def _as_dict( self ):
        return { 'player': self.player, 'action': self.action }