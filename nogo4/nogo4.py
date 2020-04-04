#!/usr/bin/python3
#/usr/local/bin/python3
# Set the path to your python3 above

from gtp_connection import GtpConnection
from board_util import GoBoardUtil
from board_util import GoBoardUtil, BLACK, WHITE, EMPTY, BORDER, PASS
import numpy as np
from gtp_connection import point_to_coord, format_point
from simple_board import SimpleGoBoard
from board_util import GoBoardUtil, EMPTY, PASS, BORDER
from ucb import runUcb
import sys






class Nogo():
    def __init__(self):
        """
        NoGo player that selects moves randomly 
        from the set of legal moves.
        Passe/resigns only at the end of game.

        """
        self.name = "NoGoAssignment2"
        self.version = 1.0


    
def run():
    """
    start the gtp connection and wait for commands.
    """
    board = SimpleGoBoard(7)
    con = GtpConnection(Nogo(), board)
    con.start_connection()

if __name__=='__main__':
    run()
