from gtp_connection import GtpConnection
from board_util import GoBoardUtil, EMPTY, BLACK, WHITE,PASS,BORDER
from simple_board import SimpleGoBoard
from ucb import runUcb

import numpy as np
import random 

def undo(board, move):
    board.board[move] = EMPTY
    board.current_player = GoBoardUtil.opponent(board.current_player)

def play_move(board, move, color):
    board.play_move(move, color)

def game_result(board):    
    legal_moves = GoBoardUtil.generate_legal_moves(board, board.current_player)
    if not legal_moves:
        result = BLACK if board.current_player == WHITE else WHITE
    else:
        result = None
    return result

class NoGoFlatMC():
    def __init__(self):
        """
        NoGo player that selects moves by flat Monte Carlo Search.
        Resigns only at the end of game.
        Replace this player's algorithm by your own.

        """
        self.name = "NoGo Assignment 4"
        self.version = 0.0
        self.simulations_per_move = 10
        self.best_move = None
        self.weights = self.parseWeights()
    
    def getPatternMoveWeights(self, state, toPlay):
        weights = {}

        for move in GoBoardUtil.generate_legal_moves(state, toPlay):
            weight = self.isPattern(state, move, toPlay)
            weights[move] = weight

        return weights

    def isPattern(self, board, point, toPlay):
        #print("Found match for move: ", format_point(point_to_coord(point, self.board.size)))
        # Get the 8 surrounding points in order from top-left to bottom-right
        neighbors = board.surroundingPoints(point)
        base10Value = 0
        count = 0

        for n in neighbors:
            if (toPlay == WHITE and board.board[n] != BORDER and board.board[n] != EMPTY):
                base10Value += pow(4, count) * GoBoardUtil.opponent(board.board[n])
            else:
                base10Value += pow(4, count) * board.board[n]
            count += 1

        weight = self.weights.get(base10Value)
        # Moves with weights of 1 are illegal patterns
        return weight

    def parseWeights(self):
        '''
        Parse weights in the file
        :return: A dictionary of the weights
        :rtype: Dictionary
        '''
        with open('nogo4/weights') as weights:
            lines = weights.readlines()

        weightDict = {}

        for line in lines:
            line.replace("\n", "")
            lineVals = line.split(" ")
            weightDict[int(lineVals[0])] = float(lineVals[1])

        return weightDict

    
    def getBestMove(self, state, toPlay):
        weightDict = self.getPatternMoveWeights(state, toPlay)
        print("finding best")
        # Generate a random move if we have no pattern matches
        if (len(weightDict) == 0):
            print("random move")
            return GoBoardUtil.generate_random_move(state, toPlay)

        weightSum = 0
        probabilities = {}

        for weight in weightDict:
            weightSum += weightDict.get(weight)

        # Calculate the probability of each move
        prev = 0

        for weight in weightDict:
            probabilities[weight] = (weightDict.get(weight) / weightSum) + prev
            prev = probabilities.get(weight)

        #print("probabilities: ", probabilities)

        uniform = random.uniform(0,1)
        for move in probabilities:
            if (uniform <= probabilities.get(move)):
                return move

    '''
    def simulateMove(self, cboard, move, color, random, N):
        win = 0
        for _ in range(N):
            result = self.run_sim(cboard, move, random, N)
            if result == color:
                win += 1
        return win
    
    def run_sim(self,board, move, random, N):

        cboard = board.copy()
        cboard.play_move(move, cboard.current_player)
        # Play through the rest of a game until it's over
        for _ in range(1000):
            color = cboard.current_player
            # uniform random
            if random:
                move = GoBoardUtil.generate_random_move(cboard, color)
            # pattern
            else:
                pass
                # Select a random pattern move based on its probability
                
                #move = self.getBestMove(cboard, color)

            cboard.play_move(move, color)
            if move == PASS:
                break
        winner = GoBoardUtil.opponent(color)
        return winner
    '''
    #analagous to run_sim
    def simulate(self, board, toplay):
        """
        Run a simulated game for a given starting move.
        """
        res = game_result(board)
        simulation_moves = []
        while (res is None):
            #print("simulating move")
            move = GoBoardUtil.generate_random_move(board, board.current_player)
            #move = self.getBestMove(board, board.current_player)
            play_move(board, move, board.current_player)
            simulation_moves.append(move)
            res = game_result(board)
        for m in simulation_moves[::-1]:
            undo(board, m)
        result = 1.0 if res == toplay else 0.0
        return result,res

    def get_move(self, original_board, color):
        print("get move")
        """
        The genmove function using one-ply MC search.
        """
        board = original_board.copy()
        moves = GoBoardUtil.generate_legal_moves(board, board.current_player)
        toplay = board.current_player
        assert color == toplay
        best_result, best_move = -1.0, None
        best_move = moves[0]
        self.best_move = moves[0]
        wins = np.zeros(len(moves))
        visits = np.zeros(len(moves))
        
        #for each potential move simulate:

        for _ in range(self.simulations_per_move):
            for i, move in enumerate(moves):
                play_move(board, move, toplay)
                res = game_result(board)
                if res == toplay:
                    # This move is a immediate win
                    undo(board, move)
                    return move 
                #ucb test
                C = 0.4
                sim_result = runUcb(self, board, C, moves, color,40)
                #sim_result,_ = self.simulate(board, toplay)
                wins[i] += sim_result
                visits[i] += 1.0
                win_rate = wins[i] / visits[i]
                if win_rate > best_result:
                    best_result = win_rate
                    best_move = move 
                    self.best_move = move 
                undo(board, move)
            assert best_move is not None 
        return best_move

def run():
    """
    start the gtp connection and wait for commands.
    """
    board = SimpleGoBoard(7)
    con = GtpConnection(NoGoFlatMC(), board)
    con.start_connection()

if __name__=='__main__':
    run()
