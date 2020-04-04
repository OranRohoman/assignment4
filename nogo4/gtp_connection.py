"""
gtp_connection.py
Module for playing games of Go using GoTextProtocol

Parts of this code were originally based on the gtp module 
in the Deep-Go project by Isaac Henrion and Amos Storkey 
at the University of Edinburgh.
"""
import traceback
from sys import stdin, stdout, stderr
from board_util import GoBoardUtil, BLACK, WHITE, EMPTY, BORDER, PASS, \
                       MAXSIZE, coord_to_point
import numpy as np
import re
import sys
from ucb import runUcb
import random
import signal


class GtpConnection():

    def __init__(self, go_engine, board, debug_mode = False):
        """
        Manage a GTP connection for a Go-playing engine

        Parameters
        ----------
        go_engine:
            a program that can reply to a set of GTP commandsbelow
        board: 
            Represents the current board state.
        """
        self._debug_mode = debug_mode
        self.go_engine = go_engine
        self.board = board
        self.N = 10
        self.random = False
        self.round_robin = False
        self.weights = self.parseWeights()
        signal.signal(signal.SIGALRM, self.handler)
        self.commands = {
            "protocol_version": self.protocol_version_cmd,
            "quit": self.quit_cmd,
            "name": self.name_cmd,
            "boardsize": self.boardsize_cmd,
            "showboard": self.showboard_cmd,
            "clear_board": self.clear_board_cmd,
            "komi": self.komi_cmd,
            "version": self.version_cmd,
            "known_command": self.known_command_cmd,
            "genmove": self.genmove_cmd,
            "list_commands": self.list_commands_cmd,
            "play": self.play_cmd,
            "legal_moves": self.legal_moves_cmd,
            "gogui-rules_game_id": self.gogui_rules_game_id_cmd,
            "gogui-rules_board_size": self.gogui_rules_board_size_cmd,
            "gogui-rules_legal_moves": self.gogui_rules_legal_moves_cmd,
            "gogui-rules_side_to_move": self.gogui_rules_side_to_move_cmd,
            "gogui-rules_board": self.gogui_rules_board_cmd,
            "gogui-rules_final_result": self.gogui_rules_final_result_cmd,
            "gogui-analyze_commands": self.gogui_analyze_cmd,
            "timelimit": self.timelimit_cmd
        }
        self.timelimit = 30

        # used for argument checking
        # values: (required number of arguments,
        #          error message on argnum failure)
        self.argmap = {
            "boardsize": (1, 'Usage: boardsize INT'),
            "komi": (1, 'Usage: komi FLOAT'),
            "known_command": (1, 'Usage: known_command CMD_NAME'),
            "genmove": (1, 'Usage: genmove {w,b}'),
            "play": (2, 'Usage: play {b,w} MOVE'),
            "legal_moves": (1, 'Usage: legal_moves {w,b}')
        }

    def timelimit_cmd(self, args):
        self.timelimit = args[0]
        self.respond('')

    def handler(self, signum, fram):
        self.board = self.sboard
        raise Exception("unknown")

    def write(self, data):
        stdout.write(data)

    def flush(self):
        stdout.flush()

    def start_connection(self):
        """
        Start a GTP connection.
        This function continuously monitors standard input for commands.
        """
        line = stdin.readline()
        while line:
            self.get_cmd(line)
            line = stdin.readline()

    def get_cmd(self, command):
        """
        Parse command string and execute it
        """
        if len(command.strip(' \r\t')) == 0:
            return
        if command[0] == '#':
            return
        # Strip leading numbers from regression tests
        if command[0].isdigit():
            command = re.sub("^\d+", "", command).lstrip()

        elements = command.split()
        if not elements:
            return
        command_name = elements[0];
        args = elements[1:]
        if self.has_arg_error(command_name, len(args)):
            return
        if command_name in self.commands:
            try:
                self.commands[command_name](args)
            except Exception as e:
                self.debug_msg("Error executing command {}\n".format(str(e)))
                self.debug_msg("Stack Trace:\n{}\n".
                               format(traceback.format_exc()))
                raise e
        else:
            self.debug_msg("Unknown command: {}\n".format(command_name))
            self.error('Unknown command')
            stdout.flush()

    def has_arg_error(self, cmd, argnum):
        """
        Verify the number of arguments of cmd.
        argnum is the number of parsed arguments
        """
        if cmd in self.argmap and self.argmap[cmd][0] != argnum:
            self.error(self.argmap[cmd][1])
            return True
        return False

    def debug_msg(self, msg):
        """ Write msg to the debug stream """
        if self._debug_mode:
            stderr.write(msg)
            stderr.flush()

    def error(self, error_msg):
        """ Send error msg to stdout """
        stdout.write('? {}\n\n'.format(error_msg))
        stdout.flush()

    def respond(self, response=''):
        """ Send response to stdout """
        stdout.write('= {}\n\n'.format(response))
        stdout.flush()

    def reset(self, size):
        """
        Reset the board to empty board of given size
        """
        self.board.reset(size)

    def board2d(self):
        return str(GoBoardUtil.get_twoD_board(self.board))

    def protocol_version_cmd(self, args):
        """ Return the GTP protocol version being used (always 2) """
        self.respond('2')

    def quit_cmd(self, args):
        """ Quit game and exit the GTP interface """
        self.respond()
        exit()

    def name_cmd(self, args):
        """ Return the name of the Go engine """
        self.respond(self.go_engine.name)

    def version_cmd(self, args):
        """ Return the version of the  Go engine """
        self.respond(self.go_engine.version)

    def clear_board_cmd(self, args):
        """ clear the board """
        self.reset(self.board.size)
        self.respond()

    def boardsize_cmd(self, args):
        """
        Reset the game with new boardsize args[0]
        """
        self.reset(int(args[0]))
        self.respond()

    def showboard_cmd(self, args):
        self.respond('\n' + self.board2d())

    def komi_cmd(self, args):
        """
        Set the engine's komi to args[0]
        """
        self.go_engine.komi = float(args[0])
        self.respond()

    def known_command_cmd(self, args):
        """
        Check if command args[0] is known to the GTP interface
        """
        if args[0] in self.commands:
            self.respond("true")
        else:
            self.respond("false")

    def list_commands_cmd(self, args):
        """ list all supported GTP commands """
        self.respond(' '.join(list(self.commands.keys())))

    def legal_moves_cmd(self, args):
        """
        List legal moves for color args[0] in {'b','w'}
        """
        board_color = args[0].lower()
        color = color_to_int(board_color)
        moves = GoBoardUtil.generate_legal_moves(self.board, color)
        gtp_moves = []
        for move in moves:
            coords = point_to_coord(move, self.board.size)
            gtp_moves.append(format_point(coords))
        sorted_moves = ' '.join(sorted(gtp_moves))
        self.respond(sorted_moves)

    def play_cmd(self, args):
        """
        play a move args[1] for given color args[0] in {'b','w'}
        """
        try:
            board_color = args[0].lower()
            board_move = args[1]
            if board_color != "b" and board_color !="w":
                self.respond("illegal move: \"{}\" wrong color".format(board_color))
                return
            color = color_to_int(board_color)
            if args[1].lower() == 'pass':
                self.respond("illegal move: \"{} {}\" wrong coordinate".format(args[0], args[1]))
                return
            coord = move_to_coord(args[1], self.board.size)
            if coord:
                move = coord_to_point(coord[0],coord[1], self.board.size)
            else:
                self.error("Error executing move {} converted from {}"
                           .format(move, args[1]))
                return
            if not self.board.play_move(move, color):
                self.respond("illegal move: \"{} {}\" ".format(args[0], board_move))
                return
            else:
                self.debug_msg("Move: {}\nBoard:\n{}\n".
                                format(board_move, self.board2d()))
            self.respond()
        except Exception as e:
            self.respond('illegal move: \"{} {}\" {}'.format(args[0], args[1], str(e)))

    def genmove_cmd(self, args):
        """
        Generate a move for the color args[0] in {'b', 'w'}, for the game of gomoku.
        """
        board_color = args[0].lower()
        color = color_to_int(board_color)


        try:
            signal.alarm(int(self.timelimit))
            self.sboard = self.board.copy()
            move = self.get_move(self.board, color, self.round_robin, self.random, self.N)
            self.board=self.sboard
            signal.alarm(0)
        except Exception as e:
            # TODO fix this to give the best move we can find
            self.respond("OUT OF TIME")
            #move=self.go_engine.best_move
            move = None

        if move is None:
            self.respond("resign")
            self.board.current_player = GoBoardUtil.opponent(self.board.current_player)
            return
      

        move_coord = point_to_coord(move, self.board.size)
        move_as_string = format_point(move_coord)
        if self.board.is_legal(move, color):
            self.board.play_move(move, color)
            self.respond(move_as_string)
        else:
            self.respond("resign")

    def gogui_rules_game_id_cmd(self, args):
        self.respond("NoGo")
    
    def gogui_rules_board_size_cmd(self, args):
        self.respond(str(self.board.size))


    def gogui_rules_legal_moves_cmd(self, args):
        empties = self.board.get_empty_points()
        color = self.board.current_player
        legal_moves = []
        for move in empties:
            if self.board.is_legal(move, color):
                legal_moves.append(move)

        gtp_moves = []
        for move in legal_moves:
            coords = point_to_coord(move, self.board.size)
            gtp_moves.append(format_point(coords))
        sorted_moves = ' '.join(sorted(gtp_moves))
        self.respond(sorted_moves)
    
    def gogui_rules_side_to_move_cmd(self, args):
        color = "black" if self.board.current_player == BLACK else "white"
        self.respond(color)
    
    def gogui_rules_board_cmd(self, args):
        size = self.board.size
        str = ''
        for row in range(size-1, -1, -1):
            start = self.board.row_start(row + 1)
            for i in range(size):
                point = self.board.board[start + i]
                if point == BLACK:
                    str += 'X'
                elif point == WHITE:
                    str += 'O'
                elif point == EMPTY:
                    str += '.'
                else:
                    assert False
            str += '\n'
        self.respond(str)
    
    def gogui_rules_final_result_cmd(self, args):
        empties = self.board.get_empty_points()
        color = self.board.current_player
        legal_moves = []
        for move in empties:
            if self.board.is_legal(move, color):
                legal_moves.append(move)
        if not legal_moves:
            result = "black" if self.board.current_player == WHITE else "white"
        else:
            result = "unknown"
        self.respond(result)

    def gogui_analyze_cmd(self, args):
        self.respond("pstring/Legal Moves For ToPlay/gogui-rules_legal_moves\n"
                     "pstring/Side to Play/gogui-rules_side_to_move\n"
                     "pstring/Final Result/gogui-rules_final_result\n"
                     "pstring/Board Size/gogui-rules_board_size\n"
                     "pstring/Rules GameID/gogui-rules_game_id\n"
                     "pstring/Show Board/gogui-rules_board\n"
                     )
    
    def set_policy(self,args):
        if(len(args) != 0):
            if(args[0] == "random"):
                self.random = True
            if(args[0] == "pattern"):
                self.random = False
            self.respond()

    def set_selection(self,args):
        if(len(args) != 0):
            if(args[0] == "rr"):
                self.round_robin = True
            if(args[0] == "ucb"):
                self.round_robin = False
            self.respond()

    def get_policy_moves(self, args):
        moveWeights = []
        if (not self.random):
            moveWeights = self.getPatternMoveProbabilities(self.board, self.board.current_player)
        else:
            moveWeights = self.get_random_move_prob(self.board,self.board.current_player)
        
        self.respond(str(moveWeights))


    def set_N(self,args):
        if(len(args) >0):
            self.N = int(args[0])

        self.respond()

    def parseWeights(self):
        '''
        Parse weights in the file
        :return: A dictionary of the weights
        :rtype: Dictionary
        '''
        try:
            with open('nogo4/weights') as weights:
                lines = weights.readlines()
        except:
            with open('weights') as weights:
                lines = weights.readlines()

        weightDict = {}

        for line in lines:
            line.replace("\n", "")
            lineVals = line.split(" ")
            weightDict[int(lineVals[0])] = float(lineVals[1])

        return weightDict
    def get_random_move_prob(self,state,toPlay):
        legal_moves = GoBoardUtil.generate_legal_moves(state,toPlay)
        moves = []
        for move in legal_moves:
            moves.append(format_point(point_to_coord(move,self.board.size)))
        prob = str(round(1/len(legal_moves),3))
        moves.sort()
        probs = [prob] * len(legal_moves)
        return "" + " ".join(moves + probs) 
        


    def getPatternMoveProbabilities(self, state, toPlay):
        weightDict = self.getPatternMoveWeights(state, toPlay)

        # Return empty array if the policy has no options
        if (len(weightDict) == 0):
            return []

        weightSum = 0
        probabilities = {}

        for weight in weightDict:
            weightSum += weightDict.get(weight)

        # Calculate the probability of each move
        for weight in weightDict:
            probabilities[format_point(point_to_coord(weight, self.board.size))] = (weightDict.get(weight) / weightSum)


        moves = []
        probs = []
        for move in sorted(probabilities.items()):
            moves.append(move[0])
            probs.append(str(round(probabilities.get(move[0]), 3)))

        return "" + " ".join(moves + probs) 


    def getPatternMoveWeights(self, state, toPlay):
        weights = {}

        for move in GoBoardUtil.generate_legal_moves(state, toPlay):
            weight = self.isPattern(state, move, toPlay)
            weights[move] = weight

        return weights

    def getBestMove(self, state, toPlay):
        weightDict = self.getPatternMoveWeights(state, toPlay)

        # Generate a random move if we have no pattern matches
        if (len(weightDict) == 0):
            return GoBoardUtil.generate_random_move(state, toPlay, True)

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

    def run_sim(self, board, move, random, N):

        cboard = board.copy()
        cboard.play_move(move, cboard.current_player)
        # Play through the rest of a game until it's over
        for _ in range(1000):
            color = cboard.current_player
            # uniform random
            if random:
                move = GoBoardUtil.generate_random_move(cboard, color, False)
            # pattern
            else:
                # Select a random pattern move based on its probability
                move = self.getBestMove(cboard, color)

            cboard.play_move(move, color)
            if move == PASS:
                break
        winner = GoBoardUtil.opponent(color)
        return winner

    def simulateMove(self, cboard, move, color, random, N):
        win = 0
        for _ in range(N):
            result = self.run_sim(cboard, move, random, N)
            if result == color:
                win += 1
        return win

    def get_move(self, board, color, round_robin, random, N):
        ""
        cboard = board.copy()
        moves = GoBoardUtil.generate_legal_moves(cboard, cboard.current_player)

        # UCB is selected, run it and find the best move
        if not round_robin:
            C = 0.4  # sqrt(2) is safe, this is more aggressive
            best = runUcb(self, cboard, C, moves, color,N)
            return best
        else:
            moveWins = []
            for move in moves:
                wins = self.simulateMove(cboard, move, color, random, N)
                moveWins.append(wins)
            #writeMoves(cboard, moves, moveWins, N)
            return select_best_move(board, moves, moveWins)

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




def byPercentage(pair):
    return pair[1]

def writeMoves(board, moves, count, numSimulations):
    """
    Write simulation results for each move.
    """
    gtp_moves = []
    for i in range(len(moves)):
        if moves[i] != None:
            x, y = point_to_coord(moves[i], board.size)
            gtp_moves.append((format_point((x, y)),
                              float(count[i])/float(numSimulations)))
        else:
            gtp_moves.append(('Pass',float(count[i])/float(numSimulations)))
    sys.stderr.write("win rates: {}\n"
                     .format(sorted(gtp_moves, key = byPercentage,
                                    reverse = True)))

def select_best_move(board, moves, moveWins):
    """
    Move select after the search.
    """
    max_child = np.argmax(moveWins)
    return moves[max_child]

sys.stderr.flush()

def point_to_coord(point, boardsize):
    """
    Transform point given as board array index 
    to (row, col) coordinate representation.
    Special case: PASS is not transformed
    """
    if point == PASS:
        return PASS
    else:
        NS = boardsize + 1
        return divmod(point, NS)

def format_point(move):
    """
    Return move coordinates as a string such as 'a1', or 'pass'.
    """
    column_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    #column_letters = "abcdefghjklmnopqrstuvwxyz"
    if move == PASS:
        return "pass"
    row, col = move
    if not 0 <= row < MAXSIZE or not 0 <= col < MAXSIZE:
        raise ValueError
    return column_letters[col - 1]+ str(row) 
    
def move_to_coord(point_str, board_size):
    """
    Convert a string point_str representing a point, as specified by GTP,
    to a pair of coordinates (row, col) in range 1 .. board_size.
    Raises ValueError if point_str is invalid
    """
    if not 2 <= board_size <= MAXSIZE:
        raise ValueError("board_size out of range")
    s = point_str.lower()
    if s == "pass":
        return PASS
    try:
        col_c = s[0]
        if (not "a" <= col_c <= "z") or col_c == "i":
            raise ValueError
        col = ord(col_c) - ord("a")
        if col_c < "i":
            col += 1
        row = int(s[1:])
        if row < 1:
            raise ValueError
    except (IndexError, ValueError):
        # e.g. "a0"
        raise ValueError("wrong coordinate")
    if not (col <= board_size and row <= board_size):
        # e.g. "a20"
        raise ValueError("wrong coordinate")
    return row, col

def color_to_int(c):
    """convert character to the appropriate integer code"""
    color_to_int = {"b": BLACK , "w": WHITE, "e": EMPTY, 
                    "BORDER": BORDER}
    return color_to_int[c]

