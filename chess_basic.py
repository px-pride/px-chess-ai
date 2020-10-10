import torch
import display
import chess
import random
import math
import time
import sys
from pprint import pprint
from collections import namedtuple
from collections import defaultdict
from interruptingcow import timeout
from dataclasses import dataclass
from scipy.special import softmax

import cython
if cython.compiled:
    print("Yep, I'm compiled.")
else:
    print("Just a lowly interpreted script.")


class PxBoard(chess.Board):

    def better_result(self):
        original_result = self.result(claim_draw=True)
        if original_result == "1-0":
            return chess.WHITE
        elif original_result == "0-1":
            return chess.BLACK
        else:
            return None

    def print(self):
        print(self)
        '''
        original_string = str(self)
        piece_map = {
            'P': u'\u2659',
            'N': u'\u2658',
            'B': u'\u2657',
            'R': u'\u2656',
            'Q': u'\u2655',
            'K': u'\u2654',
            'p': u'\u265F',
            'n': u'\u265E',
            'b': u'\u265D',
            'r': u'\u265C',
            'q': u'\u265B',
            'k': u'\u265A',
        }
        print_repr = original_string
        for key in piece_map:
            print_repr = print_repr.replace(key, piece_map[key])
        print(print_repr)
        '''

class Agent:
    def __init__(self):
        self.cache = dict()
        #self.cache = defaultdict(lambda: float("-inf"), self.cache)

    def start_game(self, board, color):
        self.board = board
        self.color = color
        self.cache = dict()
        #self.cache = defaultdict(lambda: float("-inf"), self.cache)
        self.cache_calls = 0
        self.eval_calls = 0


class HumanAgent(Agent):
    def __init__(self):
        super().__init__()

    def get_move(self):
        move = None
        move_uci = None
        while True:
            try:
                move_uci = input("Your move: ")
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                print("Not a move at all.")
                continue
            if move in self.board.legal_moves:
                break
            else:
                print("Not a legal move.")
        return move_uci


class RandomCpuAgent(Agent):
    def __init__(self):
        super().__init__()

    def get_move(self):
        legal_moves = list(self.board.legal_moves)
        return random.choice(legal_moves).uci()


class TreeNode:
    def __init__(self, position, move, parents):
        self.position = position
        self.move = move
        self.parents = parents
        self.children = []
        self.numerator = 0.5
        self.denominator =  1.0
        self.flag = 0.0
        self.gg = False

    def payout(self):
        return self.numerator / float(self.denominator)
        

class AbstractMctsAgent(Agent):

    def __init__(self):
        super().__init__()
        self.node_tree = None
        self.node_dict = None

    def start_game(self, board, color):
        super().start_game(board, color)
        self.node_tree = TreeNode(position=chess.STARTING_FEN, move="", parents=[])
        self.node_dict = {chess.STARTING_FEN: self.node_tree}

    def mcts(self, board=None, num_steps=100):
        if board is None:
            board = self.board
        fen = board.fen()
        if fen not in self.node_dict:
            parent_board = board.copy()
            parent_board.pop()
            parent_fen = parent_board.fen()
            self.node_dict[fen] = TreeNode(
                position=fen, move=board.peek(), parents=[self.node_dict[parent_fen]])
        root = self.node_dict[fen]

        for t in range(num_steps):
            print(t)
            # 1. select node to expand
            current_time = time.time()
            selected_node = self.select_node(root)
            print("1\t" + str(time.time() - current_time))
            current_time = time.time()
            # 2. look at children of selected node and choose one
            expanding_child = self.expand(selected_node)
            print("2\t" + str(time.time() - current_time))
            current_time = time.time()
            if expanding_child:
                # 3. play a full random game from selected node
                result = self.play_game(expanding_child)
                print("3\t" + str(time.time() - current_time))
                current_time = time.time()
                # 4. backprop result
                self.backprop(expanding_child, result)
                print("4\t" + str(time.time() - current_time))
                current_time = time.time()
        
        best_child = max(root.children, key=lambda x: x.payout())
        return best_child.payout(), best_child.move

    def select_node(self, root):
        raise NotImplementedError("Abstract agent.")

    def expand(self, node):
        raise NotImplementedError("Abstract agent.")

    def play_game(self, node):
        # returns 1 for win, 0 for loss, 0.5 for draw
        raise NotImplementedError("Abstract agent.")

    def backprop(self, child_node, result):
        nodes_to_update = [child_node]
        nodes_to_clear_flags = []
        while len(nodes_to_update) > 0:
            current_node = nodes_to_update[0]
            current_node.flag += 1
            if current_node.flag < 60:
                for node in current_node.parents:
                    nodes_to_update.append(node)
                current_node.denominator += 1
                current_node.numerator += result
            nodes_to_clear_flags.append(current_node)
            del nodes_to_update[0]
        for node in nodes_to_clear_flags:
            node.flag = 0

    def get_move(self):
        #best_value, best_move = self.alphabeta(depth=4)
        best_value, best_move = self.mcts(num_steps=10)
        print(str(best_move) + ":\t" + str(best_value))
        return best_move.uci()


class DumbMctsAgent(AbstractMctsAgent):
    def select_node(self, root):
        selected_node = root
        while len(selected_node.children) > 0:
            random_selection = random.choice(selected_node.children)
            selected_node = random_selection
        return selected_node

        '''
        selected_node_stack = [root]
        while len(selected_node_stack[-1].children) > 0:
            not_game_over_kids = []
            for child in selected_node_stack[-1].children:
                if not child.gg:
                    not_game_over_kids.append(child)
            if len(not_game_over_kids) == 0:
                del selected_node_stack[-1]
            else:
                random_selection = random.choice(not_game_over_kids)
                selected_node_stack.append(random_selection)
        return selected_node_stack[-1]
        '''

    def select_random_weighted_move(self, fen):
        board = PxBoard(fen)
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 0:
            return None, board
        legal_values = []
        for i in range(len(legal_moves)):
            board_copy = board.copy()
            board_copy.push(legal_moves[i])
            value = self.evaluate_board(board_copy)
            if self.color != board.turn:
                value = -value
            legal_values.append(value)
        legal_values = softmax(legal_values)
        chosen_move = random.choices(legal_moves, weights=legal_values)[0]
        board.push(chosen_move)
        return chosen_move, board
        
    def expand(self, node):
        board = PxBoard(node.position)
        legal_moves = list(board.legal_moves)
        for move in board.legal_moves:
            board_copy = board.copy()
            board_copy.push(move)
            child = TreeNode(position=board_copy.fen(), move=move, parents=[node])
            if board_copy.is_game_over():
                child.gg = True
            self.node_dict[child.position] = child
            node.children.append(child)
        _, board_copy = self.select_random_weighted_move(node.position)
        return self.node_dict[board_copy.fen()]
        child = TreeNode(position=board_copy.fen(), move=chosen_move, parents=[node])
        return child

    def play_game(self, node):
        board = PxBoard(node.position)
        #print() 
        #print("SIMULATED GAME")
        move_count = 0
        while move_count < 10 and not board.is_game_over(claim_draw=True):
            chosen_move, _ = self.select_random_weighted_move(board.fen())
            board.push(chosen_move)
            move_count += 1
            #print(chosen_move.uci())
        if board.is_game_over(claim_draw=True):
            result = board.better_result()
            if result == self.color:
                result = 1
            elif result is None:
                result = 0.5
            else:
                result = 0
        else:
            result = self.evaluate_board(board) ** 2
            result = 1.0 / (1 + math.exp(-result))
        #print(len(board.move_stack))
        print("RESULT:\t" + str(result))
        #print()
        return result

    def __init__(self):
        super().__init__()
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 10000
        }
        self.piece_square_table = {
            chess.PAWN: [0,  0,  0,  0,  0,  0,  0,  0,
                         50, 50, 50, 50, 50, 50, 50, 50,
                         10, 10, 20, 30, 30, 20, 10, 10,
                         5,  5, 10, 27, 27, 10,  5,  5,
                         0,  0,  0, 25, 25,  0,  0,  0,
                         5, -5,-10,  0,  0,-10, -5,  5,
                         5, 10, 10,-25,-25, 10, 10,  5,
                         0,  0,  0,  0,  0,  0,  0,  0],
            chess.KNIGHT: [-50,-40,-30,-30,-30,-30,-40,-50,
                           -40,-20,  0,  0,  0,  0,-20,-40,
                           -30,  0, 10, 15, 15, 10,  0,-30,
                           -30,  5, 15, 20, 20, 15,  5,-30,
                           -30,  0, 15, 20, 20, 15,  0,-30,
                           -30,  5, 10, 15, 15, 10,  5,-30,
                           -40,-20,  0,  5,  5,  0,-20,-40,
                           -50,-40,-20,-30,-30,-20,-40,-50],
            chess.BISHOP: [-20,-10,-10,-10,-10,-10,-10,-20,
                           -10,  0,  0,  0,  0,  0,  0,-10,
                           -10,  0,  5, 10, 10,  5,  0,-10,
                           -10,  5,  5, 10, 10,  5,  5,-10,
                           -10,  0, 10, 10, 10, 10,  0,-10,
                           -10, 10, 10, 10, 10, 10, 10,-10,
                           -10,  5,  0,  0,  0,  0,  5,-10,
                           -20,-10,-40,-10,-10,-40,-10,-20],
            chess.KING: [-30, -40, -40, -50, -50, -40, -40, -30,
                         -30, -40, -40, -50, -50, -40, -40, -30,
                         -30, -40, -40, -50, -50, -40, -40, -30,
                         -30, -40, -40, -50, -50, -40, -40, -30,
                         -20, -30, -30, -40, -40, -30, -30, -20,
                         -10, -20, -20, -20, -20, -20, -20, -10, 
                         20,  20,   0,   0,   0,   0,  20,  20,
                         20,  30,  10,   0,   0,  10,  30,  20],
            chess.QUEEN: [0 for _ in range(64)],
            chess.ROOK: [0 for _ in range(64)]}

    def evaluate_board_v2(self, board=None):
        if board is None:
            board = self.board
        min_value = float("inf")
        for move in board.legal_moves:
            board_copy = board.copy()
            board_copy.push(move)
            value = self.evaluate_board(board_copy)
            if value < min_value:
                min_value = value
        return min_value

    def evaluate_board(self, board=None, printing=False):
        if board is None:
            board = self.board
        fen = board.fen()
        if fen in self.cache:
            self.cache_calls += 1
            return self.cache[fen]
        self.eval_calls += 1
        piece_map = board.piece_map()
        material_value = 0
        pure_material_value = 0
        square_adjustments = 0
        for square in piece_map:
            row = square // 8
            col = square % 8
            piece_type = piece_map[square].piece_type
            piece_value = self.piece_values[piece_type]
            if piece_map[square].color == self.color:
                material_value += piece_value
                pure_material_value += piece_value
                if self.color == chess.WHITE:
                    material_value += self.piece_square_table[piece_type][(7 - row) * 8 + col] / 100.0
                    square_adjustments += self.piece_square_table[piece_type][(7 - row) * 8 + col] / 100.0
                else:
                    material_value += self.piece_square_table[piece_type][square] / 100.0
                    square_adjustments += self.piece_square_table[piece_type][square] / 100.0
            else:
                material_value -= piece_value
                pure_material_value -= piece_value
                if self.color == chess.WHITE:
                    material_value -= self.piece_square_table[piece_type][square] / 100.0
                    square_adjustments -= self.piece_square_table[piece_type][square] / 100.0
                else:
                    material_value -= self.piece_square_table[piece_type][(7 - row) * 8 + col] / 100.0
                    square_adjustments -= self.piece_square_table[piece_type][(7 - row) * 8 + col] / 100.0

        if board.is_game_over(claim_draw=True):
            result = board.better_result()
            if result == self.color:
                material_value = 10000
            elif result is None:
                material_value = 0
            else:
                material_value = -10000

        self.cache[fen] = material_value
        #if printing:
        #    print("MY MATERIAL VALUE")
        #    print(material_value)
        #    print(pure_material_value)
        #    print(square_adjustments)
        return material_value #+ 0.1 * random.random()



class AbstractAlphaBetaAgent(Agent):

    def evaluate_board(self, board=None):
        raise NotImplementedError("Abstract agent class.")

    def alphabeta(self, depth=4, orig_depth=None, board=None, 
                  maximizingPlayer=True, alpha=float("-inf"), beta=float("inf")):
        if board is None:
           board = self.board
        if orig_depth is None:
            orig_depth = depth
        if depth == 0 or board.is_game_over(claim_draw=True):
            return self.evaluate_board(board), None
        #print("TURN\t" + str(board.turn))
        if maximizingPlayer:
            sorted_child_nodes = []
            for move in board.legal_moves:
                child_node = board.copy()
                child_node.push(move)
                sorted_child_nodes.append((child_node, move))
            sorted_child_nodes = sorted(
                sorted_child_nodes,
                key = lambda child: -self.cache[child[0].fen()] if child[0].fen() in self.cache else float("-inf"))
            #pprint(self.cache)
            #pprint([child[0].fen() for child in sorted_child_nodes])
            current_value = float("-inf")
            current_move = None
            move_value_map = dict()
            for child_node, move in sorted_child_nodes:
                child_value, _ = self.alphabeta(depth - 1, orig_depth, child_node, False, alpha, beta)
                child_value = round(child_value, 4)
                if depth == orig_depth:
                    move_value_map[move] = child_value
                #print("move = " + str(move.uci()) + "\t" + "cache = " + str(self.cache[child_node.fen()]) + "\tdepth = " + str(depth) + "\t" + "alpha = " + str(alpha) + "\t" + "beta = " + str(beta))
                if current_value < child_value:
                    current_value = child_value
                    current_move = move
                alpha = max(alpha, current_value)
                if alpha >= beta:
                    break
            if depth == orig_depth:
                pprint(move_value_map)
            '''
            best_moves = []
            for move in move_value_map:
                if move_value_map[move] == current_value:
                    best_moves.append(move)
            current_move = None
            print("VALUE: " + str(current_value))
            print(move_value_map)
            print(best_moves)
            #if depth == orig_depth:
                #current_move = random.choice(best_moves)
            current_move = best_moves[-1]
            '''
            return current_value, current_move
        else:
            sorted_child_nodes = []
            for move in board.legal_moves:
                child_node = board.copy()
                child_node.push(move)
                sorted_child_nodes.append((child_node, move))
            sorted_child_nodes = sorted(
                sorted_child_nodes,
                key = lambda child: -self.cache[child[0].fen()] if child[0].fen() in self.cache else float("-inf"))
            current_value = float("inf")
            current_move = None
            #print(depth)
            for child_node, move in sorted_child_nodes:
                child_value, _ = self.alphabeta(depth - 1, orig_depth, child_node, True, alpha, beta)
                child_value = round(child_value, 4)
                #print("move = " + str(move.uci()) + "\t" + "cache = " + str(self.cache[child_node.fen()]) + "\tdepth = " + str(depth) + "\t" + "beta = " + str(beta) + "\t" + "alpha = " + str(alpha))
                if current_value > child_value:
                    current_value = child_value
                    current_move = move
                beta = min(beta, current_value)
                '''
                if depth == 1 and board.uci(board.move_stack[-3]) == "f6d5":
                    print("VALUES")
                    print(board.move_stack[-3:])
                    print(move)
                    print("alpha:\t" + str(alpha))
                    print("beta\t" + str(beta))
                    print("current\t" + str(current_value))
                    print("child\t" + str(child_value))
                    print()
                #'''
                if beta <= alpha:
                    break
            return current_value, current_move

    def minimax(self, depth=4, orig_depth=None, board=None, maximizingPlayer=True):
        if board is None:
           board = self.board
        if orig_depth is None:
            orig_depth = depth
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board), None
        if maximizingPlayer:
            current_value = float("-inf")
            current_move = None
            move_value_map = dict()
            for move in board.legal_moves:
                child_node = board.copy()
                child_node.push(move)
                child_value, _ = self.minimax(depth - 1, orig_depth, child_node, False)
                child_value = round(child_value, 4)
                if depth == orig_depth:
                    move_value_map[move] = child_value
                if current_value < child_value:
                    current_value = child_value
                    current_move = move
            #if depth == orig_depth:
                #print(move_value_map)
            return current_value, current_move
        else:
            current_value = float("inf")
            current_move = None
            for move in board.legal_moves:
                child_node = board.copy()
                child_node.push(move)
                child_value, _ = self.minimax(depth - 1, orig_depth, child_node, True)
                child_value = round(child_value, 4)
                if current_value > child_value:
                    current_value = child_value
                    current_move = move
            return current_value, current_move

    def iterative_deepening(self, max_seconds=5.0, maximum_depth=2):
        self.cache_calls = 0
        self.eval_calls = 0
        current_depth = 0
        start_time = time.time()
        current_time = time.time()
        while current_depth < maximum_depth:# and current_time - start_time < max_seconds:
            current_depth += 1
            best_value, best_move = self.alphabeta(depth=current_depth)
            current_time = time.time()
        #print(current_depth)
        #print(current_time - start_time)
        #print(self.cache_calls)
        #print(self.eval_calls)
        return best_value, best_move

    def get_move(self):
        #best_value, best_move = self.alphabeta(depth=4)
        best_value, best_move = self.iterative_deepening(max_seconds=5.0, maximum_depth=4)
        print(str(best_move) + ":\t" + str(best_value))
        return best_move.uci()


class DumbGreedyCpuAgent(AbstractAlphaBetaAgent):
    def __init__(self):
        super().__init__()
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 10000
        }

        self.piece_square_table = {
            chess.PAWN: [0,  0,  0,  0,  0,  0,  0,  0,
                         50, 50, 50, 50, 50, 50, 50, 50,
                         10, 10, 20, 30, 30, 20, 10, 10,
                         5,  5, 10, 27, 27, 10,  5,  5,
                         0,  0,  0, 25, 25,  0,  0,  0,
                         5, -5,-10,  0,  0,-10, -5,  5,
                         5, 10, 10,-25,-25, 10, 10,  5,
                         0,  0,  0,  0,  0,  0,  0,  0],
            chess.KNIGHT: [-50,-40,-30,-30,-30,-30,-40,-50,
                           -40,-20,  0,  0,  0,  0,-20,-40,
                           -30,  0, 10, 15, 15, 10,  0,-30,
                           -30,  5, 15, 20, 20, 15,  5,-30,
                           -30,  0, 15, 20, 20, 15,  0,-30,
                           -30,  5, 10, 15, 15, 10,  5,-30,
                           -40,-20,  0,  5,  5,  0,-20,-40,
                           -50,-40,-20,-30,-30,-20,-40,-50],
            chess.BISHOP: [-20,-10,-10,-10,-10,-10,-10,-20,
                           -10,  0,  0,  0,  0,  0,  0,-10,
                           -10,  0,  5, 10, 10,  5,  0,-10,
                           -10,  5,  5, 10, 10,  5,  5,-10,
                           -10,  0, 10, 10, 10, 10,  0,-10,
                           -10, 10, 10, 10, 10, 10, 10,-10,
                           -10,  5,  0,  0,  0,  0,  5,-10,
                           -20,-10,-40,-10,-10,-40,-10,-20],
            chess.KING: [-30, -40, -40, -50, -50, -40, -40, -30,
                         -30, -40, -40, -50, -50, -40, -40, -30,
                         -30, -40, -40, -50, -50, -40, -40, -30,
                         -30, -40, -40, -50, -50, -40, -40, -30,
                         -20, -30, -30, -40, -40, -30, -30, -20,
                         -10, -20, -20, -20, -20, -20, -20, -10, 
                         20,  20,   0,   0,   0,   0,  20,  20,
                         20,  30,  10,   0,   0,  10,  30,  20],
            chess.QUEEN: [0 for _ in range(64)],
            chess.ROOK: [0 for _ in range(64)]}


    def evaluate_board(self, board=None, printing=False):
        if board is None:
            board = self.board
        fen = board.fen()
        if fen in self.cache:
            self.cache_calls += 1
            return self.cache[fen]
        self.eval_calls += 1
        piece_map = board.piece_map()
        material_value = 0
        pure_material_value = 0
        square_adjustments = 0
        for square in piece_map:
            row = square // 8
            col = square % 8
            piece_type = piece_map[square].piece_type
            piece_value = self.piece_values[piece_type]
            if piece_map[square].color == self.color:
                material_value += piece_value
                pure_material_value += piece_value
                if self.color == chess.WHITE:
                    material_value += self.piece_square_table[piece_type][(7 - row) * 8 + col] / 100.0
                    square_adjustments += self.piece_square_table[piece_type][(7 - row) * 8 + col] / 100.0
                else:
                    material_value += self.piece_square_table[piece_type][square] / 100.0
                    square_adjustments += self.piece_square_table[piece_type][square] / 100.0
            else:
                material_value -= piece_value
                pure_material_value -= piece_value
                if self.color == chess.WHITE:
                    material_value -= self.piece_square_table[piece_type][square] / 100.0
                    square_adjustments -= self.piece_square_table[piece_type][square] / 100.0
                else:
                    material_value -= self.piece_square_table[piece_type][(7 - row) * 8 + col] / 100.0
                    square_adjustments -= self.piece_square_table[piece_type][(7 - row) * 8 + col] / 100.0

        if board.is_game_over(claim_draw=True):
            result = board.better_result()
            if result == self.color:
                material_value = 10000
            elif result is None:
                material_value = 0
            else:
                material_value = -10000

        self.cache[fen] = material_value
        #if printing:
        #    print("MY MATERIAL VALUE")
        #    print(material_value)
        #    print(pure_material_value)
        #    print(square_adjustments)
        return material_value #+ 0.1 * random.random()



# start game with two players (human or cpu)
players = [HumanAgent(), DumbMctsAgent()]

# choose colors for each player
#random.shuffle(players)
print(players)

# core game loop
board = PxBoard()
players[0].start_game(board, chess.WHITE)
players[1].start_game(board, chess.BLACK)
board.print()

with open("moves_abtest.txt", "r") as f:
    move_list = []
    for line in f.readlines():
        move_list.append(line[:-1])


#with open("moves_abtest.txt", "w") as f:
for _ in range(1):
    gameboard = display.start(chess.STARTING_FEN)
    while not board.is_game_over(claim_draw=True):
        if board.turn == chess.WHITE:
            current_player = players[0]
        else:
            current_player = players[1]
        if len(move_list) > 10000:
            move_uci = move_list[0]
            del move_list[0]
        else:
            move_uci = current_player.get_move()
        board.push_uci(move_uci)

        #f.write(str(move_uci) + "\n")
        #f.flush()

        print(move_uci)
        board.print()
        display.update(board.fen(), gameboard)
        players[1].evaluate_board(printing=True)
        print()
    result = board.result()
    if result == "1-0":
        print("White wins.")
    elif result == "0-1":
        print("Black wins.")
    else:
        print("There are no winners in war.")


