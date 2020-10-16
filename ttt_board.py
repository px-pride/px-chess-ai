import copy

class TttBoard:
    def __init__(self):
        self.X = True
        self.O = False
        self.three_in_a_rows = [
            ((0,0), (0,1), (0,2)),
            ((1,0), (1,1), (1,2)),
            ((2,0), (2,1), (2,2)),
            ((0,0), (1,0), (2,0)),
            ((0,1), (1,1), (2,1)),
            ((0,2), (1,2), (2,2)),
            ((0,0), (1,1), (2,2)),
            ((0,2), (1,1), (2,0))]
        self.reset()
    
    def __getitem__(self, key):
        return self.board[key]

    def reset(self):
        self.board = [['.' for _ in range(3)] for _ in range(3)]
        self.stack = []        

    def print(self):
        for i in range(3):
            print(self.board[2-i][0] + self.board[2-i][1] + self.board[2-i][2])
        print()

    def is_game_over(self, claim_draw=True):
        return self.result() is not None

    def fen(self):
        fen_string = ""
        for i in range(3):
            for j in range(3):
                fen_string += self.board[i][j]
        return fen_string

    def result(self):
        for threeset in self.three_in_a_rows:
            current_symbol = None
            break_flag = False
            for coords in threeset:
                symbol = self.board[coords[0]][coords[1]]
                if symbol == '.':
                    break_flag = True
                    break
                elif current_symbol is None:
                    current_symbol = symbol
                elif symbol != current_symbol:
                    break_flag = True
                    break
            if not break_flag:
                if current_symbol == 'X':
                    return '1-0'
                elif current_symbol == 'O':
                    return '0-1'
        if len(self.stack) == 9:
            return '1/2-1/2'
        else:
            return None

    def better_result(self):
        original_result = self.result(claim_draw=True)
        if original_result == "1-0":
            return self.X
        elif original_result == "0-1":
            return self.O
        else:
            return None

    def copy(self):
        board_copy = TttBoard()
        board_copy.board = copy.deepcopy(self.board)
        board_copy.stack = copy.deepcopy(self.stack)
        return board_copy

    def pop(self):
        popped = self.stack[-1]
        del self.stack[-1]
        return popped

    def peek(self):
        return self.stack[-1]

    def push(self, move):
        # check if move is legal
        # if so update board
        # and also update stack
        char2idx = {"a": 0, "b": 1, "c": 2}
        j = char2idx[move[0]]
        i = int(move[1]) - 1
        if not self.is_move_legal(i, j):
            raise ValueError("Illegal moves.")
        if self.turn == self.X:
            self.board[i][j] = 'X'
        else:
            self.board[i][j] = 'O'
        self.stack.append(move)

    def is_move_legal(self, i, j):
        return self.board[i][j] == "."

    def push_uci(self, move):
        self.push(move)

    #def piece_map(self):
        #raise NotImplementedError("finish coding this")

    @property
    def legal_moves(self):
        def move_gen():
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == ".":
                        yield self.move_from_coords(i, j)
        return move_gen()

    @property
    def turn(self):
        if len(self.stack) % 2 == 0:
            return self.X
        return self.O

    def move_from_coords(self, i, j):
        idx2char = {0: "a", 1: "b", 2: "c"}
        return idx2char[j] + str(i+1)
