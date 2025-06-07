from argparse import ArgumentError
from copy import deepcopy
import time
import platform
# import io͏︍͏︆͏󠄁
from io import StringIO

# import resource͏︍͏︆͏󠄁
if platform.system() != 'Windows':
    import resource

import sys
import os

sys.path[0] = os.getcwd()


class Board:
    BLANK = " "
    BLOCKED = "X"
    TRAIL = "O"
    NOT_MOVED = (-1, -1)

    def __init__(self, player_1, player_2, width=7, height=7):
        self.width = width
        self.height = height

        self.__player_1__ = player_1
        self.__player_2__ = player_2

        self.__rook_1__ = player_1.__class__.__name__ + " - R1"
        self.__rook_2__ = player_1.__class__.__name__ + " - R2"
        self.__rook_3__ = player_2.__class__.__name__ + " - R3"
        self.__rook_4__ = player_2.__class__.__name__ + " - R4"

        self.__rooks__ = [self.__rook_1__, self.__rook_2__, self.__rook_3__, self.__rook_4__]

        self.__rook_symbols__ = {self.__rook_1__: "R1", self.__rook_2__: "R2",
                                 self.__rook_3__: "R3", self.__rook_4__: "R4"}

        self.__board_state__ = [[Board.BLANK for i in range(0, width)] for j in range(0, height)]

        self.__last_rook_move__ = {self.__rook_1__: Board.NOT_MOVED, self.__rook_2__: Board.NOT_MOVED,
                                   self.__rook_3__: Board.NOT_MOVED, self.__rook_4__: Board.NOT_MOVED}

        self.__active_player__ = player_1
        self.__inactive_player__ = player_2
        self.__p1_rooks__ = [self.__rook_1__, self.__rook_2__]
        self.__active_players_rook__ = self.__p1_rooks__
        self.__p2_rooks__ = [self.__rook_3__, self.__rook_4__]
        self.__inactive_players_rook__ = self.__p2_rooks__

        self.move_count = 0

    def get_state(self):
        """
        Get physical board state
        Parameters:
            None
        Returns: 
            State of the board: list[char]
        """
        return deepcopy(self.__board_state__)

    def set_state(self, board_state, p1_turn=True):
        '''
        Function to immediately bring a board to a desired state. Useful for testing purposes; call board.play_isolation() afterwards to play
        Parameters:
            board_state: list[str], Desired state to set to board
            p1_turn: bool, Flag to determine which player is active
        Returns:
            None
        '''
        self.__board_state__ = board_state
        self.width = len(board_state[0])
        self.height = len(board_state)

        for rook in self.__rooks__:
            rook_symbol = self.__rook_symbols__[rook]
            last_move = [(row, column.index(rook_symbol)) for row, column in enumerate(board_state) if rook_symbol in column]
            if last_move:
                # set last move to the first found occurrence of the given Rook͏︍͏︆͏󠄁
                self.__last_rook_move__[rook] = last_move[0]
            else:
                self.__last_rook_move__[rook] = Board.NOT_MOVED

        if p1_turn:
            self.__active_player__ = self.__player_1__
            self.__active_players_rook__ = self.__p1_rooks__
            self.__inactive_player__ = self.__player_2__
            self.__inactive_players_rook__ = self.__p2_rooks__
        else:
            self.__active_player__ = self.__player_2__
            self.__active_players_rook__ = self.__p2_rooks__
            self.__inactive_player__ = self.__player_1__
            self.__inactive_players_rook__ = self.__p1_rooks__
        # Count X's to get move count + 2 for initial moves͏︍͏︆͏󠄁
        self.move_count = sum(row.count('R1') + row.count('R3') + row.count('R2') + row.count('R4') for row in board_state)

    #function to edit to introduce any variant - edited for castle isolation variant by Abirath Raju (2/08/2024)͏︍͏︆͏󠄁
    def __apply_move__(self, rook_move):
        '''
        Apply chosen move to a board state and check for game end
        Parameters:
            rook_move: (int, int), Desired move to apply. Takes the
            form of (rook, row, column). Move must be legal.
        Returns:
            result: (bool, str), Game Over flag, winner 
        '''
        # print("Applying move:: ", rook_move)͏︍͏︆͏󠄁
        rook, row, col = rook_move
        my_pos = self.__last_rook_move__[rook]
        #opponent_pos = self.__last_rook_move__[self.__inactive_players_rook__]͏︍͏︆͏󠄁

        ######Change the following lines to introduce any variant######͏︍͏︆͏󠄁
        if my_pos != Board.NOT_MOVED:
            self.__board_state__[my_pos[0]][my_pos[1]] = Board.BLOCKED

            #check if rook moves more than 1 space in any direction͏︍͏︆͏󠄁
            # if abs(col - my_pos[0]) > 1 or abs(row - my_pos[1]) > 1:͏︍͏︆͏󠄁
            #     self.__create_crater__(rook_move)͏︍͏︆͏󠄁
       ######Change above lines to introduce any variant######͏︍͏︆͏󠄁

        # apply move of active player͏︍͏︆͏󠄁
        self.__last_rook_move__[rook] = (row,col)
        self.__board_state__[row][col] = self.__rook_symbols__[rook]


        # rotate the players͏︍͏︆͏󠄁
        self.__active_player__, self.__inactive_player__ = self.__inactive_player__, self.__active_player__

        # rotate the rooks͏︍͏︆͏󠄁
        self.__active_players_rook__, self.__inactive_players_rook__ = self.__inactive_players_rook__, self.__active_players_rook__

        # increment move count͏︍͏︆͏󠄁
        self.move_count = self.move_count + 1

        # If opponent is isolated͏︍͏︆͏󠄁
        if not self.get_active_moves():
            return True, self.__inactive_player__.__class__.__name__

        return False, None

    def copy(self):
        '''
        Create a copy of this board and game state.
        Parameters:
            None
        Returns:
            Copy of self: Board class
        '''
        b = Board(self.__player_1__, self.__player_2__,
                  width=self.width, height=self.height)
        for key, value in self.__last_rook_move__.items():
            b.__last_rook_move__[key] = value
        for key, value in self.__rook_symbols__.items():
            b.__rook_symbols__[key] = value
            
        b.__board_state__ = self.get_state()
        b.__active_player__ = self.__active_player__
        b.__inactive_player__ = self.__inactive_player__
        b.__active_players_rook__ = self.__active_players_rook__
        b.__inactive_players_rook__ = self.__inactive_players_rook__
        b.move_count = self.move_count

        return b

    def forecast_move(self, rook_move):
        """
        See what board state would result from making a particular move without changing the board state itself.
        Parameters:
            rook_move: (rook, int, int), Desired move to forecast. Takes the form of
            (row, column).

        Returns:
            (Board, bool, str): Resultant board from move, flag for game-over, winner (if game is over)
        """
        new_board = self.copy()
        is_over, winner = new_board.__apply_move__(rook_move)
        return new_board, is_over, winner

    def get_active_player(self):
        """
        See which player is active. Used mostly in play_isolation for display purposes.
        Parameters:
            None
        Returns:
            object: The player object who's actively taking a turn
        """
        return self.__active_player__

    def get_inactive_player(self):
        """
        See which player is inactive. Used mostly in play_isolation for display purposes.
        Parameters:
            None
        Returns:
            object: The player object who's waiting for opponent to take a turn
        """
        return self.__inactive_player__

    def get_active_players_rooks(self):
        """
        See which rooks are active. Used mostly in play_isolation for display purposes.
        Parameters:
            None
        Returns:
            [str, str]: A list of rook names of the player who's actively taking a turn
        """
        return self.__active_players_rook__

    def get_inactive_players_rooks(self):
        """
        See which rook are inactive. Used mostly in play_isolation for display purposes.
        Parameters:
            None
        Returns:
            [str, str]: A list of rook names of the player who's waiting for opponent to take a turn
        """
        return self.__inactive_players_rook__

    def get_inactive_position(self):
        """
        Get position of inactive player (player waiting for opponent to make move) in [row, column] format
        Parameters:
            None
        Returns:
           [(int,int), (int,int)]: [[row, col], [row, col]] of inactive player
        """
        return [self.__last_rook_move__[rook][0:2] for rook in self.__inactive_players_rook__]

    def get_active_position(self):
        """
        Get position of active player (player actively making move) in [row, column] format
        Parameters:
            None
        Returns:
           [int, int]: [row, col] of active player
        """
        return [self.__last_rook_move__[rook][0:2] for rook in self.__active_players_rook__]

    def get_player_position(self, my_player=None):
        """
        Get position of certain player object. Should pass in yourself to get your position.
        Parameters:
            my_player (Player), Player to get position for
            If calling from within a player class, my_player = self can be passed.
        returns
            [int, int]: [row, col] position of player

        """
        if my_player == self.__player_1__ and self.__active_player__ == self.__player_1__:
            return self.get_active_position()
        if my_player == self.__player_1__ and self.__active_player__ != self.__player_1__:
            return self.get_inactive_position()
        elif my_player == self.__player_2__ and self.__active_player__ == self.__player_2__:
            return self.get_active_position()
        elif my_player == self.__player_2__ and self.__active_player__ != self.__player_2__:
            return self.get_inactive_position()
        else:
            raise ValueError("No value for my_player!")

    def get_opponent_position(self, my_player=None):
        """
        Get position of my_player's opponent.
        Parameters:
            my_player (Player), Player to get opponent's position
            If calling from within a player class, my_player = self can be passed.
        returns
            [int, int]: [row, col] position of my_player's opponent

        """
        if my_player == self.__player_1__ and self.__active_player__ == self.__player_1__:
            return self.get_inactive_position()
        if my_player == self.__player_1__ and self.__active_player__ != self.__player_1__:
            return self.get_active_position()
        elif my_player == self.__player_2__ and self.__active_player__ == self.__player_2__:
            return self.get_inactive_position()
        elif my_player == self.__player_2__ and self.__active_player__ != self.__player_2__:
            return self.get_active_position()
        else:
            raise ValueError("No value for my_player!")

    def get_inactive_moves(self):
        """
        Get all legal moves of inactive player on current board state as a list of possible moves.
        Parameters:
            None
        Returns:
           [(int, int)]: List of all legal moves. Each move takes the form of
            (rook, row, column).
        """
        q_move = [(rook,self.__last_rook_move__[rook][0], self.__last_rook_move__[rook][1]) for rook in self.__inactive_players_rook__]
        moves_all = []
        for rook in q_move:
            moves_all.extend(self.__get_moves__(rook))
        return moves_all

    def get_active_moves(self):
        """
        Get all legal moves of active player on current board state as a list of possible moves.
        Parameters:
            None
        Returns:
           [(int, int)]: List of all legal moves. Each move takes the form of
            (rook, row, column).
        """
        rooks_placed = sum(1 for rook in self.__rooks__ if self.__last_rook_move__[rook] != self.NOT_MOVED)

        if rooks_placed < 4:
            active_rooks_placed = sum(1 for rook in self.__active_players_rook__
                                      if self.__last_rook_move__[rook] != self.NOT_MOVED)

            if active_rooks_placed == 1:
                unused_rook = next(rook for rook in self.__active_players_rook__
                                   if self.__last_rook_move__[rook] == self.NOT_MOVED)
                return self.__get_moves__((unused_rook, -1, -1))

        moves_all = []
        for rook in self.__active_players_rook__:
            pos = self.__last_rook_move__[rook]
            moves_all.extend(self.__get_moves__((rook, pos[0], pos[1])))
        return moves_all

    def get_player_moves(self, my_player=None):
        """
        Get all legal moves of certain player object. Should pass in yourself to get your moves.
        Parameters:
            my_player (Player), Player to get moves for
            If calling from within a player class, my_player = self can be passed.
        returns
            [(int, int)]: List of all legal moves. Each move takes the form of
            (row, column).

        """
        if my_player == self.__player_1__ and self.__active_player__ == self.__player_1__:
            return self.get_active_moves()
        elif my_player == self.__player_1__ and self.__active_player__ != self.__player_1__:
            return self.get_inactive_moves()
        elif my_player == self.__player_2__ and self.__active_player__ == self.__player_2__:
            return self.get_active_moves()
        elif my_player == self.__player_2__ and self.__active_player__ != self.__player_2__:
            return self.get_inactive_moves()
        else:
            raise ValueError("No value for my_player!")

    def get_opponent_moves(self, my_player=None):
        """
        Get all legal moves of the opponent of the player provided. Should pass in yourself to get your opponent's moves.
        If calling from within a player class, my_player = self can be passed.
        Parameters:
            my_player (Player), The player facing the opponent in question
            If calling from within a player class, my_player = self can be passed.
        returns
            [(int, int)]: List of all opponent's moves. Each move takes the form of
            (row, column).

        """
        if my_player == self.__player_1__ and self.__active_player__ == self.__player_1__:
            return self.get_inactive_moves()
        if my_player == self.__player_1__ and self.__active_player__ != self.__player_1__:
            return self.get_active_moves()
        elif my_player == self.__player_2__ and self.__active_player__ == self.__player_2__:
            return self.get_inactive_moves()
        elif my_player == self.__player_2__ and self.__active_player__ != self.__player_2__:
            return self.get_active_moves()
        else:
            raise ValueError("No value for my_player!")

    def __get_moves__(self, move):
        """
        Get all legal moves of a player on current board state as a list of possible moves. Not meant to be directly called, 
        use get_active_moves or get_inactive_moves instead.
        Parameters:
            move: (rook, (int, int)), Last move made by player in question (which rook and where they currently are).
            Takes the form of (rook_id, (row, column)).
        Returns:
           [(str, int, int)]: List of all legal moves. Each move takes the form of
            (rook, row, column).
        """
        rook, r, c = move

        if (r,c) == self.NOT_MOVED:
            return self.get_first_moves(rook)

        directions = [(-1, 0), (0, -1), (0, 1),(1, 0)]

        moves = []

        for direction in directions:
            for dist in range(1, max(self.height, self.width)):
                row = direction[0] * dist + r
                col = direction[1] * dist + c
                # Allow for wrapping moves to other side of the board͏︍͏︆͏󠄁
                # if col < 0: col += self.height͏︍͏︆͏󠄁
                # if col >= self.height: col -= self.height͏︍͏︆͏󠄁
                # if row < 0: row += self.width͏︍͏︆͏󠄁
                # if row >= self.width: row -= self.width͏︍͏︆͏󠄁
                if self.space_is_open(row,col) and (row, col) not in moves:
                    moves.append((rook, row, col))
                else:
                    break

        return moves

    def get_first_moves(self, rook):
        """
        Return all moves for first turn in game (i.e. every board position)
        Parameters:
            rook: which rook to get moves for
        Returns:
           [(int, int)]: List of all legal moves. Each move takes the form of
            (rook, row, column).
        """
        return [(rook, i, j) for i in range(0, self.height)
                for j in range(0, self.width) if self.__board_state__[i][j] == Board.BLANK]

    def move_is_in_board(self, row, col):
        """
        Sanity check for making sure a move is within the bounds of the board.
        Parameters:
            col: int, Column position of move in question
            row: int, Row position of move in question
        Returns:
            bool: Whether the [row, col] values are within valid ranges
        """
        return 0 <= row < self.height and 0 <= col < self.width

    def is_spot_open(self, row, col):
        """
        Sanity check for making sure a move isn't occupied by an X.
        Parameters:
            col: int, Column position of move in question
            row: int, Row position of move in question
        Returns:
            bool: Whether the [row, col] position is blank (no X)
        """
        return self.__board_state__[row][col] == Board.BLANK

    def is_spot_rook(self, row, col):
        """
        Sanity check for checking if a spot is occupied by a player
        Parameters:
            col: int, Column position of move in question
            row: int, Row position of move in question
        Returns:
            bool: Whether the [row, col] position is currently occupied by a player's rook
        """
        return self.__board_state__[row][col] in self.__rook_symbols__.values()

    def space_is_open(self, row, col):
        """
        Sanity check to see if a space is within the bounds of the board and blank. Not meant to be called directly if you don't know what 
        you're looking for.
        Parameters:
            col: int, Col value of desired space
            row: int, Row value of desired space
        Returns:
            bool: (row, col ranges are valid) AND (space is blank)
        """
        return self.move_is_in_board(row,col) and self.is_spot_open(row,col)

    def print_board(self, legal_moves=[]):
        """
        Function for printing board state & indicating possible moves for active player.
        Parameters:
            legal_moves: [(int, int)], List of legal moves to indicate when printing board spaces. 
            Each move takes the form of (row, column), which is standard for Python arrays.
        Returns:
            Str: Visual interpretation of board state & possible moves for active player
        """

        p1_r, p1_c = self.__last_rook_move__[self.__rook_1__]
        p2_r, p2_c = self.__last_rook_move__[self.__rook_2__]
        p3_r, p3_c = self.__last_rook_move__[self.__rook_3__]
        p4_r, p4_c = self.__last_rook_move__[self.__rook_4__]
        b = self.__board_state__

        out = '  |'
        for i in range(len(b[0])):
            out += str(i) + ' |'
        out += '\n\r'

        for i in range(len(b)):
            out += str(i) + ' |'
            for j in range(len(b[i])):
                if (i, j) == (p1_r, p1_c):
                    out += self.__rook_symbols__[self.__rook_1__]
                elif (i, j) == (p2_r, p2_c):
                    out += self.__rook_symbols__[self.__rook_2__]
                elif(i, j) == (p3_r, p3_c):
                    out += self.__rook_symbols__[self.__rook_3__]
                elif(i, j) == (p4_r, p4_c):
                    out += self.__rook_symbols__[self.__rook_4__]
                elif (i, j) in legal_moves or (j, i) in legal_moves:
                    out += 'o '
                elif b[i][j] == Board.BLANK:
                    out += '  '
                elif b[i][j] == Board.TRAIL:
                   out += '- '
                elif b[i][j] == Board.BLOCKED:   #changed for skid variant
                    out += '><'
                out += '|'
            if i != len(b) - 1:
                out += '\n\r'

        return out

    def play_isolation(self, time_limit=10000, print_moves=False):
        """
        Method to play out a game of isolation with the agents passed into the Board class.
        Initializes and updates move_history variable, enforces timeouts, and prints the game.
        Parameters:
            time_limit: int, time limit in milliseconds that each player has before they time out.
            print_moves: bool, Should the method print details of the game in real time
        Returns:
            (str, [(int, int)], str): rook of Winner, Move history, Reason for game over.
            Each move in move history takes the form of (row, column).
        """
        move_history = []

        if platform.system() == 'Windows':
            def curr_time_millis():
                return int(round(time.time() * 1000))
        else:
            def curr_time_millis():
                return 1000 * resource.getrusage(resource.RUSAGE_SELF).ru_utime

        while True:
            game_copy = self.copy()
            move_start = curr_time_millis()

            def time_left():
                # print("Limit: "+str(time_limit) +" - "+str(curr_time_millis()-move_start))͏︍͏︆͏󠄁
                return time_limit - (curr_time_millis() - move_start)

            if print_moves:
                print("\n", self.__active_players_rook__, " Turn")

            curr_move = self.__active_player__.move(
                game_copy, time_left)  # rook added in return

            # Append new move to game history͏︍͏︆͏󠄁
            if self.__active_player__ == self.__player_1__:
                move_history.append([curr_move])
            else:
                move_history[-1].append(curr_move)

            # Handle Timeout͏︍͏︆͏󠄁
            if time_limit and time_left() <= 0:
                return self.__inactive_player__.__class__.__name__, move_history, \
                       (self.__active_player__.__class__.__name__ + " timed out.")

            # Safety Check͏︍͏︆͏󠄁
            legal_moves = self.get_active_moves()
            if curr_move not in legal_moves:
                return self.__inactive_player__.__class__.__name__, move_history, \
                       (self.__active_player__.__class__.__name__ + " made an invalid move.")

            # Apply move to game.͏︍͏︆͏󠄁
            is_over, winner = self.__apply_move__(curr_move)

            if print_moves:
                print("move chosen: ", curr_move)
                print(self.copy().print_board())

            if is_over:
                return self.__inactive_player__.__class__.__name__, move_history, \
                    (self.__active_player__.__class__.__name__ + " has no legal moves left.")
                # if not self.get_active_moves():͏︍͏︆͏󠄁
                #     return self.__active_players_rook__, move_history, \͏︍͏︆͏󠄁
                #            (self.__inactive_player__.__class__.__name__ + " has no legal moves left.")͏︍͏︆͏󠄁
                # return self.__active_player__.__class__.__name__, move_history, \͏︍͏︆͏󠄁
                #        (self.__inactive_player__.__class__.__name__ + " was forced off the grid.")͏︍͏︆͏󠄁

    def get_rook_symbol(self, rook):
        return self.__rook_symbols__[rook]

    def __apply_move_write__(self, move_rook):
        """
        Equivalent to __apply_move__, meant specifically for applying move history to a board 
        for analyzing an already played game.
        Parameters: 
            move_rook: (int, int), Move to apply to board. Takes
            the form of (row, column).
        Returns:
            None
        """

        if move_rook[0] is None or move_rook[1] is None:
            return

        rook, row, col = move_rook
        my_pos = self.__last_rook_move__[rook]

        self.__last_rook_move__[rook] = move_rook
        self.__board_state__[row][col] = \
            self.__rook_symbols__[rook]

        if self.move_is_in_board(my_pos[0], my_pos[1]):
            self.__board_state__[my_pos[0]][my_pos[1]] = Board.BLOCKED

        # Rotate the active player͏︍͏︆͏󠄁
        tmp = self.__active_player__
        self.__active_player__ = self.__inactive_player__
        self.__inactive_player__ = tmp

        # Rotate the active rook͏︍͏︆͏󠄁
        tmp = self.__active_players_rook__
        self.__active_players_rook__ = self.__inactive_players_rook__
        self.__inactive_players_rook__ = tmp

        self.move_count = self.move_count + 1


def game_as_text(winner, move_history, termination="", board=Board(1, 2)):
    """
    Function to play out a move history on a new board. Used for analyzing an interesting move history 
    Parameters: 
        move_history: [(rook, int, int)], History of all moves in order of game in question.
        Each move takes the form of (row, column).
        termination: str, Reason for game over of game in question. Obtained from play_isolation
        board: Board, board that game in question was played on. Used to initialize board copy
    Returns:
        Str: Print output of move_history being played out.
    """
    ans = StringIO()

    board = Board(board.__player_1__, board.__player_2__, board.width, board.height)

    print("Printing the game as text.")

    last_move = (9, 9, 0)

    for i, move in enumerate(move_history):
        if move is None or len(move) == 0:
            continue
        rook, row, col = move
        m = (row,col)
        if m != Board.NOT_MOVED:
            ans.write(board.print_board())
            board.__apply_move_write__(move)
            ans.write("\n\n" + rook + " moves to (" + str(row) + "," + str(col) + ")\r\n")

    ans.write("\n" + str(winner) + " has won. Reason: " + str(termination))
    return ans.getvalue()
