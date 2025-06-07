#!/usr/bin/env python͏︍͏︆͏󠄁
import traceback
from isolation import Board, game_as_text
from test_players import RandomPlayer, HumanPlayer, Player
import platform

if platform.system() != 'Windows':
    import resource

from time import time, sleep

def correct_open_eval_fn(yourOpenEvalFn):
    print()
    try:
        sample_board = Board(RandomPlayer(), RandomPlayer())
        # setting up the board as though we've been playing͏︍͏︆͏󠄁
        board_state = [
            [" "," "," "," "," "," "," "],
            [" ","R1","X"," "," "," "," "],
            [" ","X"," "," "," "," "," "],
            [" "," ","X"," "," "," "," "],
            ["X"," ","R3"," "," "," ","X"],
            [" "," "," "," ","X","X","R2"],
            [" "," ","X"," ","R4"," "," "]
        ]
        sample_board.set_state(board_state, True)
        h = yourOpenEvalFn()
        print('OpenMoveEvalFn Test: This board has a score of %s.' % (h.score(sample_board, sample_board.get_active_player())))
    except NotImplementedError:
        print('OpenMoveEvalFn Test: Not implemented')
    except:
        print('OpenMoveEvalFn Test: ERROR OCCURRED')
        print(traceback.format_exc())

    print()

def beat_random(yourAgent):

    """Example test you can run
    to make sure your AI does better
    than random."""

    print("")
    try:
        r = RandomPlayer()
        p = yourAgent()
        game = Board(p, r, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation(time_limit=5000, print_moves=True)
        print("\n", winner, " has won. Reason: ", termination)
        # Uncomment to see game͏︍͏︆͏󠄁
        # print game_as_text(winner, move_history, termination, output_b)͏︍͏︆͏󠄁
    except NotImplementedError:
        print('CustomPlayer Test: Not Implemented')
    except:
        print('CustomPlayer Test: ERROR OCCURRED')
        print(traceback.format_exc())
    
    print()


def minimax_depth_test(yourAgent, minimax_fn):
    """Example test to make sure
    your minimax works, using the
    OpenMoveEvalFunction evaluation function.
    This can be used for debugging your code
    with different model Board states.
    Especially important to check alphabeta
    pruning"""

    # create dummy 9x9 board͏︍͏︆͏󠄁
    print("Now running Minimax depth test 1.")
    print()
    try:
        def time_left():  # For these testing purposes, let's ignore timeouts
            return 10000

        player = yourAgent() #using as a dummy player to create a board
        sample_board = Board(player, RandomPlayer())
        # setting up the board as though we've been playing͏︍͏︆͏󠄁
        board_state = [
            [" ","X"," "," ","X"," "," "],
            [" ","R1"," "," "," ","R2"," "],
            [" "," "," "," ","X","X"," "],
            [" ","X"," ","X"," "," ","X"],
            ["X"," ","X"," ","X","X"," "],
            ["R4"," ","X","X"," ","X","X"],
            [" ","X"," ","X","R3"," ","X"]
        ]
        sample_board.set_state(board_state, True)

        test_pass = True

        expected_depth_scores = [(1, 5), (2, 6), (3, 6), (4, 7), (5, 8)]
        
        for depth, exp_score in expected_depth_scores:
            move, score = minimax_fn(player, sample_board, time_left, depth=depth, my_turn=True)
            if exp_score != score:
                print("Minimax failed for depth: ", depth)
                test_pass = False
            else:
                print("Minimax passed for depth: ", depth)

        if test_pass:
            print()
            print("Now running Minimax depth test 2.")
            print()
            player = yourAgent()
            sample_board = Board(RandomPlayer(),player)
            # setting up the board as though we've been playing͏︍͏︆͏󠄁
            board_state = [
                ["X","X"," ","X","X"," ","R4"],
                [" "," "," "," ","R3","X","X"],
                [" "," ","X","X","X"," ","X"],
                [" ","X"," ","X"," ","X"," "],
                ["X"," ","X","X","X"," ","X"],
                ["X"," ","R1","X","X"," "," "],
                ["X"," "," "," ","R2"," "," "]
            ]
            sample_board.set_state(board_state, False)

            test_pass = True

            expected_depth_scores = [(1, -2), (2, -1), (3, 0), (4, -1), (5, 0)]

            for depth, exp_score in expected_depth_scores:
                move, score = minimax_fn(player, sample_board, time_left, depth=depth, my_turn=False)
          
                if exp_score != score:
                    print("Minimax failed for depth: ", depth)
                    test_pass = False
                else:
                    print("Minimax passed for depth: ", depth)

        if test_pass:
            print("Minimax Depth Test: Runs Successfully!")

        else:
            print("Minimax Depth Test: Failed")

    except NotImplementedError:
        print('Minimax Depth Test: Not implemented')
    except:
        print('Minimax Depth Test: ERROR OCCURRED')
        print(traceback.format_exc())

def test_minimax_move_selection(yourAgent, minimax_fn):
    """Test to verify that minimax selects the move with better mobility"""
    print("\nTesting Minimax Move Selection...")
    try:
        def time_left():
            return 10000

        player = yourAgent()
        sample_board = Board(player, RandomPlayer())
        

        board_state = [
            ["X","X","X","X","X","X","X"],
            ["X"," "," ","R1"," "," ","X"],
            ["X"," ","X","X","X"," ","X"],
            ["X"," ","X","R3"," "," "," "],
            ["X"," ","X","X","X"," "," "],
            ["X","X","R2","R4","X"," "," "],
            ["X","X","X","X","X","X","X"]
        ]
        sample_board.set_state(board_state, True)
        
        # Moving left to (1, 2) leads to more open spaces͏︍͏︆͏󠄁
        # Moving right (1, 4) or (1, 5) leads to being trapped by R3͏︍͏︆͏󠄁
        expected_move = ('CustomPlayer - R1', 1, 2)
        
        move, score = minimax_fn(player, sample_board, time_left, depth=3, my_turn=True)
        
        if move == expected_move:
            print("Minimax selected the move with better mobility!")
        else:
            print(f"Expected move: {expected_move}, Selected move: {move}")
            print("This move would lead to fewer options and being trapped by your opponent")
            
    except NotImplementedError:
        print('Minimax Move Selection Test: Not implemented')
    except Exception as e:
        print('Minimax Move Selection Test: ERROR OCCURRED')
        print(traceback.format_exc())
