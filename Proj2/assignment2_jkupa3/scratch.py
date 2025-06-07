def potential_move_heuristic(game, move):
    new_game_board, _, _ = game.forecast_move(move)
    ap = new_game_board.get_active_player()
    return heuristic(new_game_board, ap)


    valid_moves.sort(key=lambda m: potential_move_heuristic(game, m))

        valid_moves.sort(key=lambda m: potential_move_heuristic(game, m), reverse=True)
