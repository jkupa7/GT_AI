# Board visualization with ipywidgets͏︍͏︆͏󠄁
import copy
from time import sleep
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets import VBox, HBox, Label, Button, GridspecLayout
from ipywidgets import Button, GridBox, Layout, ButtonStyle
from IPython.display import display, clear_output

from isolation import Board
from test_players import Player

import time
import platform
# import io͏︍͏︆͏󠄁
from io import StringIO

# import resource͏︍͏︆͏󠄁
if platform.system() != 'Windows':
    import resource


def get_details(name):
    if name == 'R1':
        color = 'Blue'
    elif name == 'R2':
        color = 'LightBlue'  # Muted red
    elif name == 'R3':
        color = 'Pink'
    elif name == 'R4':
        color = 'Red'
    elif name == 'r1':
        color = 'Blue'
        name = ' '
    elif name == 'r2':
        color = 'LightBlue'  # Yellowish orange
        name = ' '
    elif name == 'r3':
        color = 'Pink'
        name = ' '
    elif name == 'r4':
        color = 'Red'  # Yellowish orange
        name = ' '
    elif name == 'X':
        color = 'black'
    elif name == 'O':
        color = '#AA4499'  # Purple
        name = ' '
    else:
        color = 'LightGray'
    style = ButtonStyle(button_color=color)
    return name, style


def create_cell(button_name='', grid_loc=None, click_callback=None):
    layout = Layout(width='auto', height='auto')
    name, style = get_details(button_name)
    button = Button(description=name, layout=layout, style=style)
    button.x, button.y = grid_loc
    if click_callback: button.on_click(click_callback)
    return button


def get_viz_board_state(game, show_legal_moves, selected_rook=None):
    board_state = game.get_state()
    if show_legal_moves and selected_rook:
        legal_moves = game.get_active_moves()
        for move in legal_moves:
            if move[0] == selected_rook:
                if board_state[move[1]][move[2]][0] != 'R':
                    board_state[move[1]][move[2]] = game.get_rook_symbol(move[0]).lower()
    return board_state


def create_board_gridbox(game, show_legal_moves, click_callback=None, selected_rook=None):
    """Create the game board grid"""
    h, w = game.height, game.width
    board_state = get_viz_board_state(game, show_legal_moves, selected_rook)

    grid_layout = GridspecLayout(
        n_rows=h,
        n_columns=w,
        grid_gap='2px 2px',
        width='480px',
        height='480px',
        justify_content='center'
    )

    for r in range(h):
        for c in range(w):
            cell = create_cell(
                button_name=board_state[r][c],
                grid_loc=(r, c),
                click_callback=click_callback
            )
            grid_layout[r, c] = cell

    return grid_layout


class InteractiveGame():
    def __init__(self, opponent=None, show_legal_moves=False):
        self.game = Board(Player(), opponent if opponent else Player())
        self.width = self.game.width
        self.height = self.game.height
        self.show_legal_moves = show_legal_moves

        self.selected_rook = None
        self.game_is_over = False
        self.placed_rooks = set()

        self.turn_label = widgets.Label(value="Player 1's turn")
        self.message_label = widgets.Label(value="Select R1 or R2 to move")

        self.p1_rooks = [self.game.__rook_1__, self.game.__rook_2__]
        self.p2_rooks = [self.game.__rook_3__, self.game.__rook_4__]

        self.rook_buttons = {}

        for rook in self.p1_rooks:
            btn = widgets.Button(
                description=rook[-2:],
                layout=widgets.Layout(width='100px')
            )
            btn.rook = rook
            btn.on_click(self.select_rook)
            self.rook_buttons[rook] = btn

        for rook in self.p2_rooks:
            btn = widgets.Button(
                description=rook[-2:],
                layout=widgets.Layout(width='100px')
            )
            btn.rook = rook
            btn.on_click(self.select_rook)
            self.rook_buttons[rook] = btn

        self.p1_button_box = widgets.HBox([self.rook_buttons[rook] for rook in self.p1_rooks])
        self.p2_button_box = widgets.HBox([self.rook_buttons[rook] for rook in self.p2_rooks])

        self.gridb = create_board_gridbox(
            self.game,
            self.show_legal_moves,
            click_callback=self.handle_click,
            selected_rook=self.selected_rook
        )

        self.layout = widgets.VBox([
            self.turn_label,
            self.message_label,
            self.p1_button_box,
            self.p2_button_box,
            self.gridb
        ])

        self.update_ui_for_current_player()
        display(self.layout)

    def select_rook(self, btn):
        """Handle rook selection"""
        is_player1_turn = self.game.get_active_player() == self.game.__player_1__
        current_player_rooks = self.p1_rooks if is_player1_turn else self.p2_rooks

        if btn.rook not in current_player_rooks:
            self.message_label.value = f"Not your rook to move!"
            return

        if len(self.placed_rooks) < 4:
            player1_placed = self.placed_rooks & set(self.p1_rooks)
            player2_placed = self.placed_rooks & set(self.p2_rooks)

            if is_player1_turn:
                if len(player1_placed) == 1:
                    unused_rook = next(r for r in self.p1_rooks if r not in self.placed_rooks)
                    if btn.rook != unused_rook:
                        self.message_label.value = f"You must place {unused_rook[-2:]} now!"
                        return
            else:
                if len(player2_placed) == 1:
                    unused_rook = next(r for r in self.p2_rooks if r not in self.placed_rooks)
                    if btn.rook != unused_rook:
                        self.message_label.value = f"You must place {unused_rook[-2:]} now!"
                        return

        self.selected_rook = btn.rook

        for rook, button in self.rook_buttons.items():
            if rook == self.selected_rook:
                button.button_style = 'success'
            else:
                button.button_style = ''

        self.message_label.value = f"Selected {btn.description}. Click a position to move."
        self._update_display()

    def handle_click(self, b):
        """Handle board position clicks"""
        if self.game_is_over:
            self.message_label.value = 'The game is over!'
            return

        if not self.selected_rook:
            self.message_label.value = "Please select a rook first!"
            return

        legal_moves = self.game.get_active_moves()
        valid_move = None
        for move in legal_moves:
            if move[0] == self.selected_rook and move[1] == b.x and move[2] == b.y:
                valid_move = move
                break

        if not valid_move:
            self.message_label.value = f"Invalid move for {self.selected_rook}!"
            return

        self.placed_rooks.add(self.selected_rook)

        self.game_is_over, winner = self.game.__apply_move__(valid_move)

        self.selected_rook = None
        for btn in self.rook_buttons.values():
            btn.button_style = ''

        self._update_display()

        if self.game_is_over:
            winner_str = "Player 1" if winner == self.game.__player_1__.__class__.__name__ else "Player 2"
            self.turn_label.value = f"Game Over! Winner: {winner_str}"
            self.message_label.value = "Game is over!"
            return

        self.update_ui_for_current_player()

    def update_ui_for_current_player(self):
        """Update UI elements based on current player"""
        is_player1_turn = self.game.get_active_player() == self.game.__player_1__

        self.turn_label.value = "Player 1's turn" if is_player1_turn else "Player 2's turn"

        self.p1_button_box.layout.display = 'flex' if is_player1_turn else 'none'
        self.p2_button_box.layout.display = 'none' if is_player1_turn else 'flex'

        if len(self.placed_rooks) < 4:
            player1_placed = self.placed_rooks & set(self.p1_rooks)
            player2_placed = self.placed_rooks & set(self.p2_rooks)

            if is_player1_turn:
                if len(player1_placed) == 1:
                    unused_rook = next(r for r in self.p1_rooks if r not in self.placed_rooks)
                    self.message_label.value = f"Player 1 must place {unused_rook[-2:]}"
                else:
                    self.message_label.value = "Player 1 select R1 or R2"
            else:
                if len(player2_placed) == 1:
                    unused_rook = next(r for r in self.p2_rooks if r not in self.placed_rooks)
                    self.message_label.value = f"Player 2 must place {unused_rook[-2:]}"
                else:
                    self.message_label.value = "Player 2 select R3 or R4"
        else:
            self.message_label.value = "Player 1 select R1 or R2" if is_player1_turn else "Player 2 select R3 or R4"

    def _update_display(self):
        """Update the game board display"""
        board_vis_state = get_viz_board_state(self.game, self.show_legal_moves, self.selected_rook)
        for r in range(self.height):
            for c in range(self.width):
                new_name, new_style = get_details(board_vis_state[r][c])
                self.gridb[r, c].description = new_name
                self.gridb[r, c].style = new_style




class ReplayGame():
    """This class is used to replay games (only works in jupyter)"""

    def __init__(self, game, move_history, show_legal_moves=False):
        self.game = game
        self.width = self.game.width
        self.height = self.game.height
        self.move_history = move_history
        self.show_legal_moves = show_legal_moves
        self.board_history = []
        self.new_board = self.setup_new_board()
        self.gridb = create_board_gridbox(self.new_board, self.show_legal_moves)
        self.generate_board_state_history()
        self.visualized_state = None
        self.output_section = widgets.Output(layout={'border': '1px solid black'})

    def setup_new_board(self, ):
        return Board(player_1=self.game.__player_1__,
                     player_2=self.game.__player_2__,
                     width=self.width,
                     height=self.height)

    def update_board_gridbox(self, move_i):
        board_vis_state, board_state = self.board_history[move_i]
        self.visualized_state = board_state
        for r in range(self.height):
            for c in range(self.width):
                new_name, new_style = get_details(board_vis_state[r][c])
                self.gridb[r, c].description = new_name
                self.gridb[r, c].style = new_style

    def equal_board_states(self, state1, state2):
        for r in range(self.height):
            for c in range(self.width):
                if state1[r][c] != state2[r][c]:
                    return False
        return True

    def generate_board_state_history(self):
        for move_pair in self.move_history:
            for move in move_pair:
                self.new_board.__apply_move__(move)
                board_vis_state = get_viz_board_state(self.new_board, self.show_legal_moves)
                board_state = self.new_board.get_state()
                self.board_history.append((copy.deepcopy(board_vis_state), copy.deepcopy(board_state)))
        assert self.equal_board_states(self.game.get_state(), self.new_board.get_state()), \
            "End game state based of move history is not consistent with state of the 'game' object."

    def get_board_state(self, x):
        """You can use this state to with game.set_state() to replicate same Board instance."""
        self.output_section.clear_output()
        with self.output_section:
            display(self.visualized_state)

    def show_board(self):
        # Show slider for move selection͏︍͏︆͏󠄁
        input_move_i = widgets.IntText(layout=Layout(width='auto'))
        slider_move_i = widgets.IntSlider(description=r"\(move[i]\)",
                                          min=0,
                                          max=len(self.board_history) - 1,
                                          continuous_update=False,
                                          layout=Layout(width='auto')
                                          )
        mylink = widgets.link((input_move_i, 'value'), (slider_move_i, 'value'))
        slider = VBox([input_move_i, interactive(self.update_board_gridbox, move_i=slider_move_i)])

        get_state_button = Button(description='get board state')
        get_state_button.on_click(self.get_board_state)

        grid = GridspecLayout(4, 6)  # , width='auto')
        # Left side͏︍͏︆͏󠄁
        grid[:3, :-3] = self.gridb
        grid[3, :-3] = slider

        # Right side͏︍͏︆͏󠄁
        grid[:-1, -3:] = self.output_section
        grid[-1, -3:] = get_state_button
        display(grid)
