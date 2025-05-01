import gradio as gr
import chess
import chess.svg
import torch
from minimax import MinimaxNNUE


# Initialize the bot with the model
bot = MinimaxNNUE(depth=3, path_file="./nnue_3_1978880d_512bs_200es_51e.pth")

# Function to display the board as SVG
def display_board(board):
    svg = chess.svg.board(board=board)
    return f'<div style="width:400px; height:400px;">{svg}</div>'

# Function to handle user move and bot response
def play_move(board_state, user_move):
    board = board_state.copy()  # Work with a copy to preserve state
    if board.is_game_over():
        return display_board(board), "Game over: " + board.result(), board
    
    # User's move (playing as White)
    try:
        move = chess.Move.from_uci(user_move)
        if move not in board.legal_moves:
            raise ValueError
        board.push(move)
    except ValueError:
        return display_board(board), "Invalid move, try again (e.g., 'e2e4').", board
    
    if board.is_game_over():
        return display_board(board), "Game over: " + board.result(), board
    
    # Bot's turn (playing as Black)
    bot_move = bot.find_best_move(board)
    if bot_move is None:
        return display_board(board), "Game over: " + board.result(), board
    board.push(bot_move)
    
    return display_board(board), f"Bot played {bot_move.uci()}", board

# Initialize the board
initial_board = chess.Board()

# Gradio interface
with gr.Blocks(title="Chess AI") as app:
    gr.Markdown("# Play Chess Against AI\nEnter your move in UCI notation (e.g., 'e2e4'). You play as White.")
    board_display = gr.HTML(display_board(initial_board))
    move_input = gr.Textbox(label="Your move")
    output_text = gr.Textbox(label="Status")
    state = gr.State(initial_board)
    
    def submit_move(state, user_move):
        board_svg, status, new_state = play_move(state, user_move)
        return board_svg, status, new_state
    
    move_input.submit(submit_move, inputs=[state, move_input], outputs=[board_display, output_text, state])

app.launch()