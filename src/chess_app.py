import gradio as gr
import chess
import chess.svg
from chess_ai.choose_ai import ChessAIManager

# Initialize the AI manager
ai_manager = ChessAIManager()
GAME_OVER = "Game over: "
# Function to display the board as SVG
def display_board(board):
    svg = chess.svg.board(board=board)
    return f'<div style="width:400px; height:400px;">{svg}</div>'

# Function to handle user move and bot response
def play_move(board_state, user_move, ai_name):
    board = board_state.copy()  # Work with a copy to preserve state
    if board.is_game_over():
        return display_board(board), GAME_OVER + board.result(), board
    
    # User's move (playing as White)
    try:
        move = chess.Move.from_uci(user_move)
        if move not in board.legal_moves:
            raise ValueError
        board.push(move)
    except ValueError:
        return display_board(board), "Invalid move, try again (e.g., 'e2e4').", board
    
    if board.is_game_over():
        return display_board(board), GAME_OVER + board.result(), board
    
    # Bot's turn (playing as Black)
    ai = ai_manager.get_ai(ai_name)
    bot_move = ai.find_best_move(board)
    if bot_move is None:
        return display_board(board), GAME_OVER + board.result(), board
    board.push(bot_move)
    
    return display_board(board), f"{ai_name} played {bot_move.uci()}", board

# Initialize the board
initial_board = chess.Board()

# Gradio interface
with gr.Blocks(title="Chess AI") as app:
    gr.Markdown("# Play Chess Against AI\nEnter your move in UCI notation (e.g., 'e2e4'). You play as White.")
    
    # Add AI selection dropdown
    ai_selector = gr.Dropdown(
        choices=ai_manager.get_ai_list(),
        value=ai_manager.get_ai_list()[0],
        label="Select AI Opponent"
    )
    
    board_display = gr.HTML(display_board(initial_board))
    move_input = gr.Textbox(label="Your move")
    output_text = gr.Textbox(label="Status")
    state = gr.State(initial_board)
    
    def submit_move(state, user_move, ai_name):
        board = state.copy()
        if board.is_game_over():
            return display_board(board), GAME_OVER + board.result(), board
            
        # User's move
        try:
            move = chess.Move.from_uci(user_move)
            if move not in board.legal_moves:
                raise ValueError
            board.push(move)
        except ValueError:
            return display_board(board), "Invalid move, try again (e.g., 'e2e4').", board
        
        if board.is_game_over():
            return display_board(board), "Game over: " + board.result(), board
        
        # Get selected AI and make move
        ai = ai_manager.get_ai(ai_name)
        bot_move = ai.find_best_move(board)
        if bot_move is None:
            return display_board(board), GAME_OVER + board.result(), board
        board.push(bot_move)
        
        return display_board(board), f"{ai_name} played {bot_move.uci()}", board
    
    move_input.submit(
        submit_move,
        inputs=[state, move_input, ai_selector],
        outputs=[board_display, output_text, state]
    )

app.launch()