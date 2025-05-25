import PySimpleGUI as sg
import chess
from chess_ai.choose_ai import ChessAIManager 

# Constants
BOARD_SIZE = 480
SQUARE_SIZE = 60
PIECE_IMAGES = {
    (chess.WHITE, chess.PAWN): 'Images/60/wP.png',
    (chess.WHITE, chess.KNIGHT): 'Images/60/wN.png',
    (chess.WHITE, chess.BISHOP): 'Images/60/wB.png',
    (chess.WHITE, chess.ROOK): 'Images/60/wR.png',
    (chess.WHITE, chess.QUEEN): 'Images/60/wQ.png',
    (chess.WHITE, chess.KING): 'Images/60/wK.png',
    (chess.BLACK, chess.PAWN): 'Images/60/bP.png',
    (chess.BLACK, chess.KNIGHT): 'Images/60/bN.png',
    (chess.BLACK, chess.BISHOP): 'Images/60/bB.png',
    (chess.BLACK, chess.ROOK): 'Images/60/bR.png',
    (chess.BLACK, chess.QUEEN): 'Images/60/bQ.png',
    (chess.BLACK, chess.KING): 'Images/60/bK.png',
}

def draw_board(graph, board, selected_square=None):
    """Draw the chessboard and pieces on the Graph element."""
    graph.erase()
    # Draw squares with alternating colors
    for file in range(8):
        for rank in range(1, 9): # Ranks 1-8 for drawing logic
            color = 'white' if (file + rank) % 2 == 0 else 'gray'
            top_left = (file * SQUARE_SIZE, (8 - rank) * SQUARE_SIZE)
            bottom_right = ((file + 1) * SQUARE_SIZE, (9 - rank) * SQUARE_SIZE)
            graph.draw_rectangle(top_left, bottom_right, fill_color=color)
            
            square = chess.square(file, rank - 1) # Convert rank to 0-7 for chess.square
            piece = board.piece_at(square)
            if piece:
                image_file = PIECE_IMAGES[(piece.color, piece.piece_type)]
                graph.draw_image(filename=image_file, location=top_left)
    # Highlight selected square
    if selected_square is not None:
        file = chess.square_file(selected_square)
        rank = chess.square_rank(selected_square) + 1 # Convert 0-7 back to 1-8 for drawing
        top_left = (file * SQUARE_SIZE, (8 - rank) * SQUARE_SIZE)
        bottom_right = ((file + 1) * SQUARE_SIZE, (9 - rank) * SQUARE_SIZE)
        graph.draw_rectangle(top_left, bottom_right, line_color='red', line_width=3)

def update_move_history_display(window, current_board):
    """
    Updates the move history display with moves from the current board's stack.
    Formats moves into standard algebraic notation (e.g., "1. e4 e5\n2. Nf3 Nc6").
    """
    history_string = ""
    # Create a temporary board to correctly generate SAN for each historical move
    # as board.san() requires the move to be legal in the current board state.
    temp_board_for_san = chess.Board() 

    for i, move in enumerate(current_board.move_stack):
        if i % 2 == 0:  # White's move (0, 2, 4...)
            history_string += f"{i // 2 + 1}. " # Add move number
            history_string += temp_board_for_san.san(move) + " "
        else:  # Black's move (1, 3, 5...)
            history_string += temp_board_for_san.san(move) + "\n" # Add newline after Black's move
        temp_board_for_san.push(move) # Apply the move to the temporary board

    window['-MOVES_HISTORY-'].update(history_string.strip()) # Update the Multiline element

def main():
    """Main function to run the chess GUI."""
    # Initialize AI manager and get available AIs
    ai_manager = ChessAIManager()
    ai_list = ai_manager.get_ai_list()

    # Define GUI layout
    layout = [
        [sg.Graph(
            canvas_size=(BOARD_SIZE, BOARD_SIZE),
            graph_bottom_left=(0, BOARD_SIZE), 
            graph_top_right=(BOARD_SIZE, 0),   
            key='-GRAPH-',
            enable_events=True
        ), 
        sg.Column([ 
            [sg.Text('Move History:')],
            [sg.Multiline(size=(25, 20), key='-MOVES_HISTORY-', write_only=True, autoscroll=True, auto_refresh=True, font=('Helvetica', 10))]
        ], vertical_alignment='top')], 
        [sg.Text('Select AI:'), sg.Combo(ai_list, key='-AI-', default_value=ai_list[0] if ai_list else '', size=(20, 1)), 
         sg.Button('Start Game'), sg.Button('Copy FEN', key='-COPY_FEN_BUTTON-')],
        # NEW ROW FOR FEN INPUT AND LOAD BUTTON
        [sg.Text('Load FEN:'), sg.Input(key='-FEN_INPUT-', size=(50, 1)), sg.Button('Load FEN', key='-LOAD_FEN_BUTTON-')],
        [sg.Text('White to move', key='-STATUS-')],
        [sg.Text('Student: TRẦN XUÂN TRƯỜNG'), sg.Text("Teacher: NGUYỄN HOÀNG ĐIỆP"), sg.Text('Version: 1.0-beta')],
    ]

    # Create window
    window = sg.Window('Chess GUI', layout, finalize=True)
    graph = window['-GRAPH-']
    
    # Initialize board and display it (even before game starts)
    board = chess.Board() 
    draw_board(graph, board)
    update_move_history_display(window, board) # Display initial empty history

    # Game state variables
    board = None 
    selected_square = None
    ai = None
    game_active = False

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

        elif event == 'Start Game':
            ai_name = values['-AI-']
            if not ai_name:
                sg.popup_error('Please select an AI!')
                continue
            try:
                ai = ai_manager.get_ai(ai_name)
            except Exception as e:
                sg.popup_error(f'Error initializing AI: {e}')
                continue
            
            board = chess.Board() # Start a new game
            selected_square = None
            game_active = True
            draw_board(graph, board)
            update_move_history_display(window, board) # Clear history for new game
            window['-STATUS-'].update('White to move')

        elif event == '-COPY_FEN_BUTTON-':
            if board:
                fen_string = board.fen()
                sg.clipboard_set(fen_string)
                sg.popup_timed('FEN copied to clipboard!', auto_close_duration=1, no_titlebar=True)
            else:
                sg.popup_timed('No active game to copy FEN from!', auto_close_duration=1, no_titlebar=True)

        elif event == '-LOAD_FEN_BUTTON-': # NEW EVENT HANDLING FOR LOAD FEN BUTTON
            fen_string = values['-FEN_INPUT-'].strip()
            if not fen_string:
                sg.popup_timed('Please enter a FEN string to load!', auto_close_duration=1, no_titlebar=True)
                continue
            try:
                new_board = chess.Board(fen_string)
                board = new_board # Update the main board variable
                selected_square = None # Reset any current selection
                game_active = True # Game is now active
                draw_board(graph, board)
                update_move_history_display(window, board) # Update history (will be empty unless FEN includes moves)
                turn_text = 'White' if board.turn == chess.WHITE else 'Black'
                window['-STATUS-'].update(f'{turn_text} to move (Loaded from FEN)')
                window['-FEN_INPUT-'].update('') # Clear the input field
                
                # If it's AI's turn immediately after loading, trigger AI move
                if board.turn == chess.BLACK and ai and game_active:
                    window['-STATUS-'].update('Bot is thinking...')
                    window.refresh()
                    ai_move = ai.find_best_move(board)
                    if ai_move:
                        board.push(ai_move)
                        draw_board(graph, board)
                        update_move_history_display(window, board)
                        if board.is_game_over():
                            result = board.result()
                            window['-STATUS-'].update(f'Game over: {result}')
                            game_active = False
                        else:
                            window['-STATUS-'].update('White to move')
                    else:
                        sg.popup_error("AI could not find a legal move after FEN load.")
                        game_active = False

            except ValueError:
                sg.popup_error("Invalid FEN string provided! Please check the format.", title="FEN Error")
            except Exception as e:
                sg.popup_error(f"An unexpected error occurred while loading FEN: {e}", title="Error")

        elif event == '-GRAPH-' and game_active:
            # Convert PySimpleGUI graph coordinates to chess square coordinates
            x, y = values['-GRAPH-']
            file = x // SQUARE_SIZE # 0-7
            rank_from_top = y // SQUARE_SIZE # 0-7 (from top of board)
            rank = 7 - rank_from_top # Convert to 0-7 (from bottom of board, standard chess rank)
            
            # Ensure click is within board bounds (0-7 for file and rank)
            if 0 <= file < 8 and 0 <= rank < 8:
                square_clicked = chess.square(file, rank)

                if selected_square is None:
                    # Select a piece if it's White's turn and the square has a White piece
                    piece = board.piece_at(square_clicked)
                    if piece and piece.color == board.turn and board.turn == chess.WHITE:
                        selected_square = square_clicked
                        draw_board(graph, board, selected_square) # Highlight selected square
                else:
                    # Attempt to make a move from selected_square to square_clicked
                    move = chess.Move(selected_square, square_clicked)
                    
                    # Check for pawn promotion and add promotion=QUEEN by default
                    piece_on_selected = board.piece_at(selected_square)
                    if (piece_on_selected and piece_on_selected.piece_type == chess.PAWN and
                        ((piece_on_selected.color == chess.WHITE and chess.square_rank(square_clicked) == 7) or # White pawn to 8th rank
                         (piece_on_selected.color == chess.BLACK and chess.square_rank(square_clicked) == 0))): # Black pawn to 1st rank
                        
                        promotion_move = chess.Move(selected_square, square_clicked, promotion=chess.QUEEN)
                        if promotion_move in board.legal_moves:
                            move = promotion_move # Use the promotion move if it's legal

                    if move in board.legal_moves:
                        board.push(move)
                        draw_board(graph, board)
                        update_move_history_display(window, board) # Update history after human move
                        window['-STATUS-'].update('Black to move')

                        # Check if game is over after human's move
                        if board.is_game_over():
                            result = board.result()
                            window['-STATUS-'].update(f'Game over: {result}')
                            game_active = False
                        else:
                            # AI's turn (Black)
                            window['-STATUS-'].update('Bot is thinking...')
                            window.refresh() # Force GUI update to show "Bot is thinking..."
                            
                            ai_move = ai.find_best_move(board)
                            if ai_move: # Ensure AI returns a valid move
                                board.push(ai_move)
                                draw_board(graph, board)
                                update_move_history_display(window, board) # Update history after AI move
                                
                                if board.is_game_over():
                                    result = board.result()
                                    window['-STATUS-'].update(f'Game over: {result}')
                                    game_active = False
                                else:
                                    window['-STATUS-'].update('White to move')
                            else:
                                sg.popup_error("AI could not find a legal move or game already ended for AI.")
                                game_active = False 
                        selected_square = None # Deselect square after successful move
                        draw_board(graph, board, selected_square) # Redraw to remove highlight
                    else:
                        # If the attempted move (including promotion attempt) was not legal,
                        # just deselect the square without making a move.
                        selected_square = None
                        draw_board(graph, board, selected_square) # Redraw to remove highlight

    window.close()

if __name__ == '__main__':
    main()