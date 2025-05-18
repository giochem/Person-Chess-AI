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
        for rank in range(1, 9):
            color = 'white' if (file + rank) % 2 == 0 else 'gray'
            top_left = (file * SQUARE_SIZE, (8 - rank) * SQUARE_SIZE)
            bottom_right = ((file + 1) * SQUARE_SIZE, (9 - rank) * SQUARE_SIZE)
            graph.draw_rectangle(top_left, bottom_right, fill_color=color)
            square = chess.square(file, rank - 1)
            piece = board.piece_at(square)
            if piece:
                image_file = PIECE_IMAGES[(piece.color, piece.piece_type)]
                graph.draw_image(filename=image_file, location=top_left)
    # Highlight selected square
    if selected_square is not None:
        file = chess.square_file(selected_square)
        rank = chess.square_rank(selected_square) + 1
        top_left = (file * SQUARE_SIZE, (8 - rank) * SQUARE_SIZE)
        bottom_right = ((file + 1) * SQUARE_SIZE, (9 - rank) * SQUARE_SIZE)
        graph.draw_rectangle(top_left, bottom_right, line_color='red', line_width=3)

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
        )],
        [sg.Text('Select AI:'), sg.Combo(ai_list, key='-AI-', default_value=ai_list[0]), sg.Button('Start Game')],
        [sg.Text('White to move', key='-STATUS-')],
        [sg.Text('Student: TRẦN XUÂN TRƯỜNG'), sg.Text("Teacher: NGUYỄN HOÀNG ĐIỆP"), sg.Text('Version: 1.0-beta')],
    ]

    # Create window
    window = sg.Window('Chess GUI', layout, finalize=True)
    graph = window['-GRAPH-']
    
    board = chess.Board()
    draw_board(graph, board)
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
            ai = ai_manager.get_ai(ai_name)
            board = chess.Board()
            selected_square = None
            game_active = True
            draw_board(graph, board)
            window['-STATUS-'].update('White to move')

        elif event == '-GRAPH-' and game_active:
            x, y = values['-GRAPH-']
            file = x // SQUARE_SIZE
            rank = 8 - (y // SQUARE_SIZE)
            if 0 <= file < 8 and 1 <= rank <= 8:
                square = chess.square(file, rank - 1)

                if selected_square is None:
                    # Select a piece if it's the player's turn and the square has their piece
                    piece = board.piece_at(square)
                    if piece and piece.color == board.turn and board.turn == chess.WHITE:
                        selected_square = square
                        draw_board(graph, board, selected_square)
                else:
                    # Attempt to make a move
                    move = chess.Move(selected_square, square)
                    # Handle pawn promotion (default to Queen)
                    if (board.piece_at(selected_square).piece_type == chess.PAWN and
                        chess.square_rank(square) in [0, 7]):
                        move.promotion = chess.QUEEN

                    if move in board.legal_moves:
                        board.push(move)
                        draw_board(graph, board)
                        window['-STATUS-'].update('Black to move')

                        # Check if game is over
                        if board.is_game_over():
                            result = board.result()
                            window['-STATUS-'].update(f'Game over: {result}')
                            game_active = False
                        else:
                            # AI's turn (Black)
                            window['-STATUS-'].update('Bot is thinking...')
                            window.refresh()
                            ai_move = ai.find_best_move(board)
                            board.push(ai_move)
                            draw_board(graph, board)
                            if board.is_game_over():
                                result = board.result()
                                window['-STATUS-'].update(f'Game over: {result}')
                                game_active = False
                            else:
                                window['-STATUS-'].update('White to move')
                    selected_square = None
                    draw_board(graph, board, selected_square)

    window.close()

if __name__ == '__main__':
    main()