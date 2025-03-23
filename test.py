import pygame
import chess
from chess_ai import ChessAI  # Assuming ChessAI is in chess_ai.py

# Initialize Pygame
pygame.init()

# Window settings
WIDTH = 800
HEIGHT = 600
SQUARE_SIZE = HEIGHT // 8
BOARD_X = 150
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess AI Test Visualization")

# Colors
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (255, 0, 0)
TEST_BG = (200, 200, 200)
SELECTED_BG = (150, 150, 255)
BUTTON_BG = (180, 180, 180)
SELECTED_BUTTON_BG = (100, 100, 255)

# Chess piece Unicode symbols
PIECE_SYMBOLS = {
    (chess.PAWN, chess.WHITE): '♙',
    (chess.PAWN, chess.BLACK): '♟',
    (chess.KNIGHT, chess.WHITE): '♘',
    (chess.KNIGHT, chess.BLACK): '♞',
    (chess.BISHOP, chess.WHITE): '♗',
    (chess.BISHOP, chess.BLACK): '♝',
    (chess.ROOK, chess.WHITE): '♖',
    (chess.ROOK, chess.BLACK): '♜',
    (chess.QUEEN, chess.WHITE): '♕',
    (chess.QUEEN, chess.BLACK): '♛',
    (chess.KING, chess.WHITE): '♔',
    (chess.KING, chess.BLACK): '♚'
}

# Fonts
piece_font = pygame.font.SysFont("segoeuisymbol", 50)
text_font = pygame.font.SysFont("arial", 16)

# Initialize AI
ai = ChessAI(depth=1)

test_cases = [
    # Normal Cases
    {"name": "Opening Move", "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1"},
    {"name": "Pawn Trade", "fen": "rnbqkbnr/pppp1ppp/5n2/5p2/5P2/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 2"},
    {"name": "Knight Development", "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 2"},

    # Special Cases
    {"name": "Knight Fork", "fen": "rnbqkb1r/pppp1ppp/5n2/5p2/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 2"},
    {"name": "K+P Endgame", "fen": "8/8/8/5k2/5p2/5P2/5K2/8 w - - 0 1"},

    # Advantage Cases
    {"name": "Passed Pawn", "fen": "8/7k/8/5p2/5P2/5K2/8/8 w - - 0 1"},
    {"name": "Trapped King", "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 2"},

    # Pawn Structure Cases
    {"name": "Doubled Pawns", "fen": "7k/8/8/5pp1/5p1P/5P2/5K2/8 w - - 0 1"},
    {"name": "Blocked Pawns", "fen": "8/7k/8/5p2/5P2/5p2/5K2/8 w - - 0 1"},
    {"name": "Isolated Pawn", "fen": "8/7k/8/5p2/5P2/5K2/8/8 w - - 0 1"},
    {"name": "Combined Pawn Issues", "fen": "8/7k/8/4pp1p/5P1P/5K2/8/8 w - - 0 1"},

    # Checkmate Case
    {"name": "Checkmate", "fen": "rnbqkbnr/p1pppppp/8/8/1pB1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4"},
    {"name": "Trap", "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"},
]

# Precompute AI moves and boards (simplified without expected)
for test in test_cases:
    board = chess.Board(test["fen"])
    move = ai.find_best_move(board)
    test["before"] = board.copy()
    test["move"] = move
    board.push(move)
    test["after"] = board.copy()

# Drawing functions
def draw_board(board, x_offset, y_offset):
    for rank in range(8):
        for file in range(8):
            color = LIGHT_SQUARE if (file + rank) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, (x_offset + file * SQUARE_SIZE, y_offset + (7 - rank) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces(board, x_offset, y_offset):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            symbol = PIECE_SYMBOLS[(piece.piece_type, piece.color)]
            text = piece_font.render(symbol, True, (0, 0, 0))
            screen.blit(text, (x_offset + file * SQUARE_SIZE + 15, y_offset + (7 - rank) * SQUARE_SIZE + 5))

def draw_move_highlight(board, move, x_offset, y_offset):
    if move:
        from_file = chess.square_file(move.from_square)
        from_rank = chess.square_rank(move.from_square)
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        pygame.draw.circle(screen, HIGHLIGHT, 
                           (x_offset + from_file * SQUARE_SIZE + SQUARE_SIZE // 2, y_offset + (7 - from_rank) * SQUARE_SIZE + SQUARE_SIZE // 2), 
                           SQUARE_SIZE // 4, 3)
        pygame.draw.circle(screen, HIGHLIGHT, 
                           (x_offset + to_file * SQUARE_SIZE + SQUARE_SIZE // 2, y_offset + (7 - to_rank) * SQUARE_SIZE + SQUARE_SIZE // 2), 
                           SQUARE_SIZE // 4, 3)

def draw_test_list(selected_index):
    for i, test in enumerate(test_cases):
        bg_color = SELECTED_BG if i == selected_index else TEST_BG
        pygame.draw.rect(screen, bg_color, (0, i * 30, 150, 30))
        text = text_font.render(test["name"], True, (0, 0, 0))
        screen.blit(text, (5, i * 30 + 5))

def draw_buttons(selected_view):
    buttons = ["Before", "After"]
    for i, btn in enumerate(buttons):
        bg_color = SELECTED_BUTTON_BG if selected_view == btn else BUTTON_BG
        pygame.draw.rect(screen, bg_color, (BOARD_X + i * 80, HEIGHT - 40, 70, 30))
        text = text_font.render(btn, True, (0, 0, 0))
        screen.blit(text, (BOARD_X + i * 80 + 5, HEIGHT - 35))

def draw_test_view(test, view, x_offset):
    if view == "Before":
        board = test["before"]
        move = test["move"]
        title = "Before Move"
    else:  # After
        board = test["after"]
        move = None
        title = "After Move"

    draw_board(board, x_offset, 0)
    draw_pieces(board, x_offset, 0)
    if move:
        draw_move_highlight(board, move, x_offset, 0)
    title_text = text_font.render(title, True, (0, 0, 0))
    screen.blit(title_text, (x_offset + 40, 10))

    # Test details (simplified)
    move_text = text_font.render(f"AI Move: {test['move']}", True, (0, 0, 0))
    screen.blit(move_text, (x_offset + 40, HEIGHT - 60))

# Main loop
running = True
clock = pygame.time.Clock()
selected_test_index = 0
selected_view = "Before"

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            # Test list click
            if mx < 150:
                clicked_index = my // 30
                if 0 <= clicked_index < len(test_cases):
                    selected_test_index = clicked_index
            # Button click
            elif HEIGHT - 40 <= my <= HEIGHT - 10:
                if BOARD_X <= mx < BOARD_X + 80:
                    selected_view = "Before"
                elif BOARD_X + 80 <= mx < BOARD_X + 160:
                    selected_view = "After"

    # Draw everything
    screen.fill((255, 255, 255))
    draw_test_list(selected_test_index)
    draw_test_view(test_cases[selected_test_index], selected_view, BOARD_X)
    draw_buttons(selected_view)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()