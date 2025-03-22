import pygame
import chess
from chess_ai import ChessAI  # Import the AI class from the other file

# Initialize Pygame
pygame.init()

# Window settings
WIDTH = 800
HEIGHT = 800
SQUARE_SIZE = WIDTH // 8
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess: User vs Bot")

# Colors
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (255, 0, 0)

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

# Font for pieces and text
font = pygame.font.SysFont("segoeuisymbol", 60)

# Initialize chess board and AI
board = chess.Board()
ai = ChessAI()

# Game variables
selected = None  # (file, rank) of selected piece

def draw_board():
    for rank in range(8):
        for file in range(8):
            color = LIGHT_SQUARE if (file + rank) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, (file * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces():
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            symbol = PIECE_SYMBOLS[(piece.piece_type, piece.color)]
            text = font.render(symbol, True, (0, 0, 0))
            screen.blit(text, (file * SQUARE_SIZE + 20, (7 - rank) * SQUARE_SIZE + 10))

def draw_highlight():
    if selected:
        file, rank = selected
        pygame.draw.circle(screen, HIGHLIGHT, 
                          (file * SQUARE_SIZE + SQUARE_SIZE // 2, (7 - rank) * SQUARE_SIZE + SQUARE_SIZE // 2), 
                          SQUARE_SIZE // 4, 3)

def draw_game_over():
    if board.is_game_over():
        result = board.result()
        text = font.render(f"Game Over: {result}", True, (255, 0, 0))
        screen.blit(text, (WIDTH // 4, HEIGHT // 2))

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and not board.is_game_over() and board.turn == chess.WHITE:
            # User's turn (White)
            mx, my = pygame.mouse.get_pos()
            file = mx // SQUARE_SIZE
            chess_rank = 7 - (my // SQUARE_SIZE)
            square = chess.square(file, chess_rank)
            
            if selected is None:  # Select piece
                piece = board.piece_at(square)
                if piece and piece.color == chess.WHITE:
                    selected = (file, chess_rank)
            else:  # Try to move
                from_square = chess.square(*selected)
                to_square = square
                move = chess.Move(from_square, to_square)
                
                # Handle pawn promotion (always to queen for simplicity)
                if (board.piece_at(from_square).piece_type == chess.PAWN and chess_rank == 7):
                    move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
                
                if move in board.legal_moves:
                    board.push(move)
                    selected = None
                else:
                    selected = None  # Deselect if illegal

    # Bot's turn (Black)
    if not board.is_game_over() and board.turn == chess.BLACK:
        bot_move = ai.find_best_move(board)
        if bot_move:
            board.push(bot_move)
            pygame.time.wait(500)  # Delay to visualize bot's move

    # Draw everything
    screen.fill((255, 255, 255))
    draw_board()
    draw_pieces()
    draw_highlight()
    draw_game_over()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()