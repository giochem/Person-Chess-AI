import chess
import chess.polyglot
import random
class ChessAI:
    def __init__(self, depth=3, book_path="Titans.bin"):
        self.depth = depth

        # Material values for evaluation (in centipawns)
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000  # High value for king, handled via checkmate
        }

        # Piece-Square Tables for white's perspective (in centipawns)
        self.psqt_white = {
            chess.PAWN: [
                0,  0,  0,  0,  0,  0,  0,  0,   # rank 1
                5, 10, 10,-20,-20,10,10, 5,   # rank 2
                5,-5,-10,  0,  0,-10,-5, 5,   # rank 3
                0,  0,  0, 20, 20,  0,  0,  0,   # rank 4
                5,  5, 10, 25, 25, 10,  5,  5,   # rank 5
                10, 10, 20, 30, 30, 20, 10, 10,   # rank 6
                50, 50, 50, 50, 50, 50, 50, 50,   # rank 7
                0,  0,  0,  0,  0,  0,  0,  0    # rank 8
            ],
            chess.KNIGHT: [
                -50,-40,-30,-30,-30,-30,-40,-50,
                -40,-20,  0,  0,  0,  0,-20,-40,
                -30,  0, 10, 15, 15, 10,  0,-30,
                -30,  5, 15, 20, 20, 15,  5,-30,
                -30,  0, 15, 20, 20, 15,  0,-30,
                -30,  5, 10, 15, 15, 10,  5,-30,
                -40,-20,  0,  5,  5,  0,-20,-40,
                -50,-40,-30,-30,-30,-30,-40,-50
            ],
            chess.BISHOP: [
                -20,-10,-10,-10,-10,-10,-10,-20,
                -10,  0,  0,  0,  0,  0,  0,-10,
                -10,  0,  5, 10, 10,  5,  0,-10,
                -10,  5,  5, 10, 10,  5,  5,-10,
                -10,  0, 10, 10, 10, 10,  0,-10,
                -10, 10, 10, 10, 10, 10, 10,-10,
                -10,  5,  0,  0,  0,  0,  5,-10,
                -20,-10,-10,-10,-10,-10,-10,-20
            ],
            chess.ROOK: [
                0,  0,  0,  0,  0,  0,  0,  0,
                5, 10, 10, 10, 10, 10, 10,  5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  5,  5,  0,  0, -5,
                0,  0,  0,  0,  0,  0,  0,  0
            ],
            chess.QUEEN: [
                -20,-10,-10, -5, -5,-10,-10,-20,
                -10,  0,  0,  0,  0,  0,  0,-10,
                -10,  0,  5,  5,  5,  5,  0,-10,
                -5,  0,  5,  5,  5,  5,  0, -5,
                0,  0,  5,  5,  5,  5,  0, -5,
                -10,  5,  5,  5,  5,  5,  0,-10,
                -10,  0,  5,  0,  0,  0,  0,-10,
                -20,-10,-10,-10,-10,-10,-10,-20
            ],
            chess.KING: [
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -20,-30,-30,-40,-40,-30,-30,-20,
                -10,-20,-20,-30,-30,-20,-20,-10,
                0,-10,-10,-20,-20,-10,-10, 0,
                20,-5,-5,-10,-10,-5,-5, 20
            ]
        }

        # Load Polyglot opening book
        try:
            self.book = chess.polyglot.open_reader(book_path)
        except FileNotFoundError:
            print(f"Warning: Opening book file '{book_path}' not found. AI will use alpha-beta only.")
            self.book = None

    def count_doubled_blocked_isolated(self, board, color):
        """Count doubled, blocked, and isolated pawns for a given color."""
        pawns = board.pieces(chess.PAWN, color)
        files = [[] for _ in range(8)]  # List of pawn ranks per file
        
        for sq in pawns:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            files[file].append(rank)
        doubled = 0
        blocked = 0
        isolated = 0
        
        for file in range(8):
            if len(files[file]) > 1:  # Doubled pawns
                doubled += len(files[file]) - 1
            for rank in files[file]:
                ahead_sq = chess.square(file, rank + 1 if color == chess.WHITE else rank - 1)
                if (0 <= ahead_sq < 64 and 
                    board.piece_at(ahead_sq) is not None and 
                    board.piece_at(ahead_sq).color != color):
                    blocked += 1
        
        for file in range(8):
            if files[file]:
                has_neighbors = (file > 0 and files[file - 1]) or (file < 7 and files[file + 1])
                if not has_neighbors:
                    isolated += len(files[file])
        return doubled, blocked, isolated
    
    def evaluate_board(self, board):
        """Evaluate the board position based on material and piece-square tables."""
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # white (+), black: (-)
        score = 0
        wd, ws, wi = self.count_doubled_blocked_isolated(board, chess.WHITE)
        bd, bs, bi = self.count_doubled_blocked_isolated(board, chess.BLACK)
        score -= 10 * wd
        score -= 10 * ws
        score -= 10 * wi
        score += 10 * bd
        score += 10 * bs
        score += 10 * bi

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += piece_value
                    if piece.piece_type in self.psqt_white:
                        score += self.psqt_white[piece.piece_type][square]
                else:
                    score -= piece_value
                    if piece.piece_type in self.psqt_white:
                        flipped_square = (7 - (square // 8)) * 8 + (7 - (square % 8))
                        score -= self.psqt_white[piece.piece_type][flipped_square]

        return score

    def alphabeta(self, board, depth, alpha, beta, maximizing_player):
        """Alpha-Beta pruning without unused parameters."""

        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        
        legal_moves = list(board.legal_moves)
        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval = self.alphabeta(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval = self.alphabeta(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_opening_move(self, board):
        """Get a move from the Polyglot opening book."""
        if self.book is None:
            return None
        try:
            # Find the main move in the book for the current position
            entry = self.book.weighted_choice(board)
            return entry.move
        except IndexError:
            # No move found in the book for this position
            return None

    def find_best_move(self, board):
        """Find the best move, prioritizing the opening book."""
        if board.is_game_over():
            return None
        
        # Check the opening book first
        opening_move = self.get_opening_move(board)
        if opening_move is not None and opening_move in board.legal_moves:
            print('opening book')
            return opening_move

        # Fall back to alpha-beta if no book move is found
        best_move = None
        best_value = float('-inf') if board.turn == chess.WHITE else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            value = self.alphabeta(board, self.depth - 1, alpha, beta, not board.turn)
            board.pop()
            
            if board.turn == chess.WHITE:  # Maximizing
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
            else:  # Minimizing
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, value)

        return best_move

# Example usage
# if __name__ == "__main__":
#     board = chess.Board()
#     ai = ChessAI(depth=3, book_path="book.bin")
#     move_count = 0
#     while not board.is_game_over() and move_count < 20:  # Limit to 20 half-moves (10 moves per side)
#         move = ai.find_best_move(board)
#         if move:
#             print(f"Move {move_count + 1}: {move}")
#             board.push(move)
#             print(board)
#             print()
#             move_count += 1
#         else:
#             print("Game over!")
#             break