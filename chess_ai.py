import chess
import chess.polyglot
import random

class ChessAI:
    def __init__(self, depth=3, book_path="Titans.bin"):
        self.depth = depth
        self.transposition_table = {}  # Initialize transposition table

        # Material values for evaluation (in centipawns)
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

        # Piece-Square Tables for white's perspective (in centipawns)
        self.psqt_white = {
            chess.PAWN: [
                0,  0,  0,  0,  0,  0,  0,  0,
                5, 10, 10,-20,-20,10,10, 5,
                5,-5,-10,  0,  0,-10,-5, 5,
                0,  0,  0, 20, 20,  0,  0,  0,
                5,  5, 10, 25, 25, 10,  5,  5,
                10, 10, 20, 30, 30, 20, 10, 10,
                50, 50, 50, 50, 50, 50, 50, 50,
                0,  0,  0,  0,  0,  0,  0,  0
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
        files = [[] for _ in range(8)] # List of pawn ranks per file
        for sq in pawns:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            files[file].append(rank)
        doubled = 0
        blocked = 0
        isolated = 0
        for file in range(8):
            if len(files[file]) > 1:
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
        """Alpha-Beta pruning with Transposition Table, returning (evaluation, best_move)."""
        # Compute Zobrist hash key for the current position
        key = chess.polyglot.zobrist_hash(board)
        
        # Check transposition table
        if key in self.transposition_table:
            entry = self.transposition_table[key]  # (depth, score, flag, best_move)
            if entry[0] >= depth:
                if entry[2] == 'exact':
                    return entry[1], entry[3]
                elif entry[2] == 'lower' and entry[1] >= beta:
                    return entry[1], entry[3]
                elif entry[2] == 'upper' and entry[1] <= alpha:
                    return entry[1], entry[3]
        
        # Null Move Pruning
        if depth >= 3 and not board.is_check():
            board.push(chess.Move.null())
            if maximizing_player:
                null_eval, _ = self.alphabeta(board, depth - 3, -beta, -beta + 1, False)
                null_eval = -null_eval  # Switch back to current player's perspective
                if null_eval >= beta:
                    board.pop()
                    return null_eval, None
            else:
                null_eval, _ = self.alphabeta(board, depth - 3, alpha - 1, alpha, True)
                null_eval = -null_eval  # Switch back to current player's perspective
                if null_eval <= alpha:
                    board.pop()
                    return null_eval, None
            board.pop()

        # Base case: depth 0 or game over
        if depth == 0 or board.is_game_over():
            score = self.evaluate_board(board)
            return score, None
        
        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for i, move in enumerate(board.legal_moves):
                board.push(move)
                if i == 0:  # Principal variation move
                    eval, _ = self.alphabeta(board, depth - 1, alpha, beta, False)
                else:
                    # Scout search with null window
                    eval, _ = self.alphabeta(board, depth - 1, alpha, alpha + 1, False)
                    if eval > alpha:
                        # Re-search with full window if scout search suggests a better move
                        eval, _ = self.alphabeta(board, depth - 1, alpha, beta, False)
                board.pop()

                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:  # Beta cutoff
                    break
            # Store in transposition table
            flag = 'lower' if max_eval >= beta else 'exact'
            self.transposition_table[key] = (depth, max_eval, flag, best_move)
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for i, move in enumerate(board.legal_moves):
                board.push(move)
                if i == 0:  # Principal variation move
                    eval, _ = self.alphabeta(board, depth - 1, alpha, beta, True)
                else:
                    # Scout search with null window
                    eval, _ = self.alphabeta(board, depth - 1, beta - 1, beta, True)
                    if eval < beta:
                        # Re-search with full window if scout search suggests a better move
                        eval, _ = self.alphabeta(board, depth - 1, alpha, beta, True)
                board.pop()

                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:  # Alpha cutoff
                    break
            # Store in transposition table
            flag = 'upper' if min_eval <= alpha else 'exact'
            self.transposition_table[key] = (depth, min_eval, flag, best_move)
            return min_eval, best_move
        
    def get_opening_move(self, board):
        """Get a move from the Polyglot opening book."""
        if self.book is None:
            return None
        try:
            entry = self.book.weighted_choice(board)
            return entry.move
        except IndexError:
            return None

    def find_best_move(self, board):
        """Find the best move by calling alphabeta directly, prioritizing the opening book."""
        if board.is_game_over():
            return None
        opening_move = self.get_opening_move(board)
        if opening_move is not None and opening_move in board.legal_moves:
            print('opening book')
            return opening_move
        
        # Call alphabeta with full depth and get the best move
        _, best_move = self.alphabeta(board, self.depth, float('-inf'), float('inf'), board.turn == chess.WHITE)
        return best_move