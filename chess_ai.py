import chess

class ChessAI:
    def __init__(self, depth=3):
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

    def evaluate_board(self, board):
        """Evaluate the board position based on material and piece-square tables."""
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Material value
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                    # Add PSQT value for white
                    if piece.piece_type in self.psqt_white:
                        score += self.psqt_white[piece.piece_type][square]
                else:
                    score -= value
                    # For black, use the flipped square for PSQT
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
                    break  # Beta cutoff
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
                    break  # Alpha cutoff
            return min_eval

    def find_best_move(self, board, depth=3):
        """Minimax alpha-beta pruning"""
        if board.is_game_over():
            return None
        
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