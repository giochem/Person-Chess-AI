import chess
import time
import math

class KnightVision:
    def __init__(self, max_depth=5, time_limit=2.0):
        self.max_depth = max_depth
        self.time_limit = time_limit  # Time limit per move in seconds
        # Material values: [None, Pawn, Knight, Bishop, Rook, Queen, King]
        self.material_values = [0, 100, 300, 300, 500, 900, 10000]

    def evaluate_board(self, board):
        """Evaluate the board position based on material and piece-square tables."""
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        # Material balance: current player - opponent
        pieces = board.piece_map()
        material = sum(self.material_values[piece.piece_type]
                      for piece in pieces.values() if piece.color == board.turn)
        material -= sum(self.material_values[piece.piece_type]
                       for piece in pieces.values() if piece.color != board.turn)

        # Mobility: logarithm of the number of legal moves
        legal_moves = list(board.legal_moves)
        mobility = math.log(len(legal_moves)) if legal_moves else 0

        return material + mobility

    def alphabeta(self, board, depth, alpha, beta, maximizing_player):
        """Minimax with Alpha-Beta pruning."""
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        if maximizing_player:
            max_eval = float('-inf')
            for move in self.order_moves(board.legal_moves, board):
                board.push(move)
                evaluation = self.alphabeta(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, evaluation)
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.order_moves(board.legal_moves, board):
                board.push(move)
                evaluation = self.alphabeta(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def _evaluate_moves_at_depth(self, board, depth, alpha, beta):
        """Evaluate all moves at a specific depth and return the best move and its value."""
        best_move = None
        best_value = float('-inf') if board.turn == chess.WHITE else float('inf')

        for move in board.legal_moves:
            board.push(move)
            value = self.alphabeta(board, depth - 1, alpha, beta, not board.turn)
            board.pop()

            if (board.turn == chess.WHITE) == (value > best_value):
                best_value = value
                best_move = move

        return best_move, best_value
    
    def order_moves(self, moves, board):
        """Order moves to improve alpha-beta pruning efficiency."""
        def move_score(move):
            # Score based on captured piece value
            captured = board.piece_at(move.to_square)
            capture_score = self.material_values[captured.piece_type] if captured else 0
            # Score based on promotion
            promotion_score = self.material_values[move.promotion] if move.promotion else 0
            return (capture_score + promotion_score)

        return sorted(moves, key=move_score, reverse=True)
    
    def find_best_move(self, board):
        """Find the best move using Iterative Deepening."""
        if board.is_game_over():
            return None

        start_time = time.time()
        best_move = None
        for current_depth in range(1, self.max_depth + 1):
            if time.time() - start_time > self.time_limit:
                break

            current_move, _ = self._evaluate_moves_at_depth(
                board,
                current_depth,
                float('-inf'),
                float('inf')
            )
            if current_move is not None:
                best_move = current_move

        return best_move