import torch
import torch.nn as nn
import chess

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(768, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 1)

    def clipped_relu(self, x):
        return torch.clamp(x, 0, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.clipped_relu(x)
        x = self.fc2(x)
        x = self.clipped_relu(x)
        x = self.fc3(x)
        return x
    

def preprocess_fen(fen):
    board = chess.Board(fen)
    feature = torch.zeros(64, 6, 2)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            color = int(piece.color)
            piece_type = piece.piece_type - 1
            feature[square, piece_type, color] = 1

    # [64,6,2] -> [768] shape (batch size, 768 features)
    feature = feature.reshape(-1)
    return feature

class MinimaxNNUE:
    def __init__(self, depth=3, path_file='./checkpoint/nnue_512batchsize_101epochs'):
        self.depth = depth
        self.nnue_3 = NNUE()
        self.nnue_3.load_state_dict(torch.load(path_file, weights_only=True, map_location=torch.device(device))['model_state_dict'])
        self.nnue_3.eval()
        
    def evaluate_board(self, board):
        """Evaluate the board position based on material and piece-square tables."""
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        inp = preprocess_fen(board.fen())
        score = self.nnue_3(inp)
        return score

    def alphabeta(self, board, depth, alpha, beta, maximizing_player):
        """Minimax with Alpha-Beta pruning."""
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in board.legal_moves:
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
            for move in board.legal_moves:
                board.push(move)
                evaluation = self.alphabeta(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def find_best_move(self, board):
        """Find the best move for the current position."""
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
            
            if board.turn == chess.WHITE:  # Maximizing (White)
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
            else:  # Minimizing (Black)
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, value)
        
        return best_move