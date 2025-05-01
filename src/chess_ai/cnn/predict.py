from chess import Board, pgn
import chess
from .auxiliary_func import board_to_matrix
import torch
from .model import ChessModel
import pickle
import numpy as np
from pathlib import Path

def prepare_input(board: Board):
    matrix = board_to_matrix(board)
    x_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return x_tensor

with open(Path(__file__).parent / "move_to_int", "rb") as file:
    move_to_int = pickle.load(file)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Load the model
model = ChessModel(num_classes=len(move_to_int))
model.load_state_dict(torch.load(Path(__file__).parent / "TORCH_100EPOCHS.pth", map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode (it may be reductant)

int_to_move = {v: k for k, v in move_to_int.items()}
# Function to make predictions
def predict_move(board: Board):
    x_tensor = prepare_input(board).to(device)
    
    with torch.no_grad():
        logits = model(x_tensor)
    
    logits = logits.squeeze(0)  # Remove batch dimension
    
    probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(probabilities)[::-1]
    for move_index in sorted_indices:
        move = int_to_move[move_index]
        if move in legal_moves_uci:
            return move
    
    return None

class CNN:
    def find_best_move(self, board):
        x_tensor = prepare_input(board).to(device)
        with torch.no_grad():
            logits = model(x_tensor)
        logits = logits.squeeze(0)  # Remove batch dimension
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities
        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]
        sorted_indices = np.argsort(probabilities)[::-1]
        for move_index in sorted_indices:
            move = int_to_move[move_index]
            if move in legal_moves_uci:
                return chess.Move.from_uci(move)
        return None