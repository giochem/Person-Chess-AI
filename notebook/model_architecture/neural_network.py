import torch.nn as nn
import torch

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        # flatten -> linear_tanh_stack -> output
        self.flatten = nn.Flatten()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(64 * (12 + 1), 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 8),
            nn.Tanh(),
        )
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_tanh_stack(x)
        x = self.output(x)
        return x
    
class NNUE_2(nn.Module):
    """Efficiently Updatable Neural Network for chess position evaluation.
    
    Args:
        num_features (int): Total number of possible features (768).
        hidden_size (int): Size of the embedding output (256).
        hidden_size2 (int): Size of the first hidden layer (32).
        hidden_size3 (int): Size of the second hidden layer (32).
    """
    def __init__(self, num_features=768, hidden_size=256, hidden_size2=32, hidden_size3=32):
        super(NNUE_2, self).__init__()
        self.embedding = nn.EmbeddingBag(num_features, hidden_size, mode='sum')
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, hidden_size3)
        self.fc3 = nn.Linear(hidden_size3, 1)
    
    def forward(self, features_white, offsets_white, features_black, offsets_black, side_to_move):
        """Forward pass to compute the evaluation score.
        
        Args:
            features_white (Tensor): Indices of white piece features.
            offsets_white (Tensor): Offsets for white features in the batch.
            features_black (Tensor): Indices of black piece features.
            offsets_black (Tensor): Offsets for black features in the batch.
            side_to_move (Tensor): 1 if white to move, 0 if black to move.
        
        Returns:
            Tensor: Predicted evaluation score for each position in the batch.
        """
        # Sum embeddings for white and black pieces
        white_sum = self.embedding(features_white, offsets_white)  # Shape: (batch_size, hidden_size)
        black_sum = self.embedding(features_black, offsets_black)  # Shape: (batch_size, hidden_size)
        
        # Assign "us" and "them" based on side to move
        side_to_move = side_to_move.bool()
        us_sum = torch.where(side_to_move[:, None], white_sum, black_sum)
        them_sum = torch.where(side_to_move[:, None], black_sum, white_sum)
        
        # Concatenate features and pass through fully connected layers
        input_vector = torch.cat([us_sum, them_sum], dim=1)  # Shape: (batch_size, 2 * hidden_size)
        x = torch.clamp(self.fc1(input_vector), 0, 1)  # Clipped ReLU: [0, 1]
        x = torch.clamp(self.fc2(x), 0, 1)
        output = self.fc3(x)  # Linear output
        return output