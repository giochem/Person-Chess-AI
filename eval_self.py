import chess
from chess_ai import ChessAI  # Improved AI with PSQT
from chess_ai_original import ChessAIOriginal  # Original AI without PSQT

# Elo calculation constants
K_FACTOR = 32  # Elo adjustment factor

def calculate_elo(current_elo, opponent_elo, result):
    """Calculate new Elo rating based on match result."""
    expected_score = 1 / (1 + 10 ** ((opponent_elo - current_elo) / 400))
    actual_score = result  # 1 for win, 0 for loss, 0.5 for draw
    new_elo = current_elo + K_FACTOR * (actual_score - expected_score)
    return round(new_elo)

def play_match(original_ai, improved_ai, match_num, original_elo, improved_elo):
    board = chess.Board()
    print(f"\n=== Match {match_num} (Original Elo: {original_elo}, Improved Elo: {improved_elo}) ===")
    
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = original_ai.find_best_move(board)
            if move:
                board.push(move)
                print(f"Original AI (White) plays: {move}")
        else:
            move = improved_ai.find_best_move(board)
            if move:
                board.push(move)
                print(f"Improved AI (Black) plays: {move}")
    
    result = board.result()
    print(f"Result: {result}")
    
    # Convert result to score for Elo calculation
    if result == "1-0":
        score = 1  # Original AI wins
    elif result == "0-1":
        score = 0  # Improved AI wins
    else:  # Draw
        score = 0.5
    
    new_original_elo = calculate_elo(original_elo, improved_elo, score)
    new_improved_elo = calculate_elo(improved_elo, original_elo, 1 - score)  # Opponent's perspective
    print(f"New Original Elo: {new_original_elo}")
    print(f"New Improved Elo: {new_improved_elo}")
    return result, new_original_elo, new_improved_elo

def run_matches():
    original_ai = ChessAIOriginal(depth=3)
    improved_ai = ChessAI(depth=3)
    
    original_wins = 0
    improved_wins = 0
    draws = 0
    original_elo = 1500  # Starting Elo for original AI
    improved_elo = 1500  # Starting Elo for improved AI
    
    for i in range(1, 11):
        result, original_elo, improved_elo = play_match(original_ai, improved_ai, i, original_elo, improved_elo)
        if result == "1-0":
            original_wins += 1
        elif result == "0-1":
            improved_wins += 1
        else:
            draws += 1
    
    print("\n=== Final Results ===")
    print(f"Original AI Wins (White): {original_wins}")
    print(f"Improved AI Wins (Black): {improved_wins}")
    print(f"Draws: {draws}")
    print(f"Final Original Elo: {original_elo}")
    print(f"Final Improved Elo: {improved_elo}")
    print(f"Elo Difference (Improved - Original): {improved_elo - original_elo}")

if __name__ == "__main__":
    run_matches()