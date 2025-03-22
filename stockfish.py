import chess
from chess_ai import ChessAI
from stockfish import Stockfish

# Elo calculation constants
K_FACTOR = 32  # Elo adjustment factor
STOCKFISH_ELO = 2800  # Assumed Elo for Stockfish at depth 18 (adjustable)

def calculate_elo(current_elo, opponent_elo, result):
    """Calculate new Elo rating based on match result."""
    expected_score = 1 / (1 + 10 ** ((opponent_elo - current_elo) / 400))
    actual_score = result  # 1 for win, 0 for loss, 0.5 for draw
    new_elo = current_elo + K_FACTOR * (actual_score - expected_score)
    return round(new_elo)

def play_match(ai, stockfish, match_num, current_elo):
    board = chess.Board()
    print(f"\n=== Match {match_num} (ChessAI Elo: {current_elo}) ===")
    
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = ai.find_best_move(board)
            if move:
                board.push(move)
                stockfish.set_fen_position(board.fen())
                # eval = stockfish.get_evaluation()
                # print(f"ChessAI (White) plays: {move}, Eval: {eval['value']} {'cp' if eval['type'] == 'cp' else 'mate'}")
        else:
            stockfish.set_fen_position(board.fen())
            stockfish_move = stockfish.get_best_move()
            move = chess.Move.from_uci(stockfish_move)
            board.push(move)
            stockfish.set_fen_position(board.fen())
            # eval = stockfish.get_evaluation()
            # print(f"Stockfish (Black) plays: {move}, Eval: {eval['value']} {'cp' if eval['type'] == 'cp' else 'mate'}")
    
    result = board.result()
    print(f"Result: {result}")
    
    # Convert result to score for Elo calculation
    if result == "1-0":
        score = 1
    elif result == "0-1":
        score = 0
    else:  # Draw (1/2-1/2)
        score = 0.5
    
    new_elo = calculate_elo(current_elo, STOCKFISH_ELO, score)
    print(f"New Elo after match: {new_elo}")
    return result, new_elo

def run_matches():
    ai = ChessAI(depth=3)
    stockfish = Stockfish(path="./stockfish/stockfish-windows-x86-64-avx2.exe", depth=18, parameters={"Threads": 2, "Hash": 2048})
    
    wins = 0
    losses = 0
    draws = 0
    current_elo = 100  # Starting Elo for ChessAI (arbitrary)
    
    for i in range(1, 2):
        result, current_elo = play_match(ai, stockfish, i, current_elo)
        if result == "1-0":
            wins += 1
        elif result == "0-1":
            losses += 1
        else:
            draws += 1
    
    print("\n=== Final Results ===")
    print(f"Total Wins (ChessAI): {wins}")
    print(f"Total Losses (ChessAI): {losses}")
    print(f"Total Draws: {draws}")
    print(f"Final Elo: {current_elo}")

if __name__ == "__main__":
    run_matches()