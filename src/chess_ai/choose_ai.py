import chess
import chess.engine
from .cnn.predict import CNN
from .nnue.minimax import MinimaxNNUE
from .knightvision.minimax_inter import KnightVision, AdvantageMinimax
from stockfish import Stockfish
from pathlib import Path
import os
class ChessAIManager:
    def __init__(self):
        self.available_ais = {}
        self._load_available_ais()

    def _load_available_ais(self):
        # Get absolute path to the model file
        model_path = Path(__file__).parent / "nnue" / "nnue_3_1978880d_512bs_200es_51e.pth"
        # Initialize different chess AIs
        self.available_ais = {
            "KnightVision-1s": KnightVision(time_limit=1.0),
            "KnightVision-10s": KnightVision(max_depth=4, time_limit=10.0),
            "AdvantageMinimax": AdvantageMinimax(),
            "CNN": CNN(),
            "MinimaxNNUE-3": MinimaxNNUE(depth=3, path_file=str(model_path)),
            "MinimaxNNUE-4": MinimaxNNUE(depth=4, path_file=str(model_path)),
            "Stockfish-Easy": StockfishAI(depth=2, difficulty="easy"),
            "Stockfish-Medium": StockfishAI(depth=5, difficulty="medium"),
            "Stockfish-Hard": StockfishAI(depth=10, difficulty="hard")
        }

    def get_ai_list(self):
        """Return list of available AI names"""
        return list(self.available_ais.keys())

    def get_ai(self, name):
        """Get AI by name"""
        return self.available_ais.get(name)

class StockfishAI:
    SKILL_LEVEL = "Skill Level"

    def __init__(self, depth=3, difficulty="medium"):
        # Get the project root directory
        stockfish_path = Path(__file__).parent / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
        # Configure parameters based on difficulty
        if difficulty == "easy":
            parameters = {self.SKILL_LEVEL: 5, "Threads": 2, "Hash": 1024}
            engine_depth = depth
        elif difficulty == "medium":
            parameters = {self.SKILL_LEVEL: 10, "Threads": 2, "Hash": 2048}
            engine_depth = depth
        else:  # hard
            parameters = {self.SKILL_LEVEL: 20, "Threads": 2, "Hash": 2048}
            engine_depth = depth
        if not stockfish_path.exists():
            raise FileNotFoundError(f"Stockfish engine not found at {stockfish_path}")

        self.engine = Stockfish(
            path=str(stockfish_path),
            depth=engine_depth,
            parameters=parameters
        )

    def find_best_move(self, board):
        self.engine.set_fen_position(board.fen())
        best_move_str = self.engine.get_best_move()
        print("move stockfish", best_move_str)
        # Convert string move to chess.Move object
        return chess.Move.from_uci(best_move_str)