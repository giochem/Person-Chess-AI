from flask import Flask, request, jsonify, send_from_directory
import chess
from chess_ai.choose_ai import ChessAIManager

app = Flask(__name__, static_folder='static')
ai_manager = ChessAIManager()
game_state = {'board': chess.Board(), 'ai': None}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/start', methods=['POST'])
def start_game():
    data = request.json
    ai_name = data.get('ai')
    game_state['ai'] = ai_manager.get_ai(ai_name)
    game_state['board'] = chess.Board()
    return jsonify({'status': 'Game started', 'fen': game_state['board'].fen()})

@app.route('/move', methods=['POST'])
def make_move():
    data = request.json
    move_uci = data.get('move')
    try:
        move = chess.Move.from_uci(move_uci)
        if not move in game_state['board'].legal_moves:
            return jsonify({'status': 'Illegal move - This move is not allowed'}), 400
            
        game_state['board'].push(move)
        
        if game_state['board'].is_game_over():
            result = game_state['board'].result()
            message = 'Draw' if result == '1/2-1/2' else ('White wins!' if result == '1-0' else 'Black wins!')
            return jsonify({'status': 'Game over', 'result': message, 'fen': game_state['board'].fen()})
        
        ai_move = game_state['ai'].find_best_move(game_state['board'])
        game_state['board'].push(ai_move)
        
        if game_state['board'].is_game_over():
            result = game_state['board'].result()
            message = 'Draw' if result == '1/2-1/2' else ('White wins!' if result == '1-0' else 'Black wins!')
            return jsonify({
                'status': 'Game over',
                'result': message,
                'fen': game_state['board'].fen()
            })
            
        return jsonify({
            'status': 'Move accepted',
            'ai_move': ai_move.uci(),
            'fen': game_state['board'].fen()
        })
            
    except ValueError:
        return jsonify({'status': 'Invalid move format'}), 400
    except Exception as e:
        return jsonify({'status': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)