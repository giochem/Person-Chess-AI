import { Chess } from 'chess.js'
import { main_evaluation } from './evaluation_guide/evaluation.js'
import { fenToPosition } from './evaluation_guide/global.js'
export class Minimax {
  constructor(depth = 3) {
    this.depth = depth
  }
  evaluate(board) {
    const fen = board.fen()
    return main_evaluation(fenToPosition(fen))
  }
  minimax(board, depth, alpha, beta, maximizingPlayer) {
    if (depth === 0 || board.isGameOver()) {
      return { bestMove: null, val: this.evaluate(board) }
    }

    if (maximizingPlayer) {
      let bestMove = null
      let maxVal = -10000
      for (const move of board.moves()) {
        board.move(move)
        let { val } = this.minimax(board, depth - 1, alpha, beta, !maximizingPlayer)
        board.undo()
        if (val > maxVal) {
          maxVal = val
          bestMove = move
        }
        alpha = Math.max(alpha, val)
        if (beta <= alpha) {
          break
        }
      }
      return { bestMove: bestMove, val: maxVal }
    } else {
      let bestMove = null
      let minVal = 10000
      for (const move of board.moves()) {
        board.move(move)
        let { val } = this.minimax(board, depth - 1, alpha, beta, !maximizingPlayer)
        board.undo()
        if (val < minVal) {
          minVal = val
          bestMove = move
        }
        beta = Math.min(beta, val)
        if (beta <= alpha) {
          break
        }
      }
      return { bestMove: bestMove, val: minVal }
    }
  }
  findBestMove(board) {
    const turn = board.turn() === 'w' ? true : false
    const alpha = -10000
    const beta = 10000
    let { bestMove } = this.minimax(board, this.depth, alpha, beta, turn)
    return bestMove
  }
}
// const board = new Chess(
//   "r1bqkbnr/pppppppp/2n5/8/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 5 4"
// );
// const chess = new Minimax(2);
// console.time();
// console.log(chess.findBestMove(board));
// console.timeEnd();

// 12s -> 0.1s
// export default class {
//   Minimax
// }
