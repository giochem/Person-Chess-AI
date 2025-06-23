import { Chess } from 'chess.js'
import { Minimax } from '../evaluation/minimax.js' // Adjust path if necessary

self.onmessage = (event) => {
  const { fen, depth } = event.data
  console.log('worker', fen, depth)
  if (!fen || depth === undefined) {
    self.postMessage({ error: 'Invalid data received by worker' })
    return
  }

  try {
    const board = new Chess(fen)
    const minimax = new Minimax(depth)
    const bestMove = minimax.findBestMove(board)
    self.postMessage({ move: bestMove })
  } catch (error) {
    self.postMessage({ error: error.message || 'Worker calculation error' })
  }
}
