import { main_evaluation } from "./evaluation.js";
import { fenToPosition } from "./global.js";
// Convert the given FEN
const fen = "rnbqkbnr/p1pppppp/P7/1p6/8/8/1PPPPPPP/RNBQKBNR w KQkq b6 0 2";
const pos = fenToPosition(fen);
console.log(main_evaluation(pos));
