// global  export function
export function board(pos, x, y) {
  if (x >= 0 && x <= 7 && y >= 0 && y <= 7) return pos.b[x][y];
  return "x";
}
export function colorflip(pos) {
  var board = new Array(8);
  for (var i = 0; i < 8; i++) board[i] = new Array(8);
  for (let x = 0; x < 8; x++)
    for (let y = 0; y < 8; y++) {
      board[x][y] = pos.b[x][7 - y];
      var color = board[x][y].toUpperCase() == board[x][y];
      board[x][y] = color
        ? board[x][y].toLowerCase()
        : board[x][y].toUpperCase();
    }
  return {
    b: board,
    c: [pos.c[2], pos.c[3], pos.c[0], pos.c[1]],
    e: pos.e == null ? null : [pos.e[0], 7 - pos.e[1]],
    w: !pos.w,
    m: [pos.m[0], pos.m[1]],
  };
}
export function sum(pos, func, param) {
  var sum = 0;
  for (var x = 0; x < 8; x++)
    for (var y = 0; y < 8; y++) sum += func(pos, { x: x, y: y }, param);
  return sum;
}

export function fenToPosition(fen) {
  const [position, side, castling, enpassant, halfmove, fullmove] =
    fen.split(" ");

  // Initialize the board as an 8x8 array (files a-h, ranks 1-8)
  let board = [[], [], [], [], [], [], [], []]; // One subarray per file

  // Parse position
  const ranks = position.split("/"); // process rank 1 to 8
  for (let rank = 0; rank < 8; rank++) {
    let file = 0;
    for (let char of ranks[rank]) {
      if (/\d/.test(char)) {
        // Number: add that many empty squares
        for (let i = 0; i < parseInt(char); i++) {
          board[file].push("-");
          file++;
        }
      } else {
        // Piece: add the piece
        board[file].push(char);
        file++;
      }
    }
  }

  // Castling rights
  const castlingRights = [
    castling.includes("K"), // White kingside
    castling.includes("Q"), // White queenside
    castling.includes("k"), // Black kingside
    castling.includes("q"), // Black queenside
  ];

  // En passant
  const enPassantSquare = enpassant === "-" ? null : enpassant;

  // Side to move
  const whiteToMove = side === "w";

  // Move counts
  const moveCounts = [parseInt(halfmove), parseInt(fullmove)];

  return {
    b: board,
    c: castlingRights,
    e: enPassantSquare,
    w: whiteToMove,
    m: moveCounts,
  };
}

// pos = {
//   // chessboard
//   b: [
//     ["r", "p", "-", "-", "-", "-", "P", "R"],
//     ["n", "p", "-", "-", "-", "-", "P", "N"],
//     ["b", "p", "-", "-", "-", "-", "P", "B"],
//     ["q", "p", "-", "-", "-", "-", "P", "Q"],
//     ["k", "p", "-", "-", "-", "-", "P", "K"],
//     ["b", "p", "-", "-", "-", "-", "P", "B"],
//     ["n", "p", "-", "-", "-", "-", "P", "N"],
//     ["r", "p", "-", "-", "-", "-", "P", "R"],
//   ],

//   // castling rights
//   c: [true, true, true, true],

//   // enpassant
//   e: null,

//   // side to move
//   w: true,

//   // move counts
//   m: [0, 1],
// };
