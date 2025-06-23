import { board, colorflip, sum, fenToPosition } from "./global.js";

import { imbalance, bishop_pair, imbalance_total } from "./imbalance.js";
import {
  pawnless_flank,
  strength_square,
  storm_square,
  shelter_strength,
  shelter_storm,
  king_pawn_distance,
  check,
  safe_check,
  king_attackers_count,
  king_attackers_weight,
  king_attacks,
  weak_bonus,
  weak_squares,
  unsafe_checks,
  knight_defender,
  endgame_shelter,
  blockers_for_king,
  flank_attack,
  flank_defense,
  king_danger,
  king_mg,
  king_eg,
} from "./king.js";
import {
  non_pawn_material,
  piece_value_bonus,
  psqt_bonus,
  piece_value_mg,
  piece_value_eg,
  psqt_mg,
  psqt_eg,
} from "./metarial.js";
import {
  mobility,
  mobility_area,
  mobility_bonus,
  mobility_mg,
  mobility_eg,
} from "./mobility.js";
import {
  candidate_passed,
  king_proximity,
  passed_block,
  passed_file,
  passed_rank,
  passed_leverable,
} from "./passed_pawns.js";
import {
  isolated,
  opposed,
  phalanx,
  supported,
  backward,
  doubled,
  connected,
  connected_bonus,
  weak_unopposed_pawn,
  weak_lever,
  blocked,
  doubled_isolated,
  pawns_mg,
  pawns_eg,
} from "./pawns.js";
import {
  outpost,
  outpost_square,
  reachable_outpost,
  minor_behind_pawn,
  bishop_pawns,
  rook_on_file,
  trapped_rook,
  weak_queen,
  king_protector,
  long_diagonal_bishop,
  outpost_total,
  rook_on_queen_file,
  bishop_xray_pawns,
  rook_on_king_ring,
  bishop_on_king_ring,
  queen_infiltration,
  pieces_mg,
  pieces_eg,
} from "./pieces.js";
import { space, space_area } from "./space.js";
import {
  safe_pawn,
  threat_safe_pawn,
  weak_enemies,
  minor_threat,
  rook_threat,
  hanging,
  king_threat,
  pawn_push_threat,
  slider_on_queen,
  knight_on_queen,
  restricted,
  weak_queen_protection,
  threats_mg,
  threats_eg,
} from "./threats.js";
import { winnable, winnable_total_mg, winnable_total_eg } from "./winnable.js";

export function rank(pos, square) {
  if (square == null) return sum(pos, rank);
  return 8 - square.y;
}
export function file(pos, square) {
  if (square == null) return sum(pos, file);
  return 1 + square.x;
}

export function piece_count(pos, square) {
  if (square == null) return sum(pos, piece_count);
  var i = "PNBRQK".indexOf(board(pos, square.x, square.y));
  return i >= 0 ? 1 : 0;
}
export function bishop_count(pos, square) {
  if (square == null) return sum(pos, bishop_count);
  if (board(pos, square.x, square.y) == "B") return 1;
  return 0;
}
export function queen_count(pos, square) {
  if (square == null) return sum(pos, queen_count);
  if (board(pos, square.x, square.y) == "Q") return 1;
  return 0;
}
export function pawn_count(pos, square) {
  if (square == null) return sum(pos, pawn_count);
  if (board(pos, square.x, square.y) == "P") return 1;
  return 0;
}
export function knight_count(pos, square) {
  if (square == null) return sum(pos, knight_count);
  if (board(pos, square.x, square.y) == "N") return 1;
  return 0;
}
export function rook_count(pos, square) {
  if (square == null) return sum(pos, rook_count);
  if (board(pos, square.x, square.y) == "R") return 1;
  return 0;
}

export function opposite_bishops(pos) {
  if (bishop_count(pos) != 1) return 0;
  if (bishop_count(colorflip(pos)) != 1) return 0;
  var color = [0, 0];
  for (var x = 0; x < 8; x++) {
    for (var y = 0; y < 8; y++) {
      if (board(pos, x, y) == "B") color[0] = (x + y) % 2;
      if (board(pos, x, y) == "b") color[1] = (x + y) % 2;
    }
  }
  return color[0] == color[1] ? 0 : 1;
}

export function king_distance(pos, square) {
  if (square == null) return sum(pos, king_distance);
  for (var x = 0; x < 8; x++) {
    for (var y = 0; y < 8; y++) {
      if (board(pos, x, y) == "K") {
        return Math.max(Math.abs(x - square.x), Math.abs(y - square.y));
      }
    }
  }
  return 0;
}
export function king_ring(pos, square, full) {
  if (square == null) return sum(pos, king_ring);
  if (
    !full &&
    board(pos, square.x + 1, square.y - 1) == "p" &&
    board(pos, square.x - 1, square.y - 1) == "p"
  )
    return 0;
  for (var ix = -2; ix <= 2; ix++) {
    for (var iy = -2; iy <= 2; iy++) {
      if (
        board(pos, square.x + ix, square.y + iy) == "k" &&
        ((ix >= -1 && ix <= 1) || square.x + ix == 0 || square.x + ix == 7) &&
        ((iy >= -1 && iy <= 1) || square.y + iy == 0 || square.y + iy == 7)
      )
        return 1;
    }
  }
  return 0;
}

export function pawn_attacks_span(pos, square) {
  if (square == null) return sum(pos, pawn_attacks_span);
  var pos2 = colorflip(pos);
  for (var y = 0; y < square.y; y++) {
    if (
      board(pos, square.x - 1, y) == "p" &&
      (y == square.y - 1 ||
        (board(pos, square.x - 1, y + 1) != "P" &&
          !backward(pos2, { x: square.x - 1, y: 7 - y })))
    )
      return 1;
    if (
      board(pos, square.x + 1, y) == "p" &&
      (y == square.y - 1 ||
        (board(pos, square.x + 1, y + 1) != "P" &&
          !backward(pos2, { x: square.x + 1, y: 7 - y })))
    )
      return 1;
  }
  return 0;
}
