import { board, colorflip, sum, fenToPosition } from "./global.js";
import {
  rank,
  file,
  pawn_count,
  knight_count,
  bishop_count,
  rook_count,
  queen_count,
  opposite_bishops,
  king_distance,
  king_ring,
  piece_count,
  pawn_attacks_span,
} from "./helpers.js";
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
export function isolated(pos, square) {
  if (square == null) return sum(pos, isolated);
  if (board(pos, square.x, square.y) != "P") return 0;
  for (let y = 0; y < 8; y++) {
    if (board(pos, square.x - 1, y) == "P") return 0;
    if (board(pos, square.x + 1, y) == "P") return 0;
  }
  return 1;
}
export function opposed(pos, square) {
  if (square == null) return sum(pos, opposed);
  if (board(pos, square.x, square.y) != "P") return 0;
  for (let y = 0; y < square.y; y++) {
    if (board(pos, square.x, y) == "p") return 1;
  }
  return 0;
}
export function phalanx(pos, square) {
  if (square == null) return sum(pos, phalanx);
  if (board(pos, square.x, square.y) != "P") return 0;
  if (board(pos, square.x - 1, square.y) == "P") return 1;
  if (board(pos, square.x + 1, square.y) == "P") return 1;
  return 0;
}
export function supported(pos, square) {
  if (square == null) return sum(pos, supported);
  if (board(pos, square.x, square.y) != "P") return 0;
  return (
    (board(pos, square.x - 1, square.y + 1) == "P" ? 1 : 0) +
    (board(pos, square.x + 1, square.y + 1) == "P" ? 1 : 0)
  );
}
export function backward(pos, square) {
  if (square == null) return sum(pos, backward);
  if (board(pos, square.x, square.y) != "P") return 0;
  for (let y = square.y; y < 8; y++) {
    if (
      board(pos, square.x - 1, y) == "P" ||
      board(pos, square.x + 1, y) == "P"
    )
      return 0;
  }
  if (
    board(pos, square.x - 1, square.y - 2) == "p" ||
    board(pos, square.x + 1, square.y - 2) == "p" ||
    board(pos, square.x, square.y - 1) == "p"
  )
    return 1;
  return 0;
}
export function doubled(pos, square) {
  if (square == null) return sum(pos, doubled);
  if (board(pos, square.x, square.y) != "P") return 0;
  if (board(pos, square.x, square.y + 1) != "P") return 0;
  if (board(pos, square.x - 1, square.y + 1) == "P") return 0;
  if (board(pos, square.x + 1, square.y + 1) == "P") return 0;
  return 1;
}
export function connected(pos, square) {
  if (square == null) return sum(pos, connected);
  if (supported(pos, square) || phalanx(pos, square)) return 1;
  return 0;
}
export function connected_bonus(pos, square) {
  if (square == null) return sum(pos, connected_bonus);
  if (!connected(pos, square)) return 0;
  let seed = [0, 7, 8, 12, 29, 48, 86];
  let op = opposed(pos, square);
  let ph = phalanx(pos, square);
  let su = supported(pos, square);
  let bl = board(pos, square.x, square.y - 1) == "p" ? 1 : 0;
  let r = rank(pos, square);
  if (r < 2 || r > 7) return 0;
  return seed[r - 1] * (2 + ph - op) + 21 * su;
}
export function weak_unopposed_pawn(pos, square) {
  if (square == null) return sum(pos, weak_unopposed_pawn);
  if (opposed(pos, square)) return 0;
  let v = 0;
  if (isolated(pos, square)) v++;
  else if (backward(pos, square)) v++;
  return v;
}
export function weak_lever(pos, square) {
  if (square == null) return sum(pos, weak_lever);
  if (board(pos, square.x, square.y) != "P") return 0;
  if (board(pos, square.x - 1, square.y - 1) != "p") return 0;
  if (board(pos, square.x + 1, square.y - 1) != "p") return 0;
  if (board(pos, square.x - 1, square.y + 1) == "P") return 0;
  if (board(pos, square.x + 1, square.y + 1) == "P") return 0;
  return 1;
}
export function blocked(pos, square) {
  if (square == null) return sum(pos, blocked);
  if (board(pos, square.x, square.y) != "P") return 0;
  if (square.y != 2 && square.y != 3) return 0;
  if (board(pos, square.x, square.y - 1) != "p") return 0;
  return 4 - square.y;
}
export function doubled_isolated(pos, square) {
  if (square == null) return sum(pos, doubled_isolated);
  if (board(pos, square.x, square.y) != "P") return 0;
  if (isolated(pos, square)) {
    let obe = 0,
      eop = 0,
      ene = 0;
    for (let y = 0; y < 8; y++) {
      if (y > square.y && board(pos, square.x, y) == "P") obe++;
      if (y < square.y && board(pos, square.x, y) == "p") eop++;
      if (
        board(pos, square.x - 1, y) == "p" ||
        board(pos, square.x + 1, y) == "p"
      )
        ene++;
    }
    if (obe > 0 && ene == 0 && eop > 0) return 1;
  }
  return 0;
}
export function pawns_mg(pos, square) {
  if (square == null) return sum(pos, pawns_mg);
  let v = 0;
  if (doubled_isolated(pos, square)) v -= 11;
  else if (isolated(pos, square)) v -= 5;
  else if (backward(pos, square)) v -= 9;
  v -= doubled(pos, square) * 11;
  v += connected(pos, square) ? connected_bonus(pos, square) : 0;
  v -= 13 * weak_unopposed_pawn(pos, square);
  v += [0, -11, -3][blocked(pos, square)];
  return v;
}
export function pawns_eg(pos, square) {
  if (square == null) return sum(pos, pawns_eg);
  var v = 0;
  if (doubled_isolated(pos, square)) v -= 56;
  else if (isolated(pos, square)) v -= 15;
  else if (backward(pos, square)) v -= 24;
  v -= doubled(pos, square) * 56;
  v += connected(pos, square)
    ? ((connected_bonus(pos, square) * (rank(pos, square) - 3)) / 4) << 0
    : 0;
  v -= 27 * weak_unopposed_pawn(pos, square);
  v -= 56 * weak_lever(pos, square);
  v += [0, -4, 4][blocked(pos, square)];
  return v;
}
