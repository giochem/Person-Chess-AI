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

export function imbalance_total(pos, square) {
  let v = 0;
  v += imbalance(pos) - imbalance(colorflip(pos));
  v += bishop_pair(pos) - bishop_pair(colorflip(pos));
  return (v / 16) << 0;
}
export function imbalance(pos, square) {
  if (square == null) return sum(pos, imbalance);
  let qo = [
    [0],
    [40, 38],
    [32, 255, -62],
    [0, 104, 4, 0],
    [-26, -2, 47, 105, -208],
    [-189, 24, 117, 133, -134, -6],
  ];
  let qt = [
    [0],
    [36, 0],
    [9, 63, 0],
    [59, 65, 42, 0],
    [46, 39, 24, -24, 0],
    [97, 100, -42, 137, 268, 0],
  ];
  let j = "XPNBRQxpnbrq".indexOf(board(pos, square.x, square.y));
  if (j < 0 || j > 5) return 0;
  let bishop = [0, 0],
    v = 0;
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      let i = "XPNBRQxpnbrq".indexOf(board(pos, x, y));
      if (i < 0) continue;
      if (i == 9) bishop[0]++;
      if (i == 3) bishop[1]++;
      if (i % 6 > j) continue;
      if (i > 5) v += qt[j][i - 6];
      else v += qo[j][i];
    }
  }
  if (bishop[0] > 1) v += qt[j][0];
  if (bishop[1] > 1) v += qo[j][0];
  return v;
}
export function bishop_pair(pos, square) {
  if (bishop_count(pos) < 2) return 0;
  if (square == null) return 1438;
  return board(pos, square.x, square.y) == "B" ? 1 : 0;
}
