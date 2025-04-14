import { board, colorflip, sum, fenToPosition } from "./global.js";
import { attack } from "./attack.js";
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
export function space_area(pos, square) {
  if (square == null) return sum(pos, space_area);
  let v = 0;
  let rank_val = rank(pos, square);
  let file_val = file(pos, square);
  if (
    rank_val >= 2 &&
    rank_val <= 4 &&
    file_val >= 3 &&
    file_val <= 6 &&
    board(pos, square.x, square.y) != "P" &&
    board(pos, square.x - 1, square.y - 1) != "p" &&
    board(pos, square.x + 1, square.y - 1) != "p"
  ) {
    v++;
    if (
      (board(pos, square.x, square.y - 1) == "P" ||
        board(pos, square.x, square.y - 2) == "P" ||
        board(pos, square.x, square.y - 3) == "P") &&
      !attack(colorflip(pos), { x: square.x, y: 7 - square.y })
    )
      v++;
  }
  return v;
}
export function space(pos, square) {
  if (non_pawn_material(pos) + non_pawn_material(colorflip(pos)) < 12222)
    return 0;
  var pieceCount = 0,
    blockedCount = 0;
  for (var x = 0; x < 8; x++) {
    for (var y = 0; y < 8; y++) {
      if ("PNBRQK".indexOf(board(pos, x, y)) >= 0) pieceCount++;
      if (
        board(pos, x, y) == "P" &&
        (board(pos, x, y - 1) == "p" ||
          (board(pos, x - 1, y - 2) == "p" && board(pos, x + 1, y - 2) == "p"))
      )
        blockedCount++;
      if (
        board(pos, x, y) == "p" &&
        (board(pos, x, y + 1) == "P" ||
          (board(pos, x - 1, y + 2) == "P" && board(pos, x + 1, y + 2) == "P"))
      )
        blockedCount++;
    }
  }
  var weight = pieceCount - 3 + Math.min(blockedCount, 9);
  return ((space_area(pos, square) * weight * weight) / 16) << 0;
}
