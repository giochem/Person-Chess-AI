import { board, colorflip, sum, fenToPosition } from "./global.js";
import {
  pinned,
  pinned_direction,
  pawn_attack,
  knight_attack,
  bishop_xray_attack,
  rook_xray_attack,
  queen_attack,
  queen_attack_diagonal,
  king_attack,
  attack,
} from "./attack.js";'
'
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
export function mobility(pos, square) {
  if (square == null) return sum(pos, mobility);
  var v = 0;
  var b = board(pos, square.x, square.y);
  if ("NBRQ".indexOf(b) < 0) return 0;
  for (var x = 0; x < 8; x++) {
    for (var y = 0; y < 8; y++) {
      var s2 = { x: x, y: y };
      if (!mobility_area(pos, s2)) continue;
      if (b == "N" && knight_attack(pos, s2, square) && board(pos, x, y) != "Q")
        v++;
      if (
        b == "B" &&
        bishop_xray_attack(pos, s2, square) &&
        board(pos, x, y) != "Q"
      )
        v++;
      if (b == "R" && rook_xray_attack(pos, s2, square)) v++;
      if (b == "Q" && queen_attack(pos, s2, square)) v++;
    }
  }
  return v;
}
export function mobility_area(pos, square) {
  if (square == null) return sum(pos, mobility_area);
  if (board(pos, square.x, square.y) == "K") return 0;
  if (board(pos, square.x, square.y) == "Q") return 0;
  if (board(pos, square.x - 1, square.y - 1) == "p") return 0;
  if (board(pos, square.x + 1, square.y - 1) == "p") return 0;
  if (
    board(pos, square.x, square.y) == "P" &&
    (rank(pos, square) < 4 || board(pos, square.x, square.y - 1) != "-")
  )
    return 0;
  if (blockers_for_king(colorflip(pos), { x: square.x, y: 7 - square.y }))
    return 0;
  return 1;
}
export function mobility_bonus(pos, square, mg) {
  if (square == null) return sum(pos, mobility_bonus, mg);
  let bonus = mg
    ? [
        [-62, -53, -12, -4, 3, 13, 22, 28, 33],
        [-48, -20, 16, 26, 38, 51, 55, 63, 63, 68, 81, 81, 91, 98],
        [-60, -20, 2, 3, 3, 11, 22, 31, 40, 40, 41, 48, 57, 57, 62],
        [
          -30, -12, -8, -9, 20, 23, 23, 35, 38, 53, 64, 65, 65, 66, 67, 67, 72,
          72, 77, 79, 93, 108, 108, 108, 110, 114, 114, 116,
        ],
      ]
    : [
        [-81, -56, -31, -16, 5, 11, 17, 20, 25],
        [-59, -23, -3, 13, 24, 42, 54, 57, 65, 73, 78, 86, 88, 97],
        [-78, -17, 23, 39, 70, 99, 103, 121, 134, 139, 158, 164, 168, 169, 172],
        [
          -48, -30, -7, 19, 40, 55, 59, 75, 78, 96, 96, 100, 121, 127, 131, 133,
          136, 141, 147, 150, 151, 168, 168, 171, 182, 182, 192, 219,
        ],
      ];
  let i = "NBRQ".indexOf(board(pos, square.x, square.y));
  if (i < 0) return 0;
  return bonus[i][mobility(pos, square)];
}
export function mobility_mg(pos, square) {
  if (square == null) return sum(pos, mobility_mg);
  return mobility_bonus(pos, square, true);
}
export function mobility_eg(pos, square) {
  if (square == null) return sum(pos, mobility_eg);
  return mobility_bonus(pos, square, false);
}
