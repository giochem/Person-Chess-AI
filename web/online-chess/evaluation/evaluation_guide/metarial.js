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
export function non_pawn_material(pos, square) {
  if (square == null) return sum(pos, non_pawn_material);
  let i = "NBRQ".indexOf(board(pos, square.x, square.y));
  if (i >= 0) return piece_value_bonus(pos, square, true);
  return 0;
}
export function piece_value_bonus(pos, square, mg) {
  if (square == null) return sum(pos, piece_value_bonus);
  let a = mg ? [124, 781, 825, 1276, 2538] : [206, 854, 915, 1380, 2682];
  let i = "PNBRQ".indexOf(board(pos, square.x, square.y));
  if (i >= 0) return a[i];
  return 0;
}
export function psqt_bonus(pos, square, mg) {
  if (square == null) return sum(pos, psqt_bonus, mg);
  var bonus = mg
    ? [
        [
          [-175, -92, -74, -73],
          [-77, -41, -27, -15],
          [-61, -17, 6, 12],
          [-35, 8, 40, 49],
          [-34, 13, 44, 51],
          [-9, 22, 58, 53],
          [-67, -27, 4, 37],
          [-201, -83, -56, -26],
        ],
        [
          [-53, -5, -8, -23],
          [-15, 8, 19, 4],
          [-7, 21, -5, 17],
          [-5, 11, 25, 39],
          [-12, 29, 22, 31],
          [-16, 6, 1, 11],
          [-17, -14, 5, 0],
          [-48, 1, -14, -23],
        ],
        [
          [-31, -20, -14, -5],
          [-21, -13, -8, 6],
          [-25, -11, -1, 3],
          [-13, -5, -4, -6],
          [-27, -15, -4, 3],
          [-22, -2, 6, 12],
          [-2, 12, 16, 18],
          [-17, -19, -1, 9],
        ],
        [
          [3, -5, -5, 4],
          [-3, 5, 8, 12],
          [-3, 6, 13, 7],
          [4, 5, 9, 8],
          [0, 14, 12, 5],
          [-4, 10, 6, 8],
          [-5, 6, 10, 8],
          [-2, -2, 1, -2],
        ],
        [
          [271, 327, 271, 198],
          [278, 303, 234, 179],
          [195, 258, 169, 120],
          [164, 190, 138, 98],
          [154, 179, 105, 70],
          [123, 145, 81, 31],
          [88, 120, 65, 33],
          [59, 89, 45, -1],
        ],
      ]
    : [
        [
          [-96, -65, -49, -21],
          [-67, -54, -18, 8],
          [-40, -27, -8, 29],
          [-35, -2, 13, 28],
          [-45, -16, 9, 39],
          [-51, -44, -16, 17],
          [-69, -50, -51, 12],
          [-100, -88, -56, -17],
        ],
        [
          [-57, -30, -37, -12],
          [-37, -13, -17, 1],
          [-16, -1, -2, 10],
          [-20, -6, 0, 17],
          [-17, -1, -14, 15],
          [-30, 6, 4, 6],
          [-31, -20, -1, 1],
          [-46, -42, -37, -24],
        ],
        [
          [-9, -13, -10, -9],
          [-12, -9, -1, -2],
          [6, -8, -2, -6],
          [-6, 1, -9, 7],
          [-5, 8, 7, -6],
          [6, 1, -7, 10],
          [4, 5, 20, -5],
          [18, 0, 19, 13],
        ],
        [
          [-69, -57, -47, -26],
          [-55, -31, -22, -4],
          [-39, -18, -9, 3],
          [-23, -3, 13, 24],
          [-29, -6, 9, 21],
          [-38, -18, -12, 1],
          [-50, -27, -24, -8],
          [-75, -52, -43, -36],
        ],
        [
          [1, 45, 85, 76],
          [53, 100, 133, 135],
          [88, 130, 169, 175],
          [103, 156, 172, 172],
          [96, 166, 199, 199],
          [92, 172, 184, 191],
          [47, 121, 116, 131],
          [11, 59, 73, 78],
        ],
      ];
  var pbonus = mg
    ? [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [3, 3, 10, 19, 16, 19, 7, -5],
        [-9, -15, 11, 15, 32, 22, 5, -22],
        [-4, -23, 6, 20, 40, 17, 4, -8],
        [13, 0, -13, 1, 11, -2, -13, 5],
        [5, -12, -7, 22, -8, -5, -15, -8],
        [-7, 7, -3, -13, 5, -16, 10, -8],
        [0, 0, 0, 0, 0, 0, 0, 0],
      ]
    : [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [-10, -6, 10, 0, 14, 7, -5, -19],
        [-10, -10, -10, 4, 4, 3, -6, -4],
        [6, -2, -8, -4, -13, -12, -10, -9],
        [10, 5, 4, -5, -5, -5, 14, 9],
        [28, 20, 21, 28, 30, 7, 6, 13],
        [0, -11, 12, 21, 25, 19, 4, 7],
        [0, 0, 0, 0, 0, 0, 0, 0],
      ];
  var i = "PNBRQK".indexOf(board(pos, square.x, square.y));
  if (i < 0) return 0;
  if (i == 0) return pbonus[7 - square.y][square.x];
  else return bonus[i - 1][7 - square.y][Math.min(square.x, 7 - square.x)];
}
export function piece_value_mg(pos, square) {
  if (square == null) return sum(pos, piece_value_mg);
  return piece_value_bonus(pos, square, true);
}
export function piece_value_eg(pos, square) {
  if (square == null) return sum(pos, piece_value_eg);
  return piece_value_bonus(pos, square, false);
}

export function psqt_mg(pos, square) {
  if (square == null) return sum(pos, psqt_mg);
  return psqt_bonus(pos, square, true);
}
export function psqt_eg(pos, square) {
  if (square == null) return sum(pos, psqt_eg);
  return psqt_bonus(pos, square, false);
}
