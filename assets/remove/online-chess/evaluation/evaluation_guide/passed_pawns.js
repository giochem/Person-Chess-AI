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
export function candidate_passed(pos, square) {
  if (square == null) return sum(pos, candidate_passed);
  if (board(pos, square.x, square.y) != "P") return 0;
  let ty1 = 8,
    ty2 = 8,
    oy = 8;
  for (let y = square.y - 1; y >= 0; y--) {
    if (board(pos, square.x, y) == "P") return 0;
    if (board(pos, square.x, y) == "p") ty1 = y;
    if (
      board(pos, square.x - 1, y) == "p" ||
      board(pos, square.x + 1, y) == "p"
    )
      ty2 = y;
  }
  if (ty1 == 8 && ty2 >= square.y - 1) return 1;
  if (ty2 < square.y - 2 || ty1 < square.y - 1) return 0;
  if (ty2 >= square.y && ty1 == square.y - 1 && square.y < 4) {
    if (
      board(pos, square.x - 1, square.y + 1) == "P" &&
      board(pos, square.x - 1, square.y) != "p" &&
      board(pos, square.x - 2, square.y - 1) != "p"
    )
      return 1;
    if (
      board(pos, square.x + 1, square.y + 1) == "P" &&
      board(pos, square.x + 1, square.y) != "p" &&
      board(pos, square.x + 2, square.y - 1) != "p"
    )
      return 1;
  }
  if (board(pos, square.x, square.y - 1) == "p") return 0;
  let lever =
    (board(pos, square.x - 1, square.y - 1) == "p" ? 1 : 0) +
    (board(pos, square.x + 1, square.y - 1) == "p" ? 1 : 0);
  let leverpush =
    (board(pos, square.x - 1, square.y - 2) == "p" ? 1 : 0) +
    (board(pos, square.x + 1, square.y - 2) == "p" ? 1 : 0);
  let phalanx =
    (board(pos, square.x - 1, square.y) == "P" ? 1 : 0) +
    (board(pos, square.x + 1, square.y) == "P" ? 1 : 0);
  if (lever - supported(pos, square) > 1) return 0;
  if (leverpush - phalanx > 0) return 0;
  if (lever > 0 && leverpush > 0) return 0;
  return 1;
}
export function king_proximity(pos, square) {
  if (square == null) return sum(pos, king_proximity);
  if (!passed_leverable(pos, square)) return 0;
  var r = rank(pos, square) - 1;
  var w = r > 2 ? 5 * r - 13 : 0;
  var v = 0;
  if (w <= 0) return 0;
  for (var x = 0; x < 8; x++) {
    for (var y = 0; y < 8; y++) {
      if (board(pos, x, y) == "k") {
        v +=
          (((Math.min(
            Math.max(Math.abs(y - square.y + 1), Math.abs(x - square.x)),
            5
          ) *
            19) /
            4) <<
            0) *
          w;
      }
      if (board(pos, x, y) == "K") {
        v -=
          Math.min(
            Math.max(Math.abs(y - square.y + 1), Math.abs(x - square.x)),
            5
          ) *
          2 *
          w;
        if (square.y > 1) {
          v -=
            Math.min(
              Math.max(Math.abs(y - square.y + 2), Math.abs(x - square.x)),
              5
            ) * w;
        }
      }
    }
  }
  return v;
}
export function passed_block(pos, square) {
  if (square == null) return sum(pos, passed_block);
  if (!passed_leverable(pos, square)) return 0;
  if (rank(pos, square) < 4) return 0;
  if (board(pos, square.x, square.y - 1) != "-") return 0;
  var r = rank(pos, square) - 1;
  var w = r > 2 ? 5 * r - 13 : 0;
  var pos2 = colorflip(pos);
  var defended = 0,
    unsafe = 0,
    wunsafe = 0,
    defended1 = 0,
    unsafe1 = 0;
  for (var y = square.y - 1; y >= 0; y--) {
    if (attack(pos, { x: square.x, y: y })) defended++;
    if (attack(pos2, { x: square.x, y: 7 - y })) unsafe++;
    if (attack(pos2, { x: square.x - 1, y: 7 - y })) wunsafe++;
    if (attack(pos2, { x: square.x + 1, y: 7 - y })) wunsafe++;
    if (y == square.y - 1) {
      defended1 = defended;
      unsafe1 = unsafe;
    }
  }
  for (var y = square.y + 1; y < 8; y++) {
    if (board(pos, square.x, y) == "R" || board(pos, square.x, y) == "Q")
      defended1 = defended = square.y;
    if (board(pos, square.x, y) == "r" || board(pos, square.x, y) == "q")
      unsafe1 = unsafe = square.y;
  }
  var k =
    (unsafe == 0 && wunsafe == 0
      ? 35
      : unsafe == 0
      ? 20
      : unsafe1 == 0
      ? 9
      : 0) + (defended1 != 0 ? 5 : 0);
  return k * w;
}
export function passed_file(pos, square) {
  if (square == null) return sum(pos, passed_file);
  if (!passed_leverable(pos, square)) return 0;
  const file_val = file(pos, square);
  return Math.min(file_val - 1, 8 - file_val);
}
export function passed_rank(pos, square) {
  if (square == null) return sum(pos, passed_rank);
  if (!passed_leverable(pos, square)) return 0;
  return rank(pos, square) - 1;
}
export function passed_leverable(pos, square) {
  if (square == null) return sum(pos, passed_leverable);
  if (!candidate_passed(pos, square)) return 0;
  if (board(pos, square.x, square.y - 1) != "p") return 1;
  let pos2 = colorflip(pos);
  for (let i = -1; i <= 1; i += 2) {
    let s1 = { x: square.x + i, y: square.y };
    let s2 = { x: square.x + i, y: 7 - square.y };
    if (
      board(pos, square.x + i, square.y + 1) == "P" &&
      "pnbrqk".indexOf(board(pos, square.x + i, square.y)) < 0 &&
      (attack(pos, s1) > 0 || attack(pos2, s2) <= 1)
    )
      return 1;
  }
  return 0;
}
export function passed_mg(pos, square) {
  if (square == null) return sum(pos, passed_mg);
  if (!passed_leverable(pos, square)) return 0;
  let v = 0;
  v += [0, 10, 17, 15, 62, 168, 276][passed_rank(pos, square)];
  v += passed_block(pos, square);
  v -= 11 * passed_file(pos, square);
  return v;
}
export function passed_eg(pos, square) {
  if (square == null) return sum(pos, passed_eg);
  if (!passed_leverable(pos, square)) return 0;
  var v = 0;
  v += king_proximity(pos, square);
  v += [0, 28, 33, 41, 72, 177, 260][passed_rank(pos, square)];
  v += passed_block(pos, square);
  v -= 8 * passed_file(pos, square);
  return v;
}
