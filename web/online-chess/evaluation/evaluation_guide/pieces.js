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
} from "./attack.js";
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
export function outpost(pos, square) {
  if (square == null) return sum(pos, outpost);
  if (
    board(pos, square.x, square.y) != "N" &&
    board(pos, square.x, square.y) != "B"
  )
    return 0;
  if (!outpost_square(pos, square)) return 0;
  return 1;
}
export function outpost_square(pos, square) {
  if (square == null) return sum(pos, outpost_square);
  if (rank(pos, square) < 4 || rank(pos, square) > 6) return 0;
  if (
    board(pos, square.x - 1, square.y + 1) != "P" &&
    board(pos, square.x + 1, square.y + 1) != "P"
  )
    return 0;
  if (pawn_attacks_span(pos, square)) return 0;
  return 1;
}
export function reachable_outpost(pos, square) {
  if (square == null) return sum(pos, reachable_outpost);
  if (
    board(pos, square.x, square.y) != "B" &&
    board(pos, square.x, square.y) != "N"
  )
    return 0;
  var v = 0;
  for (var x = 0; x < 8; x++) {
    for (var y = 2; y < 5; y++) {
      if (
        (board(pos, square.x, square.y) == "N" &&
          "PNBRQK".indexOf(board(pos, x, y)) < 0 &&
          knight_attack(pos, { x: x, y: y }, square) &&
          outpost_square(pos, { x: x, y: y })) ||
        (board(pos, square.x, square.y) == "B" &&
          "PNBRQK".indexOf(board(pos, x, y)) < 0 &&
          bishop_xray_attack(pos, { x: x, y: y }, square) &&
          outpost_square(pos, { x: x, y: y }))
      ) {
        var support =
          board(pos, x - 1, y + 1) == "P" || board(pos, x + 1, y + 1) == "P"
            ? 2
            : 1;
        v = Math.max(v, support);
      }
    }
  }
  return v;
}
export function minor_behind_pawn(pos, square) {
  if (square == null) return sum(pos, minor_behind_pawn);
  if (
    board(pos, square.x, square.y) != "B" &&
    board(pos, square.x, square.y) != "N"
  )
    return 0;
  if (board(pos, square.x, square.y - 1).toUpperCase() != "P") return 0;
  return 1;
}
export function bishop_pawns(pos, square) {
  if (square == null) return sum(pos, bishop_pawns);
  if (board(pos, square.x, square.y) != "B") return 0;
  var c = (square.x + square.y) % 2,
    v = 0;
  var blocked = 0;
  for (var x = 0; x < 8; x++) {
    for (var y = 0; y < 8; y++) {
      if (board(pos, x, y) == "P" && c == (x + y) % 2) v++;
      if (
        board(pos, x, y) == "P" &&
        x > 1 &&
        x < 6 &&
        board(pos, x, y - 1) != "-"
      )
        blocked++;
    }
  }
  return v * (blocked + (pawn_attack(pos, square) > 0 ? 0 : 1));
}
export function rook_on_file(pos, square) {
  if (square == null) return sum(pos, rook_on_file);
  if (board(pos, square.x, square.y) != "R") return 0;
  var open = 1;
  for (var y = 0; y < 8; y++) {
    if (board(pos, square.x, y) == "P") return 0;
    if (board(pos, square.x, y) == "p") open = 0;
  }
  return open + 1;
}
export function trapped_rook(pos, square) {
  if (square == null) return sum(pos, trapped_rook);
  if (board(pos, square.x, square.y) != "R") return 0;
  if (rook_on_file(pos, square)) return 0;
  if (mobility(pos, square) > 3) return 0;
  var kx = 0,
    ky = 0;
  for (var x = 0; x < 8; x++) {
    for (var y = 0; y < 8; y++) {
      if (board(pos, x, y) == "K") {
        kx = x;
        ky = y;
      }
    }
  }
  if (kx < 4 != square.x < kx) return 0;
  return 1;
}
export function weak_queen(pos, square) {
  if (square == null) return sum(pos, weak_queen);
  if (board(pos, square.x, square.y) != "Q") return 0;
  for (var i = 0; i < 8; i++) {
    var ix = ((i + (i > 3)) % 3) - 1;
    var iy = (((i + (i > 3)) / 3) << 0) - 1;
    var count = 0;
    for (var d = 1; d < 8; d++) {
      var b = board(pos, square.x + d * ix, square.y + d * iy);
      if (b == "r" && (ix == 0 || iy == 0) && count == 1) return 1;
      if (b == "b" && ix != 0 && iy != 0 && count == 1) return 1;
      if (b != "-") count++;
    }
  }
  return 0;
}
export function king_protector(pos, square) {
  if (square == null) return sum(pos, king_protector);
  if (
    board(pos, square.x, square.y) != "N" &&
    board(pos, square.x, square.y) != "B"
  )
    return 0;
  return king_distance(pos, square);
}
export function long_diagonal_bishop(pos, square) {
  if (square == null) return sum(pos, long_diagonal_bishop);
  if (board(pos, square.x, square.y) != "B") return 0;
  if (square.x - square.y != 0 && square.x - (7 - square.y) != 0) return 0;
  var x1 = square.x,
    y1 = square.y;
  if (Math.min(x1, 7 - x1) > 2) return 0;
  for (var i = Math.min(x1, 7 - x1); i < 4; i++) {
    if (board(pos, x1, y1) == "p") return 0;
    if (board(pos, x1, y1) == "P") return 0;
    if (x1 < 4) x1++;
    else x1--;
    if (y1 < 4) y1++;
    else y1--;
  }
  return 1;
}
export function outpost_total(pos, square) {
  if (square == null) return sum(pos, outpost_total);
  if (
    board(pos, square.x, square.y) != "N" &&
    board(pos, square.x, square.y) != "B"
  )
    return 0;
  var knight = board(pos, square.x, square.y) == "N";
  var reachable = 0;
  if (!outpost(pos, square)) {
    if (!knight) return 0;
    reachable = reachable_outpost(pos, square);
    if (!reachable) return 0;
    return 1;
  }
  if (knight && (square.x < 2 || square.x > 5)) {
    var ea = 0,
      cnt = 0;
    for (var x = 0; x < 8; x++) {
      for (var y = 0; y < 8; y++) {
        if (
          ((Math.abs(square.x - x) == 2 && Math.abs(square.y - y) == 1) ||
            (Math.abs(square.x - x) == 1 && Math.abs(square.y - y) == 2)) &&
          "nbrqk".indexOf(board(pos, x, y)) >= 0
        )
          ea = 1;
        if (
          ((x < 4 && square.x < 4) || (x >= 4 && square.x >= 4)) &&
          "nbrqk".indexOf(board(pos, x, y)) >= 0
        )
          cnt++;
      }
    }
    if (!ea && cnt <= 1) return 2;
  }
  return knight ? 4 : 3;
}
export function rook_on_queen_file(pos, square) {
  if (square == null) return sum(pos, rook_on_queen_file);
  if (board(pos, square.x, square.y) != "R") return 0;
  for (var y = 0; y < 8; y++) {
    if (board(pos, square.x, y).toUpperCase() == "Q") return 1;
  }
  return 0;
}
export function bishop_xray_pawns(pos, square) {
  if (square == null) return sum(pos, bishop_xray_pawns);
  if (board(pos, square.x, square.y) != "B") return 0;
  var count = 0;
  for (var x = 0; x < 8; x++) {
    for (var y = 0; y < 8; y++) {
      if (
        board(pos, x, y) == "p" &&
        Math.abs(square.x - x) == Math.abs(square.y - y)
      )
        count++;
    }
  }
  return count;
}
export function rook_on_king_ring(pos, square) {
  if (square == null) return sum(pos, rook_on_king_ring);
  if (board(pos, square.x, square.y) != "R") return 0;
  if (king_attackers_count(pos, square) > 0) return 0;
  for (var y = 0; y < 8; y++) {
    if (king_ring(pos, { x: square.x, y: y })) return 1;
  }
  return 0;
}
export function bishop_on_king_ring(pos, square) {
  if (square == null) return sum(pos, bishop_on_king_ring);
  if (board(pos, square.x, square.y) != "B") return 0;
  if (king_attackers_count(pos, square) > 0) return 0;
  for (var i = 0; i < 4; i++) {
    var ix = (i > 1) * 2 - 1;
    var iy = (i % 2 == 0) * 2 - 1;
    for (var d = 1; d < 8; d++) {
      var x = square.x + d * ix,
        y = square.y + d * iy;
      if (board(pos, x, y) == "x") break;
      if (king_ring(pos, { x: x, y: y })) return 1;
      if (board(pos, x, y).toUpperCase() == "P") break;
    }
  }
  return 0;
}
export function queen_infiltration(pos, square) {
  if (square == null) return sum(pos, queen_infiltration);
  if (board(pos, square.x, square.y) != "Q") return 0;
  if (square.y > 3) return 0;
  if (board(pos, square.x + 1, square.y - 1) == "p") return 0;
  if (board(pos, square.x - 1, square.y - 1) == "p") return 0;
  if (pawn_attacks_span(pos, square)) return 0;
  return 1;
}
export function pieces_mg(pos, square) {
  if (square == null) return sum(pos, pieces_mg);
  if ("NBRQ".indexOf(board(pos, square.x, square.y)) < 0) return 0;
  var v = 0;
  v += [0, 31, -7, 30, 56][outpost_total(pos, square)];
  v += 18 * minor_behind_pawn(pos, square);
  v -= 3 * bishop_pawns(pos, square);
  v -= 4 * bishop_xray_pawns(pos, square);
  v += 6 * rook_on_queen_file(pos, square);
  v += 16 * rook_on_king_ring(pos, square);
  v += 24 * bishop_on_king_ring(pos, square);
  v += [0, 19, 48][rook_on_file(pos, square)];
  v -= trapped_rook(pos, square) * 55 * (pos.c[0] || pos.c[1] ? 1 : 2);
  v -= 56 * weak_queen(pos, square);
  v -= 2 * queen_infiltration(pos, square);
  v -=
    (board(pos, square.x, square.y) == "N" ? 8 : 6) *
    king_protector(pos, square);
  v += 45 * long_diagonal_bishop(pos, square);
  return v;
}
export function pieces_eg(pos, square) {
  if (square == null) return sum(pos, pieces_eg);
  if ("NBRQ".indexOf(board(pos, square.x, square.y)) < 0) return 0;
  var v = 0;
  v += [0, 22, 36, 23, 36][outpost_total(pos, square)];
  v += 3 * minor_behind_pawn(pos, square);
  v -= 7 * bishop_pawns(pos, square);
  v -= 5 * bishop_xray_pawns(pos, square);
  v += 11 * rook_on_queen_file(pos, square);
  v += [0, 7, 29][rook_on_file(pos, square)];
  v -= trapped_rook(pos, square) * 13 * (pos.c[0] || pos.c[1] ? 1 : 2);
  v -= 15 * weak_queen(pos, square);
  v += 14 * queen_infiltration(pos, square);
  v -= 9 * king_protector(pos, square);
  return v;
}
