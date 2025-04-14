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
export function pawnless_flank(pos) {
  var pawns = [0, 0, 0, 0, 0, 0, 0, 0],
    kx = 0;
  for (var x = 0; x < 8; x++) {
    for (var y = 0; y < 8; y++) {
      if (board(pos, x, y).toUpperCase() == "P") pawns[x]++;
      if (board(pos, x, y) == "k") kx = x;
    }
  }
  var sum;
  if (kx == 0) sum = pawns[0] + pawns[1] + pawns[2];
  else if (kx < 3) sum = pawns[0] + pawns[1] + pawns[2] + pawns[3];
  else if (kx < 5) sum = pawns[2] + pawns[3] + pawns[4] + pawns[5];
  else if (kx < 7) sum = pawns[4] + pawns[5] + pawns[6] + pawns[7];
  else sum = pawns[5] + pawns[6] + pawns[7];
  return sum == 0 ? 1 : 0;
}
export function strength_square(pos, square) {
  if (square == null) return sum(pos, strength_square);
  let v = 5;
  let kx = Math.min(6, Math.max(1, square.x));
  let weakness = [
    [-6, 81, 93, 58, 39, 18, 25],
    [-43, 61, 35, -49, -29, -11, -63],
    [-10, 75, 23, -2, 32, 3, -45],
    [-39, -13, -29, -52, -48, -67, -166],
  ];
  for (let x = kx - 1; x <= kx + 1; x++) {
    let us = 0;
    for (let y = 7; y >= square.y; y--) {
      if (
        board(pos, x, y) == "p" &&
        board(pos, x - 1, y + 1) != "P" &&
        board(pos, x + 1, y + 1) != "P"
      )
        us = y;
    }
    let f = Math.min(x, 7 - x);
    v += weakness[f][us] || 0;
  }
  return v;
}
export function storm_square(pos, square, eg) {
  if (square == null) return sum(pos, storm_square);
  let v = 0,
    ev = 5;
  let kx = Math.min(6, Math.max(1, square.x));
  let unblockedstorm = [
    [85, -289, -166, 97, 50, 45, 50],
    [46, -25, 122, 45, 37, -10, 20],
    [-6, 51, 168, 34, -2, -22, -14],
    [-15, -11, 101, 4, 11, -15, -29],
  ];
  let blockedstorm = [
    [0, 0, 76, -10, -7, -4, -1],
    [0, 0, 78, 15, 10, 6, 2],
  ];
  for (let x = kx - 1; x <= kx + 1; x++) {
    let us = 0,
      them = 0;
    for (let y = 7; y >= square.y; y--) {
      if (
        board(pos, x, y) == "p" &&
        board(pos, x - 1, y + 1) != "P" &&
        board(pos, x + 1, y + 1) != "P"
      )
        us = y;
      if (board(pos, x, y) == "P") them = y;
    }
    let f = Math.min(x, 7 - x);
    if (us > 0 && them == us + 1) {
      v += blockedstorm[0][them];
      ev += blockedstorm[1][them];
    } else v += unblockedstorm[f][them];
  }
  return eg ? ev : v;
}
export function shelter_strength(pos, square) {
  let w = 0,
    s = 1024,
    tx = null;
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      if (
        board(pos, x, y) == "k" ||
        (pos.c[2] && x == 6 && y == 0) ||
        (pos.c[3] && x == 2 && y == 0)
      ) {
        let w1 = strength_square(pos, { x: x, y: y });
        let s1 = storm_square(pos, { x: x, y: y });
        if (s1 - w1 < s - w) {
          w = w1;
          s = s1;
          tx = Math.max(1, Math.min(6, x));
        }
      }
    }
  }
  if (square == null) return w;
  if (
    tx != null &&
    board(pos, square.x, square.y) == "p" &&
    square.x >= tx - 1 &&
    square.x <= tx + 1
  ) {
    for (let y = square.y - 1; y >= 0; y--)
      if (board(pos, square.x, y) == "p") return 0;
    return 1;
  }
  return 0;
}
export function shelter_storm(pos, square) {
  let w = 0,
    s = 1024,
    tx = null;
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      if (
        board(pos, x, y) == "k" ||
        (pos.c[2] && x == 6 && y == 0) ||
        (pos.c[3] && x == 2 && y == 0)
      ) {
        let w1 = strength_square(pos, { x: x, y: y });
        let s1 = storm_square(pos, { x: x, y: y });
        if (s1 - w1 < s - w) {
          w = w1;
          s = s1;
          tx = Math.max(1, Math.min(6, x));
        }
      }
    }
  }
  if (square == null) return s;
  if (
    tx != null &&
    board(pos, square.x, square.y).toUpperCase() == "P" &&
    square.x >= tx - 1 &&
    square.x <= tx + 1
  ) {
    for (let y = square.y - 1; y >= 0; y--)
      if (board(pos, square.x, y) == board(pos, square.x, square.y)) return 0;
    return 1;
  }
  return 0;
}

export function king_pawn_distance(pos, square) {
  var v = 6,
    kx = 0,
    ky = 0,
    px = 0,
    py = 0;
  for (var x = 0; x < 8; x++) {
    for (var y = 0; y < 8; y++) {
      if (board(pos, x, y) == "K") {
        kx = x;
        ky = y;
      }
    }
  }
  for (var x = 0; x < 8; x++) {
    for (var y = 0; y < 8; y++) {
      var dist = Math.max(Math.abs(x - kx), Math.abs(y - ky));
      if (board(pos, x, y) == "P" && dist < v) {
        px = x;
        py = y;
        v = dist;
      }
    }
  }
  if (square == null || (square.x == px && square.y == py)) return v;
  return 0;
}
export function check(pos, square, type) {
  if (square == null) return sum(pos, check);
  if (
    (rook_xray_attack(pos, square) &&
      (type == null || type == 2 || type == 4)) ||
    (queen_attack(pos, square) && (type == null || type == 3))
  ) {
    for (let i = 0; i < 4; i++) {
      let ix = i == 0 ? -1 : i == 1 ? 1 : 0;
      let iy = i == 2 ? -1 : i == 3 ? 1 : 0;
      for (let d = 1; d < 8; d++) {
        let b = board(pos, square.x + d * ix, square.y + d * iy);
        if (b == "k") return 1;
        if (b != "-" && b != "q") break;
      }
    }
  }
  if (
    (bishop_xray_attack(pos, square) &&
      (type == null || type == 1 || type == 4)) ||
    (queen_attack(pos, square) && (type == null || type == 3))
  ) {
    for (let i = 0; i < 4; i++) {
      let ix = (i > 1) * 2 - 1;
      let iy = (i % 2 == 0) * 2 - 1;
      for (let d = 1; d < 8; d++) {
        let b = board(pos, square.x + d * ix, square.y + d * iy);
        if (b == "k") return 1;
        if (b != "-" && b != "q") break;
      }
    }
  }
  if (knight_attack(pos, square) && (type == null || type == 0 || type == 4)) {
    if (
      board(pos, square.x + 2, square.y + 1) == "k" ||
      board(pos, square.x + 2, square.y - 1) == "k" ||
      board(pos, square.x + 1, square.y + 2) == "k" ||
      board(pos, square.x + 1, square.y - 2) == "k" ||
      board(pos, square.x - 2, square.y + 1) == "k" ||
      board(pos, square.x - 2, square.y - 1) == "k" ||
      board(pos, square.x - 1, square.y + 2) == "k" ||
      board(pos, square.x - 1, square.y - 2) == "k"
    )
      return 1;
  }
  return 0;
}

export function safe_check(pos, square, type) {
  if (square == null) return sum(pos, safe_check, type);
  if ("PNBRQK".indexOf(board(pos, square.x, square.y)) >= 0) return 0;
  if (!check(pos, square, type)) return 0;
  let pos2 = colorflip(pos);
  if (type == 3 && safe_check(pos, square, 2)) return 0;
  if (type == 1 && safe_check(pos, square, 3)) return 0;
  if (
    (!attack(pos2, { x: square.x, y: 7 - square.y }) ||
      (weak_squares(pos, square) && attack(pos, square) > 1)) &&
    (type != 3 || !queen_attack(pos2, { x: square.x, y: 7 - square.y }))
  )
    return 1;
  return 0;
}
export function unsafe_checks(pos, square) {
  if (square == null) return sum(pos, unsafe_checks);
  if (check(pos, square, 0) && safe_check(pos, null, 0) == 0) return 1;
  if (check(pos, square, 1) && safe_check(pos, null, 1) == 0) return 1;
  if (check(pos, square, 2) && safe_check(pos, null, 2) == 0) return 1;
  return 0;
}

export function king_attackers_count(pos, square) {
  if (square == null) return sum(pos, king_attackers_count);
  if ("PNBRQ".indexOf(board(pos, square.x, square.y)) < 0) return 0;
  if (board(pos, square.x, square.y) == "P") {
    let v = 0;
    for (let dir = -1; dir <= 1; dir += 2) {
      let fr = board(pos, square.x + dir * 2, square.y) == "P";
      if (
        square.x + dir >= 0 &&
        square.x + dir <= 7 &&
        king_ring(pos, { x: square.x + dir, y: square.y - 1 }, true)
      )
        v = v + (fr ? 0.5 : 1);
    }
    return v;
  }
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      let s2 = { x: x, y: y };
      if (king_ring(pos, s2)) {
        if (
          knight_attack(pos, s2, square) ||
          bishop_xray_attack(pos, s2, square) ||
          rook_xray_attack(pos, s2, square) ||
          queen_attack(pos, s2, square)
        )
          return 1;
      }
    }
  }
  return 0;
}
export function king_attackers_weight(pos, square) {
  if (square == null) return sum(pos, king_attackers_weight);
  if (king_attackers_count(pos, square)) {
    return [0, 81, 52, 44, 10]["PNBRQ".indexOf(board(pos, square.x, square.y))];
  }
  return 0;
}
export function king_attacks(pos, square) {
  if (square == null) return sum(pos, king_attacks);
  if ("NBRQ".indexOf(board(pos, square.x, square.y)) < 0) return 0;
  if (king_attackers_count(pos, square) == 0) return 0;
  var kx = 0,
    ky = 0,
    v = 0;
  for (var x = 0; x < 8; x++) {
    for (var y = 0; y < 8; y++) {
      if (board(pos, x, y) == "k") {
        kx = x;
        ky = y;
      }
    }
  }
  for (var x = kx - 1; x <= kx + 1; x++) {
    for (var y = ky - 1; y <= ky + 1; y++) {
      var s2 = { x: x, y: y };
      if (x >= 0 && y >= 0 && x <= 7 && y <= 7 && (x != kx || y != ky)) {
        v += knight_attack(pos, s2, square);
        v += bishop_xray_attack(pos, s2, square);
        v += rook_xray_attack(pos, s2, square);
        v += queen_attack(pos, s2, square);
      }
    }
  }
  return v;
}
export function weak_bonus(pos, square) {
  if (square == null) return sum(pos, weak_bonus);
  if (!weak_squares(pos, square)) return 0;
  if (!king_ring(pos, square)) return 0;
  return 1;
}
export function weak_squares(pos, square) {
  if (square == null) return sum(pos, weak_squares);
  if (attack(pos, square)) {
    let pos2 = colorflip(pos);
    let attack_val = attack(pos2, { x: square.x, y: 7 - square.y });
    if (attack_val >= 2) return 0;
    if (attack_val == 0) return 1;
    if (
      king_attack(pos2, { x: square.x, y: 7 - square.y }) ||
      queen_attack(pos2, { x: square.x, y: 7 - square.y })
    )
      return 1;
  }
  return 0;
}
export function knight_defender(pos, square) {
  if (square == null) return sum(pos, knight_defender);
  if (knight_attack(pos, square) && king_attack(pos, square)) return 1;
  return 0;
}
export function endgame_shelter(pos, square) {
  let w = 0,
    s = 1024,
    e = null;
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      if (
        board(pos, x, y) == "k" ||
        (pos.c[2] && x == 6 && y == 0) ||
        (pos.c[3] && x == 2 && y == 0)
      ) {
        const w1 = strength_square(pos, { x: x, y: y });
        const s1 = storm_square(pos, { x: x, y: y });
        const e1 = storm_square(pos, { x: x, y: y }, true);
        if (s1 - w1 < s - w) {
          w = w1;
          s = s1;
          e = e1;
        }
      }
    }
  }
  if (square == null) return e;
  return 0;
}
export function blockers_for_king(pos, square) {
  if (square == null) return sum(pos, blockers_for_king);
  if (pinned_direction(colorflip(pos), { x: square.x, y: 7 - square.y }))
    return 1;
  return 0;
}
export function flank_attack(pos, square) {
  if (square == null) return sum(pos, flank_attack);
  if (square.y > 4) return 0;
  for (var x = 0; x < 8; x++) {
    for (var y = 0; y < 8; y++) {
      if (board(pos, x, y) == "k") {
        if (x == 0 && square.x > 2) return 0;
        if (x < 3 && square.x > 3) return 0;
        if (x >= 3 && x < 5 && (square.x < 2 || square.x > 5)) return 0;
        if (x >= 5 && square.x < 4) return 0;
        if (x == 7 && square.x < 5) return 0;
      }
    }
  }
  var a = attack(pos, square);
  if (!a) return 0;
  return a > 1 ? 2 : 1;
}
export function flank_defense(pos, square) {
  if (square == null) return sum(pos, flank_defense);
  if (square.y > 4) return 0;
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      if (board(pos, x, y) == "k") {
        if (x == 0 && square.x > 2) return 0;
        if (x < 3 && square.x > 3) return 0;
        if (x >= 3 && x < 5 && (square.x < 2 || square.x > 5)) return 0;
        if (x >= 5 && square.x < 4) return 0;
        if (x == 7 && square.x < 5) return 0;
      }
    }
  }
  return attack(colorflip(pos), { x: square.x, y: 7 - square.y }) > 0 ? 1 : 0;
}
export function king_danger(pos) {
  let count = king_attackers_count(pos);
  let weight = king_attackers_weight(pos);
  let kingAttacks = king_attacks(pos);
  let weak = weak_bonus(pos);
  let unsafeChecks = unsafe_checks(pos);
  let blockersForKing = blockers_for_king(pos);
  let kingFlankAttack = flank_attack(pos);
  let kingFlankDefense = flank_defense(pos);
  let noQueen = queen_count(pos) > 0 ? 0 : 1;
  let v =
    count * weight +
    69 * kingAttacks +
    185 * weak -
    100 * (knight_defender(colorflip(pos)) > 0) +
    148 * unsafeChecks +
    98 * blockersForKing -
    4 * kingFlankDefense +
    (((3 * kingFlankAttack * kingFlankAttack) / 8) << 0) -
    873 * noQueen -
    (((6 * (shelter_strength(pos) - shelter_storm(pos))) / 8) << 0) +
    mobility_mg(pos) -
    mobility_mg(colorflip(pos)) +
    37 +
    ((772 * Math.min(safe_check(pos, null, 3), 1.45)) << 0) +
    ((1084 * Math.min(safe_check(pos, null, 2), 1.75)) << 0) +
    ((645 * Math.min(safe_check(pos, null, 1), 1.5)) << 0) +
    ((792 * Math.min(safe_check(pos, null, 0), 1.62)) << 0);
  if (v > 100) return v;
  return 0;
}
export function king_mg(pos) {
  let v = 0;
  let kd = king_danger(pos);
  v -= shelter_strength(pos);
  v += shelter_storm(pos);
  v += ((kd * kd) / 4096) << 0;
  v += 8 * flank_attack(pos);
  v += 17 * pawnless_flank(pos);
  return v;
}
export function king_eg(pos) {
  var v = 0;
  v -= 16 * king_pawn_distance(pos);
  v += endgame_shelter(pos);
  v += 95 * pawnless_flank(pos);
  v += (king_danger(pos) / 16) << 0;
  return v;
}
