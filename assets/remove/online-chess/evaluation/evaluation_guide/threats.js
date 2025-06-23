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

import { winnable, winnable_total_mg, winnable_total_eg } from "./winnable.js";
export function safe_pawn(pos, square) {
  if (square == null) return sum(pos, safe_pawn);
  if (board(pos, square.x, square.y) != "P") return 0;
  if (attack(pos, square)) return 1;
  if (!attack(colorflip(pos), { x: square.x, y: 7 - square.y })) return 1;
  return 0;
}
export function threat_safe_pawn(pos, square) {
  if (square == null) return sum(pos, threat_safe_pawn);
  if ("nbrq".indexOf(board(pos, square.x, square.y)) < 0) return 0;
  if (!pawn_attack(pos, square)) return 0;
  if (
    safe_pawn(pos, { x: square.x - 1, y: square.y + 1 }) ||
    safe_pawn(pos, { x: square.x + 1, y: square.y + 1 })
  )
    return 1;
  return 0;
}
export function weak_enemies(pos, square) {
  if (square == null) return sum(pos, weak_enemies);
  if ("pnbrqk".indexOf(board(pos, square.x, square.y)) < 0) return 0;
  if (board(pos, square.x - 1, square.y - 1) == "p") return 0;
  if (board(pos, square.x + 1, square.y - 1) == "p") return 0;
  if (!attack(pos, square)) return 0;
  if (
    attack(pos, square) <= 1 &&
    attack(colorflip(pos), { x: square.x, y: 7 - square.y }) > 1
  )
    return 0;
  return 1;
}
export function minor_threat(pos, square) {
  if (square == null) return sum(pos, minor_threat);
  let type = "pnbrqk".indexOf(board(pos, square.x, square.y));
  if (type < 0) return 0;
  if (!knight_attack(pos, square) && !bishop_xray_attack(pos, square)) return 0;
  if (
    (board(pos, square.x, square.y) == "p" ||
      !(
        board(pos, square.x - 1, square.y - 1) == "p" ||
        board(pos, square.x + 1, square.y - 1) == "p" ||
        (attack(pos, square) <= 1 &&
          attack(colorflip(pos), { x: square.x, y: 7 - square.y }) > 1)
      )) &&
    !weak_enemies(pos, square)
  )
    return 0;
  return type + 1;
}
export function rook_threat(pos, square) {
  if (square == null) return sum(pos, rook_threat);
  let type = "pnbrqk".indexOf(board(pos, square.x, square.y));
  if (type < 0) return 0;
  if (!weak_enemies(pos, square)) return 0;
  if (!rook_xray_attack(pos, square)) return 0;
  return type + 1;
}
export function hanging(pos, square) {
  if (square == null) return sum(pos, hanging);
  if (!weak_enemies(pos, square)) return 0;
  if (board(pos, square.x, square.y) != "p" && attack(pos, square) > 1)
    return 1;
  if (!attack(colorflip(pos), { x: square.x, y: 7 - square.y })) return 1;
  return 0;
}
export function king_threat(pos, square) {
  if (square == null) return sum(pos, king_threat);
  if ("pnbrq".indexOf(board(pos, square.x, square.y)) < 0) return 0;
  if (!weak_enemies(pos, square)) return 0;
  if (!king_attack(pos, square)) return 0;
  return 1;
}
export function pawn_push_threat(pos, square) {
  if (square == null) return sum(pos, pawn_push_threat);
  if ("pnbrqk".indexOf(board(pos, square.x, square.y)) < 0) return 0;
  for (let ix = -1; ix <= 1; ix += 2) {
    if (
      board(pos, square.x + ix, square.y + 2) == "P" &&
      board(pos, square.x + ix, square.y + 1) == "-" &&
      board(pos, square.x + ix - 1, square.y) != "p" &&
      board(pos, square.x + ix + 1, square.y) != "p" &&
      (attack(pos, { x: square.x + ix, y: square.y + 1 }) ||
        !attack(colorflip(pos), { x: square.x + ix, y: 6 - square.y }))
    )
      return 1;

    if (
      square.y == 3 &&
      board(pos, square.x + ix, square.y + 3) == "P" &&
      board(pos, square.x + ix, square.y + 2) == "-" &&
      board(pos, square.x + ix, square.y + 1) == "-" &&
      board(pos, square.x + ix - 1, square.y) != "p" &&
      board(pos, square.x + ix + 1, square.y) != "p" &&
      (attack(pos, { x: square.x + ix, y: square.y + 1 }) ||
        !attack(colorflip(pos), { x: square.x + ix, y: 6 - square.y }))
    )
      return 1;
  }
  return 0;
}
export function slider_on_queen(pos, square) {
  if (square == null) return sum(pos, slider_on_queen);
  let pos2 = colorflip(pos);
  if (queen_count(pos2) != 1) return 0;
  if (board(pos, square.x, square.y) == "P") return 0;
  if (board(pos, square.x - 1, square.y - 1) == "p") return 0;
  if (board(pos, square.x + 1, square.y - 1) == "p") return 0;
  if (attack(pos, square) <= 1) return 0;
  if (!mobility_area(pos, square)) return 0;
  let diagonal = queen_attack_diagonal(pos2, { x: square.x, y: 7 - square.y });
  let v = queen_count(pos) == 0 ? 2 : 1;
  if (diagonal && bishop_xray_attack(pos, square)) return v;
  if (
    !diagonal &&
    rook_xray_attack(pos, square) &&
    queen_attack(pos2, { x: square.x, y: 7 - square.y })
  )
    return v;
  return 0;
}
export function knight_on_queen(pos, square) {
  if (square == null) return sum(pos, knight_on_queen);
  let pos2 = colorflip(pos);
  let qx = -1,
    qy = -1;
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      if (board(pos, x, y) == "q") {
        if (qx >= 0 || qy >= 0) return 0;
        qx = x;
        qy = y;
      }
    }
  }
  if (queen_count(pos2) != 1) return 0;
  if (board(pos, square.x, square.y) == "P") return 0;
  if (board(pos, square.x - 1, square.y - 1) == "p") return 0;
  if (board(pos, square.x + 1, square.y - 1) == "p") return 0;
  if (
    attack(pos, square) <= 1 &&
    attack(pos2, { x: square.x, y: 7 - square.y }) > 1
  )
    return 0;
  if (!mobility_area(pos, square)) return 0;
  if (!knight_attack(pos, square)) return 0;
  let v = queen_count(pos) == 0 ? 2 : 1;
  if (Math.abs(qx - square.x) == 2 && Math.abs(qy - square.y) == 1) return v;
  if (Math.abs(qx - square.x) == 1 && Math.abs(qy - square.y) == 2) return v;
  return 0;
}
export function restricted(pos, square) {
  if (square == null) return sum(pos, restricted);
  if (attack(pos, square) == 0) return 0;
  let pos2 = colorflip(pos);
  if (!attack(pos2, { x: square.x, y: 7 - square.y })) return 0;
  if (pawn_attack(pos2, { x: square.x, y: 7 - square.y }) > 0) return 0;
  if (
    attack(pos2, { x: square.x, y: 7 - square.y }) > 1 &&
    attack(pos, square) == 1
  )
    return 0;
  return 1;
}
export function weak_queen_protection(pos, square) {
  if (square == null) return sum(pos, weak_queen_protection);
  if (!weak_enemies(pos, square)) return 0;
  if (!queen_attack(colorflip(pos), { x: square.x, y: 7 - square.y })) return 0;
  return 1;
}
export function threats_mg(pos) {
  let v = 0;
  v += 69 * hanging(pos);
  v += king_threat(pos) > 0 ? 24 : 0;
  v += 48 * pawn_push_threat(pos);
  v += 173 * threat_safe_pawn(pos);
  v += 60 * slider_on_queen(pos);
  v += 16 * knight_on_queen(pos);
  v += 7 * restricted(pos);
  v += 14 * weak_queen_protection(pos);
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      let s = { x: x, y: y };
      v += [0, 5, 57, 77, 88, 79, 0][minor_threat(pos, s)];
      v += [0, 3, 37, 42, 0, 58, 0][rook_threat(pos, s)];
    }
  }
  return v;
}
export function threats_eg(pos) {
  var v = 0;
  v += 36 * hanging(pos);
  v += king_threat(pos) > 0 ? 89 : 0;
  v += 39 * pawn_push_threat(pos);
  v += 94 * threat_safe_pawn(pos);
  v += 18 * slider_on_queen(pos);
  v += 11 * knight_on_queen(pos);
  v += 7 * restricted(pos);
  for (var x = 0; x < 8; x++) {
    for (var y = 0; y < 8; y++) {
      var s = { x: x, y: y };
      v += [0, 32, 41, 56, 119, 161, 0][minor_threat(pos, s)];
      v += [0, 46, 68, 60, 38, 41, 0][rook_threat(pos, s)];
    }
  }
  return v;
}
