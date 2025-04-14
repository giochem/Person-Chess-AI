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

export function attack(pos, square) {
  if (square == null) return sum(pos, attack);
  let v = 0;
  v += pawn_attack(pos, square);
  v += king_attack(pos, square);
  v += knight_attack(pos, square);
  v += bishop_xray_attack(pos, square);
  v += rook_xray_attack(pos, square);
  v += queen_attack(pos, square);
  return v;
}
export function pawn_attack(pos, square) {
  if (square == null) return sum(pos, pawn_attack);
  let v = 0;
  if (board(pos, square.x - 1, square.y + 1) == "P") v++;
  if (board(pos, square.x + 1, square.y + 1) == "P") v++;
  return v;
}
export function king_attack(pos, square) {
  if (square == null) return sum(pos, king_attack);
  for (let i = 0; i < 8; i++) {
    let ix = ((i + (i > 3)) % 3) - 1;
    let iy = (((i + (i > 3)) / 3) << 0) - 1;
    if (board(pos, square.x + ix, square.y + iy) == "K") return 1;
  }

  return 0;
}
export function knight_attack(pos, square, s2) {
  if (square == null) return sum(pos, knight_attack);
  let v = 0;
  for (let i = 0; i < 8; i++) {
    let ix = ((i > 3) + 1) * ((i % 4 > 1) * 2 - 1);
    let iy = (2 - (i > 3)) * ((i % 2 == 0) * 2 - 1);
    let b = board(pos, square.x + ix, square.y + iy);
    if (
      b == "N" &&
      (s2 == null || (s2.x == square.x + ix && s2.y == square.y + iy)) &&
      !pinned(pos, { x: square.x + ix, y: square.y + iy })
    )
      v++;
  }
  return v;
}
export function bishop_xray_attack(pos, square, s2) {
  if (square == null) return sum(pos, bishop_xray_attack);
  let v = 0;
  for (let i = 0; i < 4; i++) {
    let ix = (i > 1) * 2 - 1;
    let iy = (i % 2 == 0) * 2 - 1;
    for (let d = 1; d < 8; d++) {
      let b = board(pos, square.x + d * ix, square.y + d * iy);
      if (
        b == "B" &&
        (s2 == null || (s2.x == square.x + d * ix && s2.y == square.y + d * iy))
      ) {
        let dir = pinned_direction(pos, {
          x: square.x + d * ix,
          y: square.y + d * iy,
        });
        if (dir == 0 || Math.abs(ix + iy * 3) == dir) v++;
      }
      if (b != "-" && b != "Q" && b != "q") break;
    }
  }
  return v;
}
export function rook_xray_attack(pos, square, s2) {
  if (square == null) return sum(pos, rook_xray_attack);
  let v = 0;
  for (let i = 0; i < 4; i++) {
    let ix = i == 0 ? -1 : i == 1 ? 1 : 0;
    let iy = i == 2 ? -1 : i == 3 ? 1 : 0;
    for (let d = 1; d < 8; d++) {
      let b = board(pos, square.x + d * ix, square.y + d * iy);
      if (
        b == "R" &&
        (s2 == null || (s2.x == square.x + d * ix && s2.y == square.y + d * iy))
      ) {
        let dir = pinned_direction(pos, {
          x: square.x + d * ix,
          y: square.y + d * iy,
        });
        if (dir == 0 || Math.abs(ix + iy * 3) == dir) v++;
      }
      if (b != "-" && b != "R" && b != "Q" && b != "q") break;
    }
  }

  return v;
}
export function queen_attack(pos, square, s2) {
  if (square == null) return sum(pos, queen_attack);
  let v = 0;
  for (let i = 0; i < 8; i++) {
    let ix = ((i + (i > 3)) % 3) - 1;
    let iy = (((i + (i > 3)) / 3) << 0) - 1;
    for (let d = 1; d < 8; d++) {
      let b = board(pos, square.x + d * ix, square.y + d * iy);
      if (
        b == "Q" &&
        (s2 == null || (s2.x == square.x + d * ix && s2.y == square.y + d * iy))
      ) {
        let dir = pinned_direction(pos, {
          x: square.x + d * ix,
          y: square.y + d * iy,
        });
        if (dir == 0 || Math.abs(ix + iy * 3) == dir) v++;
      }
      if (b != "-") break;
    }
  }
  return v;
}
export function queen_attack_diagonal(pos, square, s2) {
  if (square == null) return sum(pos, queen_attack_diagonal);
  let v = 0;
  for (let i = 0; i < 8; i++) {
    let ix = ((i + (i > 3)) % 3) - 1;
    let iy = (((i + (i > 3)) / 3) << 0) - 1;
    if (ix == 0 || iy == 0) continue;
    for (let d = 1; d < 8; d++) {
      let b = board(pos, square.x + d * ix, square.y + d * iy);
      if (
        b == "Q" &&
        (s2 == null || (s2.x == square.x + d * ix && s2.y == square.y + d * iy))
      ) {
        let dir = pinned_direction(pos, {
          x: square.x + d * ix,
          y: square.y + d * iy,
        });
        if (dir == 0 || Math.abs(ix + iy * 3) == dir) v++;
      }
      if (b != "-") break;
    }
  }
  return v;
}

export function pinned(pos, square) {
  if (square == null) return sum(pos, pinned);
  if ("PNBRQK".indexOf(board(pos, square.x, square.y)) < 0) return 0;
  return pinned_direction(pos, square) > 0 ? 1 : 0;
}
export function pinned_direction(pos, square) {
  if (square == null) return sum(pos, pinned_direction);
  if ("PNBRQK".indexOf(board(pos, square.x, square.y).toUpperCase()) < 0)
    return 0;
  let color = 1;
  if ("PNBRQK".indexOf(board(pos, square.x, square.y)) < 0) color = -1;
  for (let i = 0; i < 8; i++) {
    let ix = ((i + (i > 3)) % 3) - 1;
    let iy = (((i + (i > 3)) / 3) << 0) - 1;
    let king = false;
    for (let d = 1; d < 8; d++) {
      let b = board(pos, square.x + d * ix, square.y + d * iy);
      if (b == "K") king = true;
      if (b != "-") break;
    }
    if (king) {
      for (let d = 1; d < 8; d++) {
        let b = board(pos, square.x - d * ix, square.y - d * iy);
        if (
          b == "q" ||
          (b == "b" && ix * iy != 0) ||
          (b == "r" && ix * iy == 0)
        )
          return Math.abs(ix + iy * 3) * color;
        if (b != "-") break;
      }
    }
  }
  return 0;
}
