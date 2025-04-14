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
  passed_mg,
  passed_eg,
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

export function main_evaluation(pos) {
  const mg = middle_game_evaluation(pos);
  let eg = end_game_evaluation(pos);
  const p = phase(pos),
    rule50_val = rule50(pos);
  eg = (eg * scale_factor(pos, eg)) / 64;
  let v = ((mg * p + ((eg * (128 - p)) << 0)) / 128) << 0;
  if (arguments.length == 1) v = ((v / 16) << 0) * 16;
  v += tempo(pos);
  v = ((v * (100 - rule50_val)) / 100) << 0;
  return v;
}
export function middle_game_evaluation(pos, nowinnable) {
  let v = 0;
  v += piece_value_mg(pos) - piece_value_mg(colorflip(pos));
  v += psqt_mg(pos) - psqt_mg(colorflip(pos));
  v += imbalance_total(pos);
  v += pawns_mg(pos) - pawns_mg(colorflip(pos));
  v += pieces_mg(pos) - pieces_mg(colorflip(pos));
  v += mobility_mg(pos) - mobility_mg(colorflip(pos));
  v += threats_mg(pos) - threats_mg(colorflip(pos));
  v += passed_mg(pos) - passed_mg(colorflip(pos));
  v += space(pos) - space(colorflip(pos));
  v += king_mg(pos) - king_mg(colorflip(pos));
  if (!nowinnable) v += winnable_total_mg(pos, v);
  return v;
}
function end_game_evaluation(pos, nowinnable) {
  var v = 0;
  v += piece_value_eg(pos) - piece_value_eg(colorflip(pos));
  v += psqt_eg(pos) - psqt_eg(colorflip(pos));
  v += imbalance_total(pos);
  v += pawns_eg(pos) - pawns_eg(colorflip(pos));
  v += pieces_eg(pos) - pieces_eg(colorflip(pos));
  v += mobility_eg(pos) - mobility_eg(colorflip(pos));
  v += threats_eg(pos) - threats_eg(colorflip(pos));
  v += passed_eg(pos) - passed_eg(colorflip(pos));
  v += king_eg(pos) - king_eg(colorflip(pos));
  if (!nowinnable) v += winnable_total_eg(pos, v);
  return v;
}
export function phase(pos) {
  var midgameLimit = 15258,
    endgameLimit = 3915;
  var npm = non_pawn_material(pos) + non_pawn_material(colorflip(pos));
  npm = Math.max(endgameLimit, Math.min(npm, midgameLimit));
  return (((npm - endgameLimit) * 128) / (midgameLimit - endgameLimit)) << 0;
}
export function rule50(pos, square) {
  if (square != null) return 0;
  return pos.m[0];
}
export function tempo(pos, square) {
  if (square != null) return 0;
  return 28 * (pos.w ? 1 : -1);
}
export function scale_factor(pos, eg) {
  if (eg == null) eg = end_game_evaluation(pos);
  var pos2 = colorflip(pos);
  var pos_w = eg > 0 ? pos : pos2;
  var pos_b = eg > 0 ? pos2 : pos;
  var sf = 64;
  var pc_w = pawn_count(pos_w),
    pc_b = pawn_count(pos_b);
  var qc_w = queen_count(pos_w),
    qc_b = queen_count(pos_b);
  var bc_w = bishop_count(pos_w),
    bc_b = bishop_count(pos_b);
  var nc_w = knight_count(pos_w),
    nc_b = knight_count(pos_b);
  var npm_w = non_pawn_material(pos_w),
    npm_b = non_pawn_material(pos_b);
  var bishopValueMg = 825,
    bishopValueEg = 915,
    rookValueMg = 1276;
  if (pc_w == 0 && npm_w - npm_b <= bishopValueMg)
    sf = npm_w < rookValueMg ? 0 : npm_b <= bishopValueMg ? 4 : 14;
  if (sf == 64) {
    var ob = opposite_bishops(pos);
    if (ob && npm_w == bishopValueMg && npm_b == bishopValueMg) {
      sf = 22 + 4 * candidate_passed(pos_w);
    } else if (ob) {
      sf = 22 + 3 * piece_count(pos_w);
    } else {
      if (npm_w == rookValueMg && npm_b == rookValueMg && pc_w - pc_b <= 1) {
        var pawnking_b = 0,
          pcw_flank = [0, 0];
        for (var x = 0; x < 8; x++) {
          for (var y = 0; y < 8; y++) {
            if (board(pos_w, x, y) == "P") pcw_flank[x < 4 ? 1 : 0] = 1;
            if (board(pos_b, x, y) == "K") {
              for (var ix = -1; ix <= 1; ix++) {
                for (var iy = -1; iy <= 1; iy++) {
                  if (board(pos_b, x + ix, y + iy) == "P") pawnking_b = 1;
                }
              }
            }
          }
        }
        if (pcw_flank[0] != pcw_flank[1] && pawnking_b) return 36;
      }
      if (qc_w + qc_b == 1) {
        sf = 37 + 3 * (qc_w == 1 ? bc_b + nc_b : bc_w + nc_w);
      } else {
        sf = Math.min(sf, 36 + 7 * pc_w);
      }
    }
  }
  return sf;
}
