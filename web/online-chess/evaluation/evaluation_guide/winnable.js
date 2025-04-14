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

export function winnable(pos, square) {
  if (square != null) return 0;
  var pawns = 0,
    kx = [0, 0],
    ky = [0, 0],
    flanks = [0, 0];
  for (var x = 0; x < 8; x++) {
    var open = [0, 0];
    for (var y = 0; y < 8; y++) {
      if (board(pos, x, y).toUpperCase() == "P") {
        open[board(pos, x, y) == "P" ? 0 : 1] = 1;
        pawns++;
      }
      if (board(pos, x, y).toUpperCase() == "K") {
        kx[board(pos, x, y) == "K" ? 0 : 1] = x;
        ky[board(pos, x, y) == "K" ? 0 : 1] = y;
      }
    }
    if (open[0] + open[1] > 0) flanks[x < 4 ? 0 : 1] = 1;
  }
  var pos2 = colorflip(pos);
  var passedCount = candidate_passed(pos) + candidate_passed(pos2);
  var bothFlanks = flanks[0] && flanks[1] ? 1 : 0;
  var outflanking = Math.abs(kx[0] - kx[1]) - Math.abs(ky[0] - ky[1]);
  var purePawn = non_pawn_material(pos) + non_pawn_material(pos2) == 0 ? 1 : 0;
  var almostUnwinnable = outflanking < 0 && bothFlanks == 0;
  var infiltration = ky[0] < 4 || ky[1] > 3 ? 1 : 0;
  return (
    9 * passedCount +
    12 * pawns +
    9 * outflanking +
    21 * bothFlanks +
    24 * infiltration +
    51 * purePawn -
    43 * almostUnwinnable -
    110
  );
}
export function winnable_total_mg(pos, v) {
  if (v == null) v = middle_game_evaluation(pos, true);
  return (
    (v > 0 ? 1 : v < 0 ? -1 : 0) *
    Math.max(Math.min(winnable(pos) + 50, 0), -Math.abs(v))
  );
}
export function winnable_total_eg(pos, v) {
  if (v == null) v = end_game_evaluation(pos, true);
  return (v > 0 ? 1 : v < 0 ? -1 : 0) * Math.max(winnable(pos), -Math.abs(v));
}
