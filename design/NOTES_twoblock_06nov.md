# Two-block design (the #1 team's structure), 6 Nov 2026

Target puzzles: n = 498 (K = 22, board 22.7939, ours 22.8231) and n = 544 (K = 23, board 23.7941, ours 23.8238).
Validation: n = 299 (leader's published packing at 17.8133; ours 17.8012, board 17.7936).

## What the leader's n = 299 packing is

Two thick blocks of tilted squares and upright squares everywhere else:

- Left block: 52 squares at atan(1/2) = 26.57 deg standing against the left wall from the bottom corner up to y = 12.
  It is a brick lattice: rows of 6 squares along e1 = (cos t, sin t), consecutive rows stacked straight up the wall with
  step 1 / cos t = 1.118 (offset (tan t, 1) in the square frame, rows touch and interlock, density 1). Because the step
  is vertical, every row covers the same strip of width W = 6 cos t + sin t = 5.81 = L - 12, and 12 upright columns
  fit beside it exactly. The condition that sets the angle is N cos t + sin t = N - 1 + f (f = fractional part of L):
  the band replaces N - 1 upright columns plus the fractional slack of the box.
- Top block: 38 squares at 63 deg (the same construction reflected onto the top wall, from the right corner leftward,
  height 5.81 = L - 12, 12 upright rows below it).
- The blocks are bent (a slide of one column in the middle of the left block, with single upright fillers on either
  side of the slide) and their ends are staircases of partial rows meshing with upright fillers.
- Waste 18.3 area units at n = 299, L = 17.8133. The gain over the K x K grid is roughly f x (band length) per band,
  minus the slanted triangles at the four band ends.

## Code (design/)

- `twoblock.py`: rigid assembly. Parameters N1, N2 (squares across each band), h1, h2 (band lengths along the walls),
  q1, q2 (lattice phases along the wall), x01, x02 in {0, 1} (band offset from the wall), m1, m2 (mirror: rows up-right
  or down-right). Band interiors are fixed, band squares within 2.5 of the ends are free, upright candidates are the
  four wall-aligned unit grids (optionally finer offsets, `--fine`, and band lattice squares beyond the band length,
  `--ext`); a MILP (scipy milp / HiGHS, witness-point cliques for upright pairs, pairwise SAT constraints otherwise)
  picks the maximum non-overlapping set. 0.05 to 0.2 s per assembly at n ~ 500; a full pairwise SAT check (1e-7)
  validates the result. Output: competition csv (radians, box from 0).
- `tbsearch.py`: random restarts plus hill climbing over (h1, h2, q1, q2, x01, x02, m1, m2) for a list of (N1, N2);
  can continue from an earlier `_best.json`.
- `tbfill.py`: adds the missing squares at the largest-clearance spots (fine grid, several angles, penetration depth
  by SAT; the box is hard), or `--rows`: inserts each missing square into the longest upright row, re-spaced so the
  overlap is spread thinly (1/m per pair).

## Rigid counts (best found, about 10 min of one core per (n, L), plus focused continuations)

| n | L | count | deficit | params (N1 h1 q1 N2 h2 q2 x01 x02 m1 m2) |
|---|---|---|---|---|
| 299 | 17.8133 | 297 | 2 | 5 12.97 0.73 6 13.91 0.50 0 0 1 0 |
| 299 | 17.794 | 296 | 3 | 6 13.0 1.05 5 12.9 0.18 0 0 1 1 |
| 498 | 22.794 | 496 | 2 | 6 16.48 0.54 6 16.18 0.52 0 0 1 0 |
| 498 | 22.80 | 495 | 3 | 8 17.33 1.06 6 15.89 0.53 0 0 1 0 |
| 498 | 22.81 | 497 | 1 | 6 16.80 0.61 6 17.28 0.53 0 0 1 0 |
| 498 | 22.82 | 497 | 1 | 6 17.65 0.51 6 16.01 0.59 0 0 0 0 |
| 498 | 22.85 | 497 | 1 | 6 20.06 0.51 5 16.70 0.06 0 0 1 0 |
| 498 | 22.86 | 498 | 0 | 6 20.60 0.60 5 15.68 0.06 0 0 0 0 (feasible packing, ours 22.8231) |
| 544 | 23.794 | 541 | 3 | 6 18.43 0.45 6 17.45 0.51 0 0 1 0 |
| 544 | 23.80 | 541 | 3 | 8 18.49 0.66 8 17.37 1.07 0 0 0 1 |
| 544 | 23.81 | 543 | 1 | 6 17.29 0.54 6 17.44 0.51 0 1 1 1 |
| 544 | 23.82 | 543 | 1 | 6 18.98 0.50 6 17.19 0.57 0 0 0 1 |
| 544 | 23.84 | 543 | 1 | 6 19.59 0.50 5 19.42 0.50 0 1 1 0, bend ys2 13.6 |
| 544 | 23.85 | 544 | 0 | 6 18.77 0.56 5 19.35 0.42 0 0 1 0 (feasible packing, ours 23.8238) |
| 544 | 23.86 | 544 | 0 | 5 19.02 0.12 7 21.36 0.71 0 0 1 1, bend ys1 19.0 (feasible) |

The count barely depends on L within 0.03 (0.02 of f is 0.7 square of theoretical gain) and the search is noisy by
about 1 (run several seeds). The rigid plateau is n - 1 from the board side + 0.02 up to + 0.055, the full count only
at 22.86 / 23.85: the same 1 percent gap as 3jeonghun's test (296 of 299). The rigid family is also no better than our
current packings at the neighbouring counts (rigid 496 at 22.794 vs ours 22.7811, 497 at 22.81 vs 22.7982, 543 at
23.81 vs 23.8024), so the two-block structure only pays with the leader's bent and fanned ends.

Later additions that did not raise the rigid count: the leader's bend (`ys1`, `ys2`: the upper segment of a band is the
same lattice shifted one column towards the wall, its first row touching the last lower row along e2 with the along-wall
offset (1 - sin t) / cos t; exact, no overlap, but the slack column just changes side and the counts stay at n - 1 or
n - 2), ledge candidates (`--ledges`: upright candidates on the corners of the band end squares, off the wall grids),
finer upright grids, band lattice extensions, wider free zones at the ends, and the intermediate sides 22.815, 22.822,
23.815, 23.823. The plateau is n - 1 for 498 at 22.81 to 22.82 and 544 at 23.81 to 23.82, n - 2 for 498 at the board side.

Where the leader's extra density is (empty area by region, leader's 299 vs rigid 297 at 17.8133): 18.7 vs 20.6 in all,
spread over the band ends (left band bottom 3.3 vs 4.0, left band top 3.0 vs 4.1, corner 1.9 vs 2.7); nothing in the
band interiors. The leader's top block is fanned (angles 56 to 64 deg), the left block is a rigid lattice with a bend.

Waste accounting (L^2 - count): rigid 496 at 22.794 = 23.6, rigid 497 at 22.81 = 23.3, ours 498 at 22.8231 = 22.9,
board 22.7939 = 21.6; rigid 543 at 23.81 = 23.9, ours 544 at 23.8238 = 23.6, board = 22.2. So a rigid assembly short
by one or two squares is no denser than our current packings: the relaxer has to find real density (the fanned ends),
not just absorb the missing squares.

## Relaxation on GitHub (relax.yml, engines quench / quenchhop / shrink, seeds_x)

Seeds: the best rigid assembly per (n, L) filled to n squares by `tbfill.py` (largest-clearance spots, penetration
0.09 to 0.21; or `--rows`, the missing square inserted into the longest upright row with the overlap spread 1/m per
pair). The relax workflow hops only puzzles whose seed side is above the board (gap-weighted targets): a seed at or
below the board side (544 at 23.794 in round 1) is silently skipped, so seeds must sit above the board (or the
workflow needs an HOP_NLIST input).

| run | release | seeds (seeds_x, commit) | engines, minutes x seeds | certified 498 | certified 544 |
|---|---|---|---|---|---|
| 1 | relax-1 (run 42232291268) | 8 files, deficit 1 to 3, sides 22.794 to 22.82 / 23.794 to 23.81 (e234d10) | quench, quenchhop; 25 x 2 | 22.874869 (quenchhop, from tb498_794a) | not hopped (seed below board) |
| 2 | relax-2 (run 42236637415) | 6 files, deficit 1, 22.82 / 23.81, 23.82, fill and rows variants (99e812c) | quench, quenchhop; 30 x 2 | 22.867730 (from tb498_82r) | 23.868997 (from tb544_81s) |
| 3 | relax-3 (run 42239931916) | 7 files, round 2 plus tb498_81a (a9dd764) | shrink, quenchhop; 50 x 1 | 22.863394 (quenchhop; shrink gave about 22.894) | 23.846281 (shrink, from tb544_82a; quenchhop about 23.868) |
| 4 | relax-4 (run 42247207702) | 2 feasible full-count packings, 22.86 / 23.86 (cb5a45e) | shrink, quenchhop; 70 x 2 | 22.855715 (best of the day) | 23.841719 (best of the day) |
| 5 | relax-5 (run 42249859834) | 2 pre-fanned deficit-1 seeds, 22.81 / 23.81 (c1bc3fd) | quenchhop; 60 x 2 | 22.865194 | nothing certified |

Reference: ours 22.823086 / 23.823801, board 22.793863 / 23.794130. Nothing certified below ours: the best relaxed
sides are 22.8557 (498, release relax-4) and 23.8417 (544, relax-4), 0.033 and 0.018 above ours and 0.062 and 0.048
above the board. The relaxers do not merge these into the submission (the merge keeps our better sides).

## What happened in the relaxations

- Every quench / quenchhop run from an overlapping seed (deficit 1 or 2, penetration 0.1 to 0.2) ends 0.045 to 0.08
  above the seed side, i.e. at the side where the rigid construction holds n squares anyway (22.86 / 23.85). The hop
  at temp 1.0 disorders the band ends (the relaxed 498 keeps the two blocks but the top band's right end becomes a
  jumble of tilts) and never produces the leader's fanned ends.
- fq --shrink (engine shrink) keeps the structure best: from the 23.82 deficit-1 seed it ends at 23.8463, the inserted
  square settling as a small tilted cluster in the top-left corner; from the feasible 23.86 packing it reaches 23.8417
  (with 70 min of quenchhop in parallel giving 22.8557 for 498). Shrink from a feasible rigid packing is the most
  productive combination found, and it is cheap (minutes).
- Pre-fanning the band ends (4 deg progressive extra rotation) did not help (22.8652, and 544 not certified).
- The certifier rejects the rigid packings themselves as "overlap certified" (touching squares at exactly zero
  separation): a rigid seed meant to be certified directly needs a tiny gap (for example 1e-7) between touching squares.
- The relax workflow silently skips any seed whose side is at or below the board (544 at 23.794 in round 1).

## Conclusions

1. The leader's structure is understood and reproducible: two wall bands of brick lattices at the angle set by
   N cos t + sin t = N - 1 + f (atan(1/2) is just the solution for N = 6, f = 0.8133), upright fill everywhere else,
   a one-column slide in the middle of a band (exact, density-neutral) and fanned ends. `design/twoblock.py` builds
   it in 0.1 s and `tbsearch.py` tunes it.
2. The rigid version is worth n - 1 squares at the board side + 0.02 and n squares at + 0.065 (22.86 / 23.85). That is
   the same density as our current packings (waste 23.3 vs 22.9 for 498), so the structure alone gains nothing; the
   leader's last two squares (about 2 area units, 1 percent) come from the bent and fanned band ends, which fq quench,
   packer hop (temp 1.0, up to 70 min) and fq shrink did not find from our seeds. The rigid family is also no better
   than ours at the neighbouring counts (496, 497, 541 to 543).
3. What could still work (not tried today): an end-only optimiser that keeps the band interiors and the upright blocks
   fixed and searches the two end regions of each band with the squares free to rotate individually (a few dozen
   squares, so a direct continuous optimisation with the exact SAT penetration is feasible); or taking the leader's
   fanned end geometry from their 299 packing as a template (the fan spans 56 to 64 deg over the last rows) and
   transplanting it onto the rigid bands before the quench.

Scratch files (not committed): rigid assemblies, plots and release logs in the session scratchpad under tb/ and rel1..rel5/.



