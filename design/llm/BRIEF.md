# LLM program search, brief for program authors (round 1, 2 Nov 2026)

Problem: pack n unit squares (any rotation) in the smallest axis-aligned square. Score per puzzle = side/sqrt(n) - 1.
We are Kaggle team "Mais Oui", rank ~90; the leaders hold better *designs* in the middle of each band.

## Target family (the "0.794 family")
Write K = ceil(sqrt n) - 1 ... in our notes K = ceil(sqrt n), c = n - (K-1)^2, side = (K-1) + f.
Panel for round 1 (our side / board side):
- 299: ours 17.80124 (two tilted bands at 27 and 63 deg, 104 tilted squares, f 0.8037 gadget) / board 17.79364
- 412: ours 20.80370 (same 0.8037 gadget) / board 20.79413
- 498: ours 22.82309 (one two-row diagonal strip at ~26 deg absolute, blocks wall-flush) / board 22.79386
- 544: ours 23.82380 (same strip design, f 0.8238) / board 23.79413
Board f(c) at fixed K is a smooth increasing curve (~0.015 per extra square); ours is a staircase of fixed gadgets
0.005-0.03 above it. So the leaders' construction has a continuous parameter (tilt / band length / row phase) tuned per c.
Waste accounting of our 544 (area 23.57 vs board 22.16): 94 % of the middle waste is at the two strip-block interfaces
(staircase notches the tilted squares only partly fill); per-row waste repeats with period 4 (0.65/0.75/0.79/0.73);
corners 3.46. Record packings elsewhere (414, 457) are asymmetric: mixed tilts, arms short of walls, ~50 upright
squares off the wall lattices.

## What already FAILED (do not repeat as-is)
- Re-tilting our strip squares to commensurate angles (19.47, 24.6, 27.89, 31.95 deg) or phase-shifting the strip,
  then relaxing: always relaxes back to our own side.
- Rigid constructions (Goebel strips, two-bulb / V-arc, two wall channels of tan~2 brick lattice like the 176 record,
  width-2/3 diagonal strips) built from scratch: after relaxation 0.07-0.17 above our side. Rigid counts are ~10-22
  squares short; relaxation sheds tilted squares instead of bending them.
- Single surgery moves (kick, band flip, row insert/remove, retilt, shuffle) + short quench: back to the start basin or worse.
- Adding a crude third row to the strip: does not relax.

## Interface
A program is one Python file defining
    def build(n, base, board, rng) -> np.ndarray of shape (n, 3): x, y, theta (RADIANS) = centres + rotation of unit squares
- base: our certified best packing for n, same format (a feasible packing, side = ours). You may use it, edit it, or ignore it.
- board: the board side for n (float). rng: numpy Generator (use it for any randomness; different seeds are evaluated).
- Only numpy / scipy / stdlib. Must run in < 60 s per call. Deterministic given rng.
- Overlaps are allowed: the evaluator relaxes your output with fq quench (augmented Lagrangian, side free, loosen 1.03,
  3 min) then packer hop --temp 1.0 (Metropolis basin hopping, 3 min), then re-quench, and reports the best feasible
  side. Fitness = mean over the panel of (relaxed side - our side); negative = a real win, which gets certified and
  submitted.
- Practical rules learned: keep the bounding side BELOW the grid side ceil(sqrt n) (above it the relaxation collapses to
  the grid); keep overlaps modest (check.py depth_sum of a few units at most, 0 is best); keep exactly n squares.

## Tools here
- design/llm/check.py prog.py [n ...]: count, bounding side, overlapping pairs and SAT depth sum, ours and board.
- design/llm/base.csv: our best packings for n >= 250 (competition CSV: n, square_id, x, y, angle in radians).
- design/llm/r1/p00_identity.py: the calibration program (returns base).
You cannot run fq / packer locally (no Rust build allowed); judge your seeds with check.py and reasoning.

## Round 1 results (llm-r1-2; fitness = mean relaxed side - ours over 299/412/498/544; per-n values)
- p00_identity +0.00002 (calibration: quench + 3 min hop from our own packing finds nothing)
- pA1_bent_strip +0.00002 (relaxes straight back to our packing), pA2_notch_retile +0.0006 (same)
- pC1_neighbour +0.027 (0.022 0.026 0.022 0.038), pA3_corner_to_notch +0.066
- pC2_scale +0.14, pC3_crossover +0.18
- pB1/pB2/pB3 (constructions at board + 0.015 WITH overlaps from inserted deficit squares): +0.19 = exactly the GRID side.
  fq quench (loosen 1.03, side free) collapsed every one of them to full grid lines in seconds.
Lessons: (1) small edits of our packing are pulled back to our lock; (2) an overlapping construction collapses to the
grid: the quench resolves overlaps by opening full lines. Earlier evidence (176 calibration): a FEASIBLE seed (zero
overlap) at a side above ours keeps its topology under packer hop --temp 1.0, which then compresses it slowly (176:
14.136 -> 13.87 in 9 min; board 13.796).

## Round 2 rules (new)
- Evaluator now: fq quench 3 min, then packer hop --temp 1.0 for 12 min, re-quench. Hop needs a FEASIBLE state.
- Construction programs must return a FEASIBLE packing: depth_sum = 0 in check.py (no overlap at all), exactly n squares,
  side below ceil(sqrt n) - 0.02 (can be well above ours, e.g. ours + 0.05 to 0.15). Do NOT insert deficit squares with
  overlaps; instead raise the construction side (or the parameter) until the exact count fits feasibly, then return the
  smallest feasible side you find. The relaxer then only compresses.
- Edit programs (on base) must make a change large enough to leave our basin (round 1 edits were undone), but stay
  feasible: e.g. remove a whole region (a band or interface strip), re-build it feasibly at a slightly larger side by a
  different rule, keep the rest.
