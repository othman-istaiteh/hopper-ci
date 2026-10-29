# squares-hop

Basin-hopping campaign for the Kaggle "Squares in Squares" competition (team Mais Oui), run as a GitHub Actions
workflow (`.github/workflows/hop.yml`, manual dispatch): N parallel jobs each hop a slice of the gap-weighted target
list from our current certified packings (`seeds/sq09.csv.gz`), re-quench every hop output, certify with a copy of
the competition metric, and a final job merges the slices and publishes the result as a release asset.

Engines: `packer hop` (basin hopping with a descending target side) and `fq` (augmented-Lagrangian quench + LP polish)
by Evan Daniel, MIT licence, [evand/square-packing](https://github.com/evand/square-packing) (search/packer), built
from source in every job. Own code in `hop/`: `sq10_hop.py` (target selection, hop jobs, re-quench, certified merge),
`sq09_requench.py` (certified merge, quench schedules), `sq_metric.py` (re-implementation of the official metric
notebook, Apache 2.0), `board_29oct.txt` (per-puzzle leaderboard sides, 29 Oct 2026).

Seeds: the merge of public Kaggle notebook outputs (credited in our notebooks: evgendvorkin, avazaku, busyaprime,
malyshevdanil, nilanjansaha123, alexchilton) plus our own hop results.
