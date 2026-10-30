# squares-hop

Compute pipeline of team **Mais Oui** for the Kaggle competition *Squares in Squares* (pack n unit squares into the
smallest square, n = 1..1000). The repository exists so that the search can run on GitHub Actions; it is not a library.

What runs here:

- `hop/sq10_hop.py` hops our current packings with Evan Daniel's `packer hop` and `fq` (MIT, from
  [evand/square-packing](https://github.com/evand/square-packing)), re-quenches the results and keeps per puzzle the
  first candidate that `hop/sq_metric.py` (a re-implementation of the competition metric's checks) certifies.
- `hop/sq09_requench.py` is the certified merge used by every workflow.
- `.github/workflows/hop.yml` runs 12 to 20 slices in parallel, merges them, publishes the merge as a release asset and
  can chain the next run from the merged seeds.
- `polish.yml`, `explore.yml`, `shtest.yml` are one-off experiments (deep polish, basin explorer, engine A/B test).

Seeds and releases are our own competition submissions, built from the public Kaggle notebooks (credited in the
notebook `notebook5586544269` and in the hop scripts: evgendvorkin, avazaku, busyaprime, malyshevdanil, nilanjansaha123,
alexchilton) plus our own search. They are published here only because the workflow needs a place to pass results
between runs; please take them in that spirit and credit the original finders if you use them.

Own code in this repository: MIT. Evan Daniel's engines are fetched at build time from his repository under his licence.
