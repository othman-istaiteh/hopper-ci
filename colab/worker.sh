#!/usr/bin/env bash
# Colab worker (TPU v5e-1 runtime = 24 CPU threads): build fq/packer, hop the given puzzles at --temp 1.0 in rounds, and
# after each round upload the improved puzzles to the private Kaggle dataset <user>/sq-colab-hop (merge_now fetches it).
# Needs KAGGLE_API_TOKEN in the environment (Colab secret), never printed.
# Usage: bash worker.sh "n1,n2,..." ROUNDS MINUTES_PER_HOP SEEDS_PER_ROUND
set -uo pipefail
NLIST="$1"; ROUNDS="${2:-3}"; HMIN="${3:-15}"; SEEDS="${4:-3}"; KUSER="${KUSER:-istatieh}"
cd /content
export PATH="$HOME/.cargo/bin:$PATH"
if [ ! -x /content/evand/search/packer/target/release/fq ]; then
  echo "building fq and packer"
  command -v cargo >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q
  export PATH="$HOME/.cargo/bin:$PATH"
  rm -rf evand && git clone -q --depth 1 --filter=blob:none --sparse https://github.com/evand/square-packing.git evand
  (cd evand && git sparse-checkout set --no-cone /search/packer/src/ /search/packer/Cargo.toml /search/packer/Cargo.lock \
     && cd search/packer && cargo build -q --release --bin fq --bin packer) || { echo "build failed"; exit 1; }
fi
pip -q install mpmath kaggle >/dev/null 2>&1
rm -rf hopper-ci seeds_un && git clone -q --depth 1 https://github.com/othman-istaiteh/hopper-ci.git
mkdir -p seeds_un up
for f in hopper-ci/seeds/*.csv.gz hopper-ci/results/*.csv.gz; do n=$(basename "$f" .csv.gz); mkdir -p "seeds_un/$n"; gunzip -c "$f" > "seeds_un/$n/submission.csv"; done
T=$(echo "$NLIST" | tr ',' '\n' | grep -c .)
W=$(nproc)
echo "$(date) targets $T, rounds $ROUNDS, $SEEDS seeds x $HMIN min per round on $W threads"
for r in $(seq 1 "$ROUNDS"); do
  S0=$(( ($(date +%s) / 60) % 90000 + 10000 ))
  rm -rf /content/work /content/out && mkdir -p /content/out
  ( cd hopper-ci/hop && FQ_BIN=/content/evand/search/packer/target/release/fq PACKER_BIN=/content/evand/search/packer/target/release/packer \
    SQ_IN=/content/seeds_un SQ_WORK=/content/work SQ_BOARD=board_29oct.txt SQ_WORKERS=$W SQ_MAXC=1 SQ_THREADS=1 \
    HOP_MIN=$HMIN HOP_SEEDS=$SEEDS HOP_SEED0=$S0 HOP_NTARGET=$T HOP_SKIP=0 HOP_NLIST="$NLIST" HOP_BUDGET_H=3 \
    HOP_ARGS="--temp 1.0" HOP_ENGINE=hop FQ_DEADLINE=120 \
    python3 -u sq10_hop.py /content/out/submission.csv 2>&1 | grep -v "^SIDES" | grep -E "hop jobs|hop stage done|certified|IMPROVED|search improved" )
  if [ -s /content/out/submission.csv ]; then
    ( cd hopper-ci/hop && python3 improved_only.py /content/seeds_un /content/out/submission.csv /content/up/colab_r${r}_$(date +%d%H%M).csv )
    # keep the improvements as seeds for the next round
    for g in /content/up/colab_*.csv; do n=$(basename "$g" .csv); mkdir -p "seeds_un/$n"; cp "$g" "seeds_un/$n/submission.csv"; done
    printf '{"title": "sq-colab-hop", "id": "%s/sq-colab-hop", "licenses": [{"name": "CC0-1.0"}]}' "$KUSER" > /content/up/dataset-metadata.json
    if [ "$(ls /content/up/colab_*.csv 2>/dev/null | wc -l)" -gt 0 ]; then
      kaggle datasets version -p /content/up -m "colab round $r" -q 2>&1 | tail -1 || true
      kaggle datasets status "$KUSER/sq-colab-hop" >/dev/null 2>&1 || kaggle datasets create -p /content/up -q 2>&1 | tail -1
    fi
  fi
  echo "$(date) round $r done"
done
