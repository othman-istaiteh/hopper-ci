#!/usr/bin/env bash
# Endless hop loop for the Raspberry Pi worker (run by the pi_hop systemd service). Each round: pull the repository,
# unpack the seeds and results, hop the responsive puzzles (hop/soft_list.txt) for PI_HOURS, then commit the certified
# output as results/pi_latest.csv.gz so that every GitHub merge picks it up. Needs the stored git credentials (see
# pi_setup.sh). Tunables through the environment: PI_HOURS (default 4), PI_MIN (minutes per hop, 12), PI_SEEDS (2),
# PI_ENGINE (hop or shrink), PI_NLIST (comma list; default = hop/soft_list.txt).
set -uo pipefail
cd "$HOME/hopper-ci"
export PATH="$HOME/.cargo/bin:$PATH"
while true; do
  git pull -q --rebase origin main || { echo "pull failed, retrying in 10 min"; sleep 600; continue; }
  rm -rf seeds_un && mkdir -p seeds_un
  for f in seeds/*.csv.gz results/*.csv.gz; do n=$(basename "$f" .csv.gz); mkdir -p "seeds_un/$n"; gunzip -c "$f" > "seeds_un/$n/submission.csv"; done
  NLIST="${PI_NLIST:-$(cat hop/soft_list.txt)}"
  T=$(echo "$NLIST" | tr ',' '\n' | grep -c .)
  SEED0=$(( ($(date +%s) / 60) % 100000 + 100000 ))
  echo "$(date) round: $T targets, seed0 $SEED0, ${PI_HOURS:-4} h on $(nproc) cores"
  rm -rf "$HOME/pi_work" "$HOME/pi_out" && mkdir -p "$HOME/pi_out"
  ( cd hop && FQ_BIN="$HOME/evand/search/packer/target/release/fq" PACKER_BIN="$HOME/evand/search/packer/target/release/packer" \
    SQ_IN="$HOME/hopper-ci/seeds_un" SQ_WORK="$HOME/pi_work" SQ_BOARD=board_29oct.txt SQ_WORKERS=$(nproc) SQ_MAXC=1 SQ_THREADS=1 \
    HOP_MIN="${PI_MIN:-12}" HOP_SEEDS="${PI_SEEDS:-2}" HOP_SEED0="$SEED0" HOP_NTARGET="$T" HOP_SKIP=0 HOP_NLIST="$NLIST" \
    HOP_BUDGET_H="${PI_HOURS:-4}" HOP_ENGINE="${PI_ENGINE:-hop}" FQ_DEADLINE=120 \
    python3 -u sq10_hop.py "$HOME/pi_out/submission.csv" | grep -v "^SIDES" | tail -5 )
  if [ -s "$HOME/pi_out/submission.csv" ]; then
    # commit only the improved puzzles (small file); merges take the best per puzzle anyway
    python3 hop/improved_only.py seeds_un "$HOME/pi_out/submission.csv" "$HOME/pi_out/improved.csv"
    if [ "$(wc -l < "$HOME/pi_out/improved.csv")" -gt 1 ]; then
      gzip -9 -c "$HOME/pi_out/improved.csv" > "results/pi_$(date +%m%d_%H%M).csv.gz"
      git add results/ && git commit -q -m "results: pi worker round $(date +%F_%H%M)" || true
    fi
    for i in 1 2 3 4 5; do git pull -q --rebase origin main && git push -q origin HEAD:main && break; sleep $((i * 30)); done
  fi
  sleep 60
done
