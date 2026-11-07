#!/usr/bin/env bash
# Private repo helper (7 Nov 2026). Our code AND packings live in the PRIVATE repo <owner>/hopper-private (hop/, design/,
# seeds_x*/, colab/, pi/, seeds/*.csv.gz, results/*.csv.gz); this public repo keeps only the workflows, this script and
# hop/max_jobs.txt (free Actions minutes). Workflows reach the private repo with the repository secret DATA_TOKEN
# (fine-grained token, Contents read/write on hopper-private only), passed to the steps that call this script and to nothing else.
#   data.sh pull                      clone hopper-private (depth 1) into ./data, copy its code dirs and seeds/results files here
#   data.sh push MSG PATH...          copy PATH(s) (workspace-relative files or dirs) into ./data, commit, push
#   data.sh release TAG TITLE NOTES FILE...   create (or add to) release TAG in hopper-private with the given assets
#   data.sh dl TAG PATTERN DIR        download the assets matching PATTERN of release TAG of hopper-private into DIR
#   data.sh seal DIR                  DIR -> DIR.tar.enc (aes-256-cbc, pbkdf2, key derived from DATA_TOKEN); for artifacts
#   data.sh unseal PARENT             decrypt and extract every PARENT/*/*.tar.enc in place (after download-artifact)
set -euo pipefail
OWNER=${DATA_OWNER:-${GITHUB_REPOSITORY_OWNER:-othman-istaiteh}}
REPO=${DATA_REPO:-hopper-private}
ROOT=${GITHUB_WORKSPACE:-$(pwd)}
D=$ROOT/data
[ -n "${DATA_TOKEN:-}" ] || { echo "data.sh: DATA_TOKEN is not set (repository secret missing or not passed to this step)"; exit 1; }
URL=${DATA_URL:-"https://x-access-token:${DATA_TOKEN}@github.com/$OWNER/$REPO.git"}   # DATA_URL: local tests only
cmd=${1:-}; shift || true
case "$cmd" in
  pull)
    rm -rf "$D"
    for i in 1 2 3; do git clone -q --depth 1 "$URL" "$D" 2> clone.err && break; sleep $((i * 10)); done
    [ -d "$D/.git" ] || { echo "data.sh: clone of $OWNER/$REPO failed: $(sed 's#https://[^@]*@#https://#g' clone.err | head -c 300)"; exit 1; }
    rm -f clone.err
    # code: copied (not linked) so that this public hop/ (data.sh, max_jobs.txt) and the private hop/ merge into one directory
    for c in hop design colab pi $(cd "$D" && ls -d seeds_x* 2> /dev/null); do
      [ -d "$D/$c" ] || continue
      mkdir -p "$ROOT/$c" && cp -r "$D/$c/." "$ROOT/$c/"
    done
    mkdir -p "$ROOT/seeds" "$ROOT/results"
    cp "$D"/seeds/*.csv.gz "$ROOT/seeds/" 2> /dev/null || true
    cp "$D"/results/*.csv.gz "$ROOT/results/" 2> /dev/null || true
    echo "data: $(ls "$ROOT"/seeds/*.csv.gz 2> /dev/null | wc -l) seed files, $(ls "$ROOT"/results/*.csv.gz 2> /dev/null | wc -l) result files, $(ls "$ROOT"/hop/*.py 2> /dev/null | wc -l) hop scripts from $OWNER/$REPO@$(git -C "$D" rev-parse --short HEAD)" ;;
  push)
    MSG=$1; shift
    [ -d "$D/.git" ] || { echo "data.sh: run 'data.sh pull' first"; exit 1; }
    for f in "$@"; do
      f=${f%/}; mkdir -p "$D/$(dirname "$f")"
      if [ -d "$ROOT/$f" ]; then mkdir -p "$D/$f" && cp -r "$ROOT/$f/." "$D/$f/"; else cp "$ROOT/$f" "$D/$f"; fi
    done
    cd "$D"
    git config user.name "othman-istaiteh" && git config user.email "othmanistaiteh@gmail.com"
    for f in "$@"; do git add -A -- "${f%/}"; done
    git commit -q -m "$MSG" || { echo "data: nothing to commit"; exit 0; }
    B=$(git rev-parse --abbrev-ref HEAD)
    for i in 1 2 3 4 5; do git pull -q --rebase origin "$B" && git push -q origin "HEAD:$B" && { echo "data: pushed $* ($MSG)"; exit 0; }; sleep $((i * 15)); done
    echo "data.sh: push failed"; exit 1 ;;
  release)
    TAG=$1 TITLE=$2 NOTES=$3; shift 3
    export GH_TOKEN=$DATA_TOKEN
    for i in 1 2 3 4 5; do
      if gh release view "$TAG" -R "$OWNER/$REPO" > /dev/null 2>&1; then gh release upload "$TAG" "$@" -R "$OWNER/$REPO" --clobber && { echo "data: assets added to $REPO release $TAG"; exit 0; }
      else gh release create "$TAG" "$@" -R "$OWNER/$REPO" --title "$TITLE" --notes "$NOTES" && { echo "data: $REPO release $TAG created"; exit 0; }; fi
      sleep $((i * 15))
    done
    echo "data.sh: release $TAG failed"; exit 1 ;;
  dl)
    TAG=$1 PAT=$2 DIR=$3
    export GH_TOKEN=$DATA_TOKEN
    mkdir -p "$DIR"
    gh release download "$TAG" -R "$OWNER/$REPO" -p "$PAT" -D "$DIR" --clobber && echo "data: release $TAG fetched" || echo "data: release $TAG has no $PAT" ;;
  seal)
    DIR=$1
    tar -C "$DIR" -czf - . | openssl enc -aes-256-cbc -pbkdf2 -iter 100000 -salt -pass env:DATA_TOKEN -out "$DIR.tar.enc"
    echo "data: sealed $DIR -> $DIR.tar.enc ($(stat -c %s "$DIR.tar.enc") bytes)" ;;
  unseal)
    P=$1; k=0
    for e in "$P"/*/*.tar.enc; do
      [ -f "$e" ] || continue
      openssl enc -d -aes-256-cbc -pbkdf2 -iter 100000 -pass env:DATA_TOKEN -in "$e" | tar -C "$(dirname "$e")" -xzf - && rm -f "$e" && k=$((k + 1))
    done
    echo "data: unsealed $k artifact(s) under $P" ;;
  *)
    echo "usage: data.sh pull | push MSG FILE... | release TAG TITLE NOTES FILE... | dl TAG PATTERN DIR | seal DIR | unseal PARENT"; exit 2 ;;
esac
