#!/usr/bin/env bash
# One-time setup of a Raspberry Pi (64-bit OS, aarch64) as a background hop worker.
#   curl -sSL https://raw.githubusercontent.com/othman-istaiteh/hopper-ci/main/pi/pi_setup.sh | bash
# Installs Rust, builds Evan Daniel's fq and packer (MIT) at the pinned commit, clones this repository to ~/hopper-ci
# and installs the systemd service pi_hop (see pi_loop.sh). Afterwards: store a GitHub token once (see pi_loop.sh)
# and `sudo systemctl start pi_hop`.
set -euo pipefail
EVAND_REF=0e76d9fe5e48fc2a614e3c132a0a3fd67e0a6a8d
if [ "$(uname -m)" != "aarch64" ]; then echo "needs a 64-bit OS (uname -m says $(uname -m))"; exit 1; fi
sudo apt-get update -qq && sudo apt-get install -y -qq build-essential git curl python3 python3-numpy python3-pip
pip3 install -q --break-system-packages mpmath 2>/dev/null || pip3 install -q mpmath
if [ ! -x "$HOME/.cargo/bin/cargo" ]; then curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal; fi
export PATH="$HOME/.cargo/bin:$PATH"
if [ ! -d "$HOME/evand" ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/evand/square-packing.git "$HOME/evand"
  cd "$HOME/evand" && git sparse-checkout set --no-cone /search/packer/src/ /search/packer/Cargo.toml /search/packer/Cargo.lock
  git fetch -q --depth 1 --filter=blob:none origin "$EVAND_REF" && git checkout -q FETCH_HEAD
fi
cd "$HOME/evand/search/packer" && cargo build --release --bin fq --bin packer 2>&1 | tail -2
ls -la target/release/fq target/release/packer
if [ ! -d "$HOME/hopper-ci" ]; then git clone -q https://github.com/othman-istaiteh/hopper-ci.git "$HOME/hopper-ci"; fi
cd "$HOME/hopper-ci" && git config credential.helper store && git config user.name "pi worker" && git config user.email "pi@users.noreply.github.com"
chmod +x pi/pi_loop.sh
sudo tee /etc/systemd/system/pi_hop.service > /dev/null <<EOF
[Unit]
Description=hop worker
After=network-online.target

[Service]
User=$USER
WorkingDirectory=$HOME/hopper-ci
ExecStart=$HOME/hopper-ci/pi/pi_loop.sh
Restart=always
RestartSec=60
Nice=10

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable pi_hop
echo "setup done. Store a token once:  cd ~/hopper-ci && git push   (username: othman-istaiteh, password: the fine-grained token with contents read/write)"
echo "then:  sudo systemctl start pi_hop   and   journalctl -u pi_hop -f"
