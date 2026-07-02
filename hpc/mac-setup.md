# Mac-side setup — one-command-per-day multi-cluster access

The personal Mac is the single place code is written; every HPC cluster in
Phase 9 is reached from here over SSH. This doc is the **canonical, reproducible
record** of the Mac-side configuration: the SSH client config, the
`morning-login` helper, and the ssh-agent / GitHub-key setup.

Nothing here lives in the repo except this file. The actual `~/.ssh/config`,
`~/.ssh/controlmasters/`, and `~/bin/morning-login` are user-local and are
recreated from the templates below.

The design goal: **authenticate once per cluster per day**, then every `ssh`,
`scp`, `rsync`, and `git` operation for the next 24 hours reuses that
connection with no further MFA/DUO prompts. This is what makes
`sync-all-clusters.sh` and the per-cluster Claude skills usable without a
password prompt on every command.

---

## How it works

Two SSH features do the heavy lifting:

- **`ControlMaster` + `ControlPersist 24h`** — the first `ssh <cluster>`
  authenticates and opens a *master* connection whose control socket lives in
  `~/.ssh/controlmasters/`. Every later `ssh <cluster>` (from any process on
  the Mac — including Claude's Bash tool) rides that socket instead of
  re-authenticating. The master lingers 24 h after the last client detaches,
  so one morning login covers a full workday.
- **`ForwardAgent yes`** — the Mac's `ssh-agent` (holding your GitHub key) is
  exposed on the remote host, so `git pull` / `git clone` on a cluster
  authenticates to GitHub with the Mac's key. No per-cluster GitHub key needed.

---

## One-time setup

```bash
# 1. Control-socket directory (private).
mkdir -p ~/.ssh/controlmasters && chmod 700 ~/.ssh/controlmasters

# 2. ~/bin for the morning-login helper.
mkdir -p ~/bin

# 3. (Optional) put ~/bin on PATH so you can type `morning-login` directly.
#    Login shell here is bash, so this goes in ~/.bash_profile (or ~/.profile):
#      export PATH="$HOME/bin:$PATH"
#    Until then, invoke it by full path: ~/bin/morning-login

# 4. Load the GitHub key into the agent, persisted in the macOS Keychain
#    (survives reboots — only needed once):
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
ssh-add -l                       # should now list your key
```

---

## SSH config (`~/.ssh/config`)

The Phase 9 workflow adds **six short-name aliases** — `delta`, `deltaai`,
`stampede3`, `derecho`, `midway3`, `dsi` — each carrying the ControlMaster
persistence settings. These aliases are what `morning-login` and
`sync-all-clusters.sh` target.

> **Placement note.** These blocks coexist with any existing full-hostname or
> compute-node entries in your `~/.ssh/config`. Put them **above** the trailing
> `Host *` catch-all so their per-host options win (SSH is first-match-wins per
> option). The short aliases don't collide with full-hostname entries, so both
> forms keep working.

```sshconfig
# ═══ Phase 9 cluster aliases (morning-login / sync-all-clusters.sh) ═══════════
# Short-name aliases with ControlMaster persistence. `ssh <alias>` reuses one
# authenticated connection for 24 h, so MFA/DUO prompts fire once per day.

Host delta
  HostName login.delta.ncsa.illinois.edu
  User awikner
  ForwardAgent yes
  ControlMaster auto
  ControlPath ~/.ssh/controlmasters/%r@%h:%p
  ControlPersist 24h
  ServerAliveInterval 60
  ServerAliveCountMax 3

Host deltaai
  HostName dtai-login.delta.ncsa.illinois.edu
  # NB: login.deltaai.ncsa.illinois.edu has no DNS record; dtai-login is the real login node.
  User awikner
  ForwardAgent yes
  ControlMaster auto
  ControlPath ~/.ssh/controlmasters/%r@%h:%p
  ControlPersist 24h
  ServerAliveInterval 60
  ServerAliveCountMax 3

Host stampede3
  HostName stampede3.tacc.utexas.edu
  User awikner
  ForwardAgent yes
  ControlMaster auto
  ControlPath ~/.ssh/controlmasters/%r@%h:%p
  ControlPersist 24h
  ServerAliveInterval 60
  ServerAliveCountMax 3

Host derecho
  HostName derecho.hpc.ucar.edu
  User awikner
  ForwardAgent yes
  ControlMaster auto
  ControlPath ~/.ssh/controlmasters/%r@%h:%p
  ControlPersist 24h
  ServerAliveInterval 60
  ServerAliveCountMax 3

Host midway3
  HostName midway3.rcc.uchicago.edu
  User awikner
  ForwardAgent yes
  ControlMaster auto
  ControlPath ~/.ssh/controlmasters/%r@%h:%p
  ControlPersist 24h
  ServerAliveInterval 60
  ServerAliveCountMax 3

Host dsi
  HostName login.ds.uchicago.edu
  # fe01/fe02 direct SSH retired 2026-06-08; login.ds.uchicago.edu is the required load balancer.
  User awikner
  ForwardAgent yes
  ControlMaster auto
  ControlPath ~/.ssh/controlmasters/%r@%h:%p
  ControlPersist 24h
  ServerAliveInterval 60
  ServerAliveCountMax 3
```

### Per-cluster auth mechanism

| Alias | HostName | First-of-day auth |
|---|---|---|
| `delta` | `login.delta.ncsa.illinois.edu` | NCSA identity + 2FA (Duo) |
| `deltaai` | `dtai-login.delta.ncsa.illinois.edu` | NCSA identity + 2FA (Duo) |
| `stampede3` | `stampede3.tacc.utexas.edu` | TACC: enter `password,TOTP` at the single password prompt (e.g. `mypass,123456`) |
| `derecho` | `derecho.hpc.ucar.edu` | NCAR identity + Duo |
| `midway3` | `midway3.rcc.uchicago.edu` | UChicago CNet + Duo |
| `dsi` | `login.ds.uchicago.edu` | SSH key (first-ever login uses CNet password) |

---

## `morning-login` script

Establishes the ControlMaster connections for the day. Run it once each
morning; complete each cluster's MFA/DUO as it prompts. After that, all SSH /
SCP / rsync / git-over-SSH traffic to those clusters skips re-authentication
for 24 h.

Save as `~/bin/morning-login`, `chmod +x`:

```bash
#!/usr/bin/env bash
# morning-login — establish ControlMaster connections to all HPC clusters.
# Run once per day. You will be prompted for MFA/DUO per cluster that requires
# it; after that all SSH/SCP/rsync commands skip re-authentication for 24 h.
#
# Usage:  morning-login [cluster …]
#   No args   → connects to all clusters.
#   With args → connects only to the named clusters (e.g. `morning-login deltaai midway3`).

set -euo pipefail
CLUSTERS=(delta deltaai stampede3 derecho midway3 dsi)
targets=("${@:-${CLUSTERS[@]}}")

mkdir -p ~/.ssh/controlmasters

for c in "${targets[@]}"; do
    if ssh -O check "$c" &>/dev/null; then
        echo "[✓] $c — already connected"
    else
        printf "[…] %s — connecting\n" "$c"
        # -fNM: background after auth (-f), no remote command (-N), master (-M).
        # For MFA/DUO clusters the prompt fires before -f backgrounds the process.
        if ssh -fNM "$c"; then
            echo "    → OK"
        else
            echo "    → FAILED (check hostname / credentials)"
        fi
    fi
done
echo "Done."
```

Install (paste the script above into the file, then make it executable):

```bash
$EDITOR ~/bin/morning-login       # paste the text above, save
chmod +x ~/bin/morning-login
```

Typical morning:

```bash
~/bin/morning-login            # all six; complete each MFA as prompted
# or just the ones you need today:
~/bin/morning-login deltaai midway3
```

---

## GitHub key forwarding

`ForwardAgent yes` makes the Mac's `ssh-agent` available on each cluster, so
`git` operations on a cluster use the Mac's GitHub key — no per-cluster key.

Confirm the key is loaded on the Mac:

```bash
ssh-add -l                        # should show your key
ssh-add --apple-use-keychain ~/.ssh/id_ed25519   # add if missing
```

Test from a cluster (run **on the remote host**, after logging in):

```bash
ssh -T git@github.com             # should say: Hi awikner! You've successfully authenticated…
```

---

## Managing ControlMaster connections

```bash
ssh -O check <alias>     # is the master live?  → "Master running (pid=…)"
ssh -O exit  <alias>     # close the master socket early
ls ~/.ssh/controlmasters # one socket file per live connection
```

If a cluster starts refusing new sessions or a socket goes stale (e.g. after a
laptop sleep / network change), `ssh -O exit <alias>` then re-run
`morning-login <alias>`.

---

## Security note on `ForwardAgent`

Agent forwarding exposes your loaded keys to the remote host's `ssh-agent`
socket while you're connected. That is acceptable for **trusted HPC login
nodes** (the whole point here is convenient GitHub access). **Never** enable
`ForwardAgent` for untrusted or shared jump hosts — a root user on the remote
can use the forwarded agent to authenticate as you elsewhere. The six clusters
above are trusted; do not extend `ForwardAgent yes` to hosts you don't control
or trust.
