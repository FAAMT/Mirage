#!/usr/bin/env bash
# =====================================================================
# scrub_check.sh -- refuse to publish identifying or secret material.
#
#   ./tools/scrub_check.sh              # check tracked + staged files
#   ./tools/scrub_check.sh --all        # also check untracked files
#   ./tools/scrub_check.sh --history    # check every blob and message in
#                                       # history (default HEAD, or pass a ref)
#
# Exit 0 = clean, 1 = findings. Run it before every push.
#
# --history exists because the other two modes cannot answer the question
# that actually matters before a FIRST push. They read the working tree,
# which is the right scope for "am I about to commit a secret" and the
# wrong one for "is a secret already in here". Untracking a file cleans
# the tree and leaves every blob exactly where it was.
#
# That gap was not hypothetical: vivado.log named the build machine in
# three commits and reached the remote while this script reported clean,
# correctly, because the file was gone from the tree by then. Purging it
# needed a history rewrite. Run --history before publishing a branch, when
# the fix is still a rebase and not a renumbering.
#
# WHY THIS EXISTS
#
# It replaces me grepping by hand, which failed twice on the same package:
#
#   1. The pre-push scan was piped through `head -40`, so it TRUNCATED and
#      I reported the package clean. 260 occurrences of a real /home/<user> path
#      were sitting in vivado.log below the cut.
#   2. The patterns were wrong for the data. I searched for the word
#      "hostname" and for *.local. Vivado writes the machine as
#      `# Running On: <ASSET-TAG>` and `| Host : <ASSET-TAG>`, so a real
#      corporate hostname went to the remote in the first push.
#
# Both failures share a cause: an ad-hoc scan whose coverage lives in
# whoever typed it. This script has no `head`, no `-m`, and prints counts
# per pattern so silence is evidence rather than an artefact of paging.
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

ALL=0
HISTORY=0
REF="HEAD"
case "${1:-}" in
    --all)     ALL=1 ;;
    --history) HISTORY=1; REF="${2:-HEAD}" ;;
esac

# PATHS is what the filename rules look at: real repo paths in tree mode,
# historical paths in history mode. FILES is what the content rules read.
if [[ $HISTORY -eq 1 ]]; then
    TMPD=$(mktemp -d)
    trap 'rm -rf "$TMPD"' EXIT

    # Every blob ever reachable from REF, one file per blob. Named after the
    # path it had, so a finding says which file it came from rather than just
    # a sha. Deduplicated by sha: unchanged files across commits are one blob.
    PATHS=$(git rev-list --objects "$REF" \
            | git cat-file --batch-check='%(objecttype) %(objectname) %(rest)' \
            | awk '$1=="blob" && $3 != "" {print $2" "$3}' \
            | while read -r sha path; do
                  [[ "$path" == "tools/scrub_check.sh" ]] && continue
                  out="$TMPD/$(printf '%s' "$path" | tr '/' '%')@${sha:0:8}"
                  [[ -e "$out" ]] || git cat-file blob "$sha" > "$out" 2>/dev/null
                  printf '%s\n' "$path"
              done | sort -u)

    # Commit messages and identities go to the remote too, and no amount of
    # blob scanning sees them. An author email is the classic leak here.
    git log --format='%H%n%an <%ae>%n%cn <%ce>%n%B' "$REF" \
        > "$TMPD/COMMIT-METADATA@log" 2>/dev/null

    mapfile -d '' FILES < <(find "$TMPD" -type f -print0)
    echo "scrub_check: history mode, ref $REF"
elif [[ $ALL -eq 1 ]]; then
    mapfile -d '' FILES < <(git ls-files -z --cached --others --exclude-standard)
    PATHS=$(printf '%s\n' "${FILES[@]}")
else
    mapfile -d '' FILES < <(git ls-files -z --cached --others --exclude-standard \
                            --deduplicate 2>/dev/null || git ls-files -z)
    PATHS=$(printf '%s\n' "${FILES[@]}")
fi

# This script documents the exact patterns it hunts for, so it matches
# itself. Exclude it, or it can never report clean.
#
# That exclusion is a hole, and it bit once: an early version quoted a REAL
# hostname as an example in the comment above, so the identifier was pushed
# inside the very file meant to stop it -- unscannable by construction.
# Examples here must be placeholders like <ASSET-TAG>, never real values.
# The self-check below enforces that much.
FILTERED=()
for f in "${FILES[@]}"; do
    [[ "$f" == "tools/scrub_check.sh" ]] && continue
    FILTERED+=("$f")
done
FILES=("${FILTERED[@]}")

# Self-check: the excluded file must not contain a literal identifier.
# Placeholders are angle-bracketed; anything else shaped like an asset tag
# or a home path in this script is a mistake.
self_bad=$(grep -nE '/home/[a-z]|\b[A-Z]{3,6}-[0-9]{3,6}\b' "$0" \
           | grep -vE '<[A-Z-]+>|/home/REDACTED|/home/<user>|VHDL-|FROZEN-|ISO-|UTF-' || true)
if [[ -n "$self_bad" ]]; then
    echo "FAIL  scrub_check.sh contains a literal identifier in its own text"
    printf '%s\n' "$self_bad" | sed 's/^/      /'
    echo
    echo "SCRUB CHECK FAIL -- fix the example to a placeholder."
    exit 1
fi

echo "scrub_check: ${#FILES[@]} files"
echo

findings=0

# --- pattern rules -------------------------------------------------------
# name | egrep pattern | allowed-exception pattern (may be empty)
# 4th arg: extra grep flags. Pass -i for anything whose real-world spelling
# varies in case -- API_KEY vs api_key slipped past a case-sensitive pattern
# when this script was first probed against a planted secret.
run_rule() {
    local name="$1" pat="$2" allow="${3:-}" flags="${4:-}"
    local hits
    if [[ -n "$allow" ]]; then
        hits=$(printf '%s\0' "${FILES[@]}" | xargs -0 grep $flags -nHE "$pat" 2>/dev/null \
               | grep -vE "$allow")
    else
        hits=$(printf '%s\0' "${FILES[@]}" | xargs -0 grep $flags -nHE "$pat" 2>/dev/null)
    fi
    local n
    n=$(printf '%s' "$hits" | grep -c . )
    if [[ "$n" -gt 0 ]]; then
        echo "FAIL  $name  ($n)"
        # every finding, deliberately unpaged; unique file:pattern to stay readable
        printf '%s\n' "$hits" | cut -c1-140 | sort -u | sed 's/^/      /'
        findings=$((findings + n))
    else
        echo "ok    $name"
    fi
}

# Home directories. /home/REDACTED is the scrubbed form and is allowed.
# /Users/faamt is allowed: those paths are already committed upstream on
# FAAMT/Mirage main, so they are not new exposure.
run_rule "home paths"        '/home/[A-Za-z0-9_.-]+'      '/home/REDACTED'
run_rule "mac user paths"    '/Users/[A-Za-z0-9_.-]+'     '/Users/faamt'

# Vivado and Xilinx report headers that name the machine.
# Matches the line, then allows it if the value has been scrubbed. Written
# this way round on purpose: a NEW header format is flagged by default
# rather than slipping through an incomplete list of known-bad hostnames.
#
# An angle-bracketed placeholder counts as scrubbed. That is the same
# convention the self-check above already enforces on this file, extended
# to the rest of the repository: documentation explaining what a leaked
# header looks like has to be able to quote its shape. Vivado never
# writes angle brackets there, so `Running On: <ASSET-TAG>` cannot be a
# real value. Without this, --history flags this project's own commit
# messages, and a check that cries wolf is a check that gets ignored.
run_rule "tool host lines"   'Running On:|^\| Host|Host +: +[A-Za-z0-9]' \
                             'REDACTED|<[A-Za-z-]+>'

# Bare hostnames / asset tags. VHDL-2008 and the FROZEN-2026 directory name
# match the shape but are not identifiers.
run_rule "asset tags"        '\b[A-Z]{3,6}-[0-9]{3,6}\b'  'VHDL-|FROZEN-|ISO-|UTF-'

run_rule "ip addresses"      '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' \
                             'Phase [0-9]|Vivado|version|[0-9]\.[0-9]\.[0-9]\.[0-9] (BUFG|Post)'
run_rule "mac addresses"     '\b([0-9a-fA-F]{2}:){4}[0-9a-fA-F]{2}'
# GitHub's users.noreply.github.com form is allowed: it is the address git is
# deliberately configured to commit under precisely so a real one never appears.
# History mode reads commit identities, so this rule sees every author line.
run_rule "email addresses"   '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' \
                             'noreply@anthropic\.com|users\.noreply\.github\.com'

# Credential material. Keyword alone is far too noisy in an ML codebase
# ("token embeddings"), so require an assignment to a non-trivial value.
run_rule "assigned secrets" \
    '(api[_-]?key|secret|passwd|password|auth[_-]?token|access[_-]?token)[[:space:]]*[:=][[:space:]]*[A-Za-z0-9/+_-]{8,}' \
    '' -i
run_rule "private key blocks" 'BEGIN [A-Z ]*PRIVATE KEY' '' -i
run_rule "provider tokens"    '\b(ghp|gho|ghs|github_pat)_[A-Za-z0-9]{16,}|\bsk-[A-Za-z0-9]{20,}|\bxox[baprs]-[A-Za-z0-9-]{10,}|\bAIza[0-9A-Za-z_-]{35}\b'
run_rule "license servers"    '[0-9]{4}@[A-Za-z0-9._-]+|XILINXD_LICENSE|LM_LICENSE_FILE' '' -i

# --- file-name rules -----------------------------------------------------
keyfiles=$(printf '%s\n' "$PATHS" | grep -iE \
    '(^|/)\.env|\.pem$|\.p12$|\.pfx$|(^|/)id_(rsa|dsa|ecdsa|ed25519)$|(^|/)\.netrc$|\.wdf$|\.pb$' || true)
if [[ -n "$keyfiles" ]]; then
    echo "FAIL  credential / telemetry files"
    printf '%s\n' "$keyfiles" | sed 's/^/      /'
    findings=$((findings + 1))
else
    echo "ok    credential / telemetry files"
fi

echo
if [[ "$findings" -eq 0 ]]; then
    echo "SCRUB CHECK PASS -- nothing identifying found. Safe to push."
    exit 0
fi
echo "SCRUB CHECK FAIL -- $findings finding(s). Do NOT push until resolved."
echo
echo "For Vivado logs the fix is the established convention:"
echo "  sed -i 's|/home/<user>|/home/REDACTED|g; s|<HOSTNAME>|REDACTED|g' <file>"
exit 1
