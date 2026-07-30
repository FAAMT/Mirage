#!/usr/bin/env bash
# =====================================================================
# scrub_check.sh -- refuse to publish identifying or secret material.
#
#   ./tools/scrub_check.sh          # check tracked + staged files
#   ./tools/scrub_check.sh --all    # also check untracked files
#
# Exit 0 = clean, 1 = findings. Run it before every push.
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
[[ "${1:-}" == "--all" ]] && ALL=1

if [[ $ALL -eq 1 ]]; then
    mapfile -d '' FILES < <(git ls-files -z --cached --others --exclude-standard)
else
    mapfile -d '' FILES < <(git ls-files -z --cached --others --exclude-standard \
                            --deduplicate 2>/dev/null || git ls-files -z)
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
run_rule "tool host lines"   'Running On:|^\| Host|Host +: +[A-Za-z0-9]' 'REDACTED'

# Bare hostnames / asset tags. VHDL-2008 and the FROZEN-2026 directory name
# match the shape but are not identifiers.
run_rule "asset tags"        '\b[A-Z]{3,6}-[0-9]{3,6}\b'  'VHDL-|FROZEN-|ISO-|UTF-'

run_rule "ip addresses"      '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' \
                             'Phase [0-9]|Vivado|version|[0-9]\.[0-9]\.[0-9]\.[0-9] (BUFG|Post)'
run_rule "mac addresses"     '\b([0-9a-fA-F]{2}:){4}[0-9a-fA-F]{2}'
run_rule "email addresses"   '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' \
                             'noreply@anthropic\.com'

# Credential material. Keyword alone is far too noisy in an ML codebase
# ("token embeddings"), so require an assignment to a non-trivial value.
run_rule "assigned secrets" \
    '(api[_-]?key|secret|passwd|password|auth[_-]?token|access[_-]?token)[[:space:]]*[:=][[:space:]]*[A-Za-z0-9/+_-]{8,}' \
    '' -i
run_rule "private key blocks" 'BEGIN [A-Z ]*PRIVATE KEY' '' -i
run_rule "provider tokens"    '\b(ghp|gho|ghs|github_pat)_[A-Za-z0-9]{16,}|\bsk-[A-Za-z0-9]{20,}|\bxox[baprs]-[A-Za-z0-9-]{10,}|\bAIza[0-9A-Za-z_-]{35}\b'
run_rule "license servers"    '[0-9]{4}@[A-Za-z0-9._-]+|XILINXD_LICENSE|LM_LICENSE_FILE' '' -i

# --- file-name rules -----------------------------------------------------
keyfiles=$(printf '%s\n' "${FILES[@]}" | grep -iE \
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
