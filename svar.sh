#!/usr/bin/env bash
# Flip the answer on harmagdaavgatt.nu and publish it.
#   ./svar.sh ja     → big green JA
#   ./svar.sh nej    → big red NEJ
#   add --no-push to only change the file locally
set -euo pipefail
cd "$(dirname "$0")"

ans="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
case "$ans" in
  ja)  word="JA"  ;;
  nej) word="NEJ" ;;
  *)   echo "Användning: ./svar.sh ja|nej [--no-push]" >&2; exit 1 ;;
esac

# Swap the state attribute (drives the colour) and the word itself.
perl -0pi -e "s/data-svar=\"(ja|nej)\"/data-svar=\"$ans\"/; s/(<p class=\"nej\" id=\"svar\">)[^<]*(<\/p>)/\${1}$word\${2}/" index.html

grep -q "<html lang=\"sv\" data-svar=\"$ans\">" index.html && grep -q "id=\"svar\">$word<" index.html \
  || { echo "Kunde inte uppdatera index.html – kolla den manuellt." >&2; exit 1; }
echo "index.html → $word"

if [ "${2:-}" = "--no-push" ]; then exit 0; fi

if git diff --quiet -- index.html; then
  echo "Redan $word – inget att publicera."
  exit 0
fi
git add index.html
git commit -q -m "Svar: $word"
git push -q
echo "Publicerat. GitHub Pages uppdateras inom ett par minuter (rensa ev. Cloudflare-cachen)."
