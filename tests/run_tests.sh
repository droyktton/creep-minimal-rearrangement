#!/bin/bash
# Internal checks (tests/check.cpp) and, if krgk_creep is available (KRGK=path/to/krgk_creep),
# the critical force against it for the same sample.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
g++ -O2 -std=c++17 -fopenmp -o "$W"/check "$ROOT"/tests/check.cpp
"$W"/check
if [ -n "$KRGK" ]; then
	make -s -C "$ROOT" >/dev/null
	mkdir "$W"/a "$W"/b
	"$KRGK" --L 32 --M 256 --f 0.5 --seed 3 --events 1 --out "$W"/a 2>/dev/null
	"$ROOT"/build/mr_creep --L 32 --M 256 --f 0.5 --seed 3 --events 1 --out "$W"/b
	fa=$(awk '$1=="Fc"{print $2}' "$W"/a/params.txt); fb=$(awk '$1=="Fc"{print $2}' "$W"/b/params.txt)
	cmp -s "$W"/a/critical.dat "$W"/b/critical.dat && [ "$fa" = "$fb" ] && echo "4. F_c and critical configuration as krgk_creep: PASS ($fa)" || { echo "4. F_c: krgk $fa, mr $fb: FAIL"; exit 1; }
fi
