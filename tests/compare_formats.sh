#!/bin/sh
# Converts one GOAL schedule into the legacy and the GOALX binary format
# (default and, optionally, a narrow schema) and checks that LogGOPSim
# produces identical results for all of them, and that the legacy file gives
# the same results with the fixed and the interpreting decoder.
#
# usage: compare_formats.sh <LogGOPSim> <txt2bin> <schedule.goal> [goalx schema]

set -e

SIM=$1
TXT2BIN=$2
GOAL=$3
SCHEMA=$4

if [ -z "$SIM" ] || [ -z "$TXT2BIN" ] || [ -z "$GOAL" ]; then
	echo "usage: $0 <LogGOPSim> <txt2bin> <schedule.goal> [goalx schema]" >&2
	exit 2
fi

DIR=$(mktemp -d)
trap 'rm -rf "$DIR"' EXIT

"$TXT2BIN" -i "$GOAL" -o "$DIR/legacy.bin" > /dev/null
"$TXT2BIN" -i "$GOAL" -o "$DIR/goalx.bin" --format goalx > /dev/null
cp "$DIR/legacy.bin" "$DIR/legacy-interp.bin"
VARIANTS="legacy legacy-interp goalx"
if [ -n "$SCHEMA" ]; then
	"$TXT2BIN" -i "$GOAL" -o "$DIR/narrow.bin" --format goalx --schema "$SCHEMA" > /dev/null
	VARIANTS="$VARIANTS narrow"
fi

# the -V file records every simulated event; the summary lines hold the finish times
for v in $VARIANTS; do
	DECODER=auto
	[ "$v" = legacy ] && DECODER=fixed
	[ "$v" = legacy-interp ] && DECODER=interp
	"$SIM" -f "$DIR/$v.bin" --decoder $DECODER -V "$DIR/$v.viz" | grep -E '^Host|^Maximum|queue on host' > "$DIR/$v.txt"
done

status=0
for v in $VARIANTS; do
	if ! cmp -s "$DIR/legacy.txt" "$DIR/$v.txt" || ! cmp -s "$DIR/legacy.viz" "$DIR/$v.viz"; then
		echo "MISMATCH: $v differs from legacy" >&2
		diff "$DIR/legacy.txt" "$DIR/$v.txt" >&2 || true
		status=1
	fi
	echo "$v: $(wc -c < "$DIR/$v.bin") bytes, $(wc -l < "$DIR/$v.viz") events"
done
exit $status
