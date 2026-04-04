#!/bin/bash
# Download all per-patient compaction results from Modal volume.
# Each patient has: cache.pt (compacted KV cache) and results.json (eval results)
#
# Usage: bash scripts/download_per_patient.sh [output_dir]

OUT_DIR="${1:-results/per_patient}"
VOLUME="am-experiment-results"
REMOTE_DIR="/per_patient"

echo "Downloading per-patient results from Modal volume '$VOLUME'..."
echo "Output directory: $OUT_DIR"
echo

# List available patients on the volume
PATIENTS=$(modal volume ls "$VOLUME" "$REMOTE_DIR/" 2>/dev/null | grep "patient_" | sed 's|per_patient/||')

if [ -z "$PATIENTS" ]; then
    echo "No patients found on volume."
    exit 1
fi

echo "Found patients:"
echo "$PATIENTS"
echo

for patient in $PATIENTS; do
    echo "--- $patient ---"
    mkdir -p "$OUT_DIR/$patient"

    # Download cache.pt
    modal volume get "$VOLUME" "$REMOTE_DIR/$patient/cache.pt" "$OUT_DIR/$patient/cache.pt" 2>/dev/null
    if [ $? -eq 0 ]; then
        size=$(ls -lh "$OUT_DIR/$patient/cache.pt" 2>/dev/null | awk '{print $5}')
        echo "  cache.pt: $size"
    else
        echo "  cache.pt: MISSING"
    fi

    # Download results.json
    modal volume get "$VOLUME" "$REMOTE_DIR/$patient/results.json" "$OUT_DIR/$patient/results.json" 2>/dev/null
    if [ $? -eq 0 ]; then
        acc=$(python3 -c "import json; r=json.load(open('$OUT_DIR/$patient/results.json')); print(f\"{r['accuracy']:.0%} ({r['correct']}/{r['total']})\")" 2>/dev/null)
        echo "  results.json: $acc"
    else
        echo "  results.json: MISSING"
    fi
done

echo
echo "Done. Results in: $OUT_DIR/"
