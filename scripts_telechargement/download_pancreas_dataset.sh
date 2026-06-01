#!/bin/bash

# Configuration
DATA_DIR="data"
mkdir -p "$DATA_DIR"
URL="https://openproblems-data.s3.amazonaws.com/resources/datasets/openproblems_v1/pancreas/log_cp10k/dataset.h5ad"
OUTPUT="$DATA_DIR/pancreas_openproblems.h5ad"

echo "--------------------------------------------------------"
echo "📥 Downloading Pancreas Dataset (OpenProblems v1)"
echo "From: $URL"
echo "To: $OUTPUT"
echo "--------------------------------------------------------"

# Check for wget or curl
if command -v wget &> /dev/null; then
    wget -O "$OUTPUT" "$URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$OUTPUT" "$URL"
else
    echo "❌ Error: Neither 'wget' nor 'curl' was found. Please install one to proceed."
    exit 1
fi

echo ""
echo "✅ Download complete!"
echo "File location: $(pwd)/$OUTPUT"
echo "--------------------------------------------------------"
