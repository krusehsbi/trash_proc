#!/bin/bash
# Usage: ./generate_images.sh <total_images> <views_per_room>

TOTAL_IMAGES=$1
VIEWS_PER_ROOM=$2

if [ -z "$TOTAL_IMAGES" ] || [ -z "$VIEWS_PER_ROOM" ]; then
  echo "Usage: $0 <total_images> <views_per_room>"
  exit 1
fi

# Calculate how many rooms we need to render
ROOMS=$(( (TOTAL_IMAGES + VIEWS_PER_ROOM - 1) / VIEWS_PER_ROOM ))

echo "Generating $TOTAL_IMAGES images with $VIEWS_PER_ROOM views per room ($ROOMS rooms total)..."

for ((i=1; i<=ROOMS; i++)); do
  echo "[INFO] Rendering room $i with $VIEWS_PER_ROOM views..."
  blenderproc run trash_proc.py --random_room --num_views $VIEWS_PER_ROOM
done

echo "[DONE] Images generated. Check the output/ folder."
