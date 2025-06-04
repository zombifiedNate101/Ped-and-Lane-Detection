#!/bin/bash
# Run:
# chmod +x fmpeg.sh

# ./fmpeg.sh

LOGFILE="my_log.txt"
VIDEOFILE="IMG_2838.mp4"
OUTDIR="extracted_frames"

mkdir -p "$OUTDIR"

# Extract frame numbers from lines starting with "Frame"
grep -oP "^Frame \K\d+" "$LOGFILE" | sort -n | uniq > frames.txt

# Get FPS from the video (e.g. 30 or 29.97)
FPS=$(ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate "$VIDEOFILE" | awk -F/ '{ if ($2) { print $1 / $2 } else { print $1 } }')

echo "Detected FPS: $FPS"

# Read each frame number and extract using ffmpeg
while read -r FRAME; do
    # Skip empty or invalid lines
    [[ -z "$FRAME" || ! "$FRAME" =~ ^[0-9]+$ ]] && continue

    # Calculate timestamp in seconds
    TIME=$(echo "scale=6; $FRAME / $FPS" | bc)

    # Output filename
    OUTFILE=$(printf "%s/frame_%06d.jpg" "$OUTDIR" "$FRAME")

    # Extract frame using ffmpeg
    ffmpeg -hide_banner -loglevel error -ss "$TIME" -i "$VIDEOFILE" -frames:v 1 "$OUTFILE"
    echo "Extracted frame $FRAME -> $OUTFILE"
done < frames.txt

echo "✅ Done extracting all valid frames to $OUTDIR"
