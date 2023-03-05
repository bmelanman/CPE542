#!/bin/zsh -e

# Create a base directory
BASEDIR=.
echo "Creating a new log file..."
mkdir -p logs
NOW=$(date "+%d-%m-%Y_%H-%M-%S")
LOG="$BASEDIR"/logs/out_$NOW.log
ERR="$BASEDIR"/logs/err_$NOW.log

# Add date and time to each log
date >> "$LOG"
date >> "$ERR"

cat "$LOG"
cat "$ERR"
