#!/bin/zsh -e

# Create a base directory
BASEDIR=.
echo "Creating a new log file..."
mkdir logs
NOW=$(date "+%d-%m-%Y_%H-%M-%S")
LOG="$BASEDIR"/logs/out_$NOW.log
ERR="$BASEDIR"/logs/err_$NOW.log

date > "$LOG"
date > "$ERR"
echo logging... > "$LOGS"

cat "$LOG"
cat "$ERR"
