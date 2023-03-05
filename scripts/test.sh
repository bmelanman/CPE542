#!/bin/zsh -e

# Create a base directory
BASEDIR=$(pwd)
echo "Creating a new log file..."
mkdir -p logs
NOW=$(date "+%d-%m-%Y_%H-%M-%S")
LOG="$BASEDIR"/logs/out_$NOW.log
ERR="$BASEDIR"/logs/err_$NOW.log

# Add date and time to each log
date >> "$LOG"
date >> "$ERR"
echo "Done!"
echo "Logs can be found at \"$BASEDIR/logs\""

# redirect stdout/stderr to a file
exec >>"$LOG" 2>&1

/usr/bin/python3 ~/Code/GitHub/CSC202/project2-bmelanman/big_O_test.py
