#!/bin/zsh -e

cleanup() {
  rv=$?
  exec 1>&5
  if [ $rv -ne 0 ]; then
    echo "An error has occurred, "
    cat "$ERR"
  else
    printf "Done!\n"
  fi
  exit $rv
}

spinner() {
  pid=$! # Process Id of the previous running command
  spin='-\|/'
  i=0
  while kill -0 $pid 2>/dev/null
  do
    i=$(( (i+1) %4 ))
    printf "\r%s" "${spin:$i:1}"
    sleep 1
  done
  printf "\b"
}

make_basedir() {
  # Create a base directory
  BASEDIR=$(pwd)
  export BASEDIR
  echo "Creating a new log file..."
  mkdir -m 777 -p logs
  NOW=$(date "+%d-%m-%Y_%H-%M-%S")
  LOG="$BASEDIR"/logs/out_$NOW.log
  ERR="$BASEDIR"/logs/err_$NOW.log
  date >>"$LOG"
  date >>"$ERR"
  chmod 777 "$LOG" "$ERR"
  echo "Done!"
  echo "Logs can be found at \"$BASEDIR/logs\""
  return "$(true)"
}

testing() {
  # redirect stdout/stderr to a file
  exec 5>&1 >>"$LOG" 2>>"$ERR"

  # Set cleanup trigger
  trap 'cleanup' EXIT

  /usr/bin/python3 ~/Code/GitHub/CSC202/project2-bmelanman/big_O_test.py
}

make_basedir
sleep 5 & spinner
