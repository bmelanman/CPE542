#!/bin/zsh -e

cleanup() {
  rv=$?
  if [ $rv -ne 0 ]; then
    printf "Error!\n"
  else
    printf "exiting...\n"
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
  return 0
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
  rm ./logs/*.log
  date >"$LOG"
  date >"$ERR"
  chmod 777 "$LOG" "$ERR"
  echo "Done!"
  echo "Logs can be found at \"$BASEDIR/logs\""
  return 0
}

disp_msg() {
  printf "\r%s\n" "$@" >&5
  return 0
}

testing() {
  # redirect stdout/stderr to a file
  exec 5>&1 >>"$LOG" 2>>"$ERR"
  disp_msg "pipes redirected"

  # Set cleanup trigger
  trap 'cleanup' EXIT
  disp_msg "trap enabled"

  /usr/bin/python3 ~/Code/GitHub/CSC202/project2-bmelanman/big_O_test.py
  disp_msg "python finished"

  return 0
}

make_basedir

testing & spinner
