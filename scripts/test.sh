#!/bin/zsh -e

cleanup() {
  local rv=$?

  exec 1>&5; exec 2>&6;

  kill -9 "$PID"
  printf "\r"

  if [ $rv -ne 0 ]; then
    printf "Error! Please see error logs:\n"
  else
    printf "Installation complete! Goodbye :)\n"
  fi

  exit $rv
}

spinner() {
  local max_cycles=10
  local spin='┤┘┴└├┌┬┐'

  while :; do
    for _ in $(seq 0 1 $max_cycles); do
      for i in $(seq 0 1 ${#spin}); do
        printf "\r  %s\r" "${spin:$i:1}"
        sleep 0.5
      done
    done
    printf "\r"
  done

  return 0
}

disp_msg() {
  printf "\r%s\n" "$@" >&5
  return 0
}

echo_stderr() {
  echo "$@" >&6
  return 1
}

download_lib() {
  echo "Downloading $1..."
  local DIR="$BASEDIR/$1"
  if [ ! -d "$DIR" ]; then
    git clone "$2" "$3" "$DIR" || echo_stderr "git error when downloading $1"
    git -C "$DIR" submodule update --init
  else
    (git -C "$DIR" fetch && git -C "$DIR" merge) || echo_stderr "git error when fetching and merging $1"
    (git submodule update --recursive --remote) || echo_stderr "git error when updating submodules for $1"
  fi
  cd "$DIR" || return 1
  echo "Done!"

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
  echo "Set up complete, logs can be found at \"$BASEDIR/logs\""

  return 0
}

testing() {
  # redirect stdout/stderr to a file
  exec >>"$LOG" 2>>"$ERR"
  disp_msg "pipes redirected"

  /usr/bin/python3 ~/Code/GitHub/CSC202/project2-bmelanman/big_O_test.py
  disp_msg "python finished"

  return 0
}

test_install() {
  # Install Boost
  disp_msg "Installing Boost..."
  download_lib "boost" "https://github.com/boostorg/boost.git" "--recursive"
  return 0
}

# Init stuff
trap 'cleanup' EXIT
exec 5>&1; exec 6>&2;

# Start the spinner!
spinner &
PID=$!

# The rest of everything else
disp_msg "make_basedir"
make_basedir
disp_msg "testing"
testing
disp_msg "test_install"
test_install
