#!/bin/zsh

if [ "$EUID" -ne 0 ]
  then echo "Please run as root!"
  exit
fi

sed -i '' -e 's/BASH-no/watermelon/g' /etc/_test_profile && cat /etc/_test_profile
