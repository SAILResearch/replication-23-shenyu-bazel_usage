#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: git-commit-size.sh <commit hash>" 1>&2
  exit 1
fi

HASH=$1

git diff --shortstat $HASH~1 $HASH
