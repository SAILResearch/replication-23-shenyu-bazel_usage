#!/bin/bash

helpFunction() {
  echo ""
  echo "Usage: $0 -p project_url -t target -c other arguments"
  echo -e "\t-p git repository url of the project to be experimented on"
  echo -e "\t-b sha of the commit to be built"
  echo -e "\t-s subcommand to be used"
  echo -e "\t-t bazel target to be built"
  exit 1 # Exit script after printing help
}

while getopts "p:b:s:t:c:" opt; do
  case "$opt" in
  p) project_url="$OPTARG" ;;
  b) sha="$OPTARG" ;;
  s) subcommand="$OPTARG" ;;
  t) target="$OPTARG" ;;
  ?) helpFunction ;; # Print helpFunction in case parameter is non-existent
  esac
done

cmd_args="$CMD_ARGS"

# Print helpFunction in case parameters are empty
if [ -z "$project_url" ] || [ -z "$target" ]; then
  echo "the value of -p or -t is empty"
  helpFunction
fi

cp -r /repo/$(basename "$project_url" .git) /root/$(basename "$project_url" .git)

cd /root/$(basename "$project_url" .git)

if [ -z "$sha" ]; then
  echo "No sha provided, using HEAD"
else
  echo "Checking out $sha"
  git checkout $sha
fi

if [[ -f BUILD.bazel || -f BUILD ]]; then
  echo "detected Bazel build files"
else
  >&2 echo "no Bazel build files detected"
  exit 1
fi

echo "Running bazel $subcommand $cmd_args $target"

bazel $subcommand $cmd_args $target
