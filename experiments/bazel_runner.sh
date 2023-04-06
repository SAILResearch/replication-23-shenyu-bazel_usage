#!/bin/sh

helpFunction() {
  echo ""
  echo "Usage: $0 -p project_url -t target -c other arguments"
  echo -e "\t-p git repository url of the project to be experimented on"
  echo -e "\t-s subcommand to be used"
  echo -e "\t-t bazel target to be built"
  echo -e "\t-c other cmd arguments to be used"
  exit 1 # Exit script after printing help
}

while getopts "p:s:t:c:" opt; do
  case "$opt" in
  p) project_url="$OPTARG" ;;
  s) subcommand="$OPTARG" ;;
  t) target="$OPTARG" ;;
  c) cmd_args="$OPTARG" ;;
  ?) helpFunction ;; # Print helpFunction in case parameter is non-existent
  esac
done

# Print helpFunction in case parameters are empty
if [ -z "$project_url" ] || [ -z "$target" ]; then
  echo "the value of -p or -t is empty"
  helpFunction
fi

git clone --depth=1 "$project_url" && cd $(basename "$project_url" .git)

echo "Running bazel $subcommand $cmd_args $target"

bazel $subcommand $cmd_args $target
