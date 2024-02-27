#!/bin/bash

helpFunction() {
  echo ""
  echo "Usage: $0 -p project_url -t target -c other arguments"
  echo -e "\t-p git repository url of the project to be experimented on"
  echo -e "\t-t bazel target to be built"
  echo -e "\t-c commit to be built"
  exit 1 # Exit script after printing help
}

while getopts "p:t:c:" opt; do
  case "$opt" in
  p) project_url="$OPTARG" ;;
  t) target="$OPTARG" ;;
  c) commit="$OPTARG" ;;
  ?) helpFunction ;; # Print helpFunction in case parameter is non-existent
  esac
done

# Print helpFunction in case parameters are empty
if [ -z "$project_url" ] || [ -z "$target" ]; then
  echo "the value of -p or -t is empty"
  helpFunction
fi

project_name=$(basename "$project_url" .git)

git clone "$project_url"
cd "$project_name" || exit

main_branch=$(awk -F "/" '{print $NF}' .git/refs/remotes/origin/HEAD)

if [[ -z "$commit" ]]; then
    git checkout $(git rev-list -n 1 --before="2023-07-31" "$main_branch")

    if [[ "$project_name" == "brunsli" ]]; then
        git checkout 300af107deecab45bec40c2df90611bb533b606b
    fi

    if [[ "$project_name" == "squzy" ]]; then
        git checkout 0babb18b3ae72179fa4bab237a240e14879fa122
    fi

    if [[ "$project_name" == "rules_proto" ]]; then
        git checkout 3799dab3ead79435332c90b8770ea31a8af14bbc
    fi
else
    git checkout "$commit"
fi


bazel build "$target"

echo "Running bazel cquery \"deps($target)\" --noimplicit_deps --check_visibility=false --output graph_$project_name > graph.out"

bazel cquery "deps($target)" --noimplicit_deps --check_visibility=false --output graph > graph_"$project_name".out

echo "Generated graph_$project_name.out"

echo "Running bazel aquery \"deps($target)\" --output=jsonproto --check_visibility=false --include_commandline=false > aquery_$project_name.json"


bazel aquery 'deps('"$target"')' --output=jsonproto --noimplicit_deps --check_visibility=false --include_commandline=false > aquery_"$project_name".json

mkdir -p /results/"$project_name"

if [[ -z "$commit" ]]; then
    mv graph_"$project_name".out /results/"$project_name"
    mv aquery_"$project_name".json /results/"$project_name"
else
    mv graph_"$project_name".out /results/"$project_name"/graph_"$project_name"_"$commit".out
    mv aquery_"$project_name".json /results/"$project_name"/aquery_"$project_name"_"$commit".json
fi

