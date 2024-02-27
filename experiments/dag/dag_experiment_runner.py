import csv
import json
import logging
import os.path
import subprocess
from argparse import ArgumentParser


def generate_dag_and_action_deps(org, project, build_target, commit):
    project_git_url = f"https://github.com/{org}/{project}.git"

    cmd = f"/root/bazel_runner.sh -p {project_git_url} -t {build_target} -c {commit}"
    process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate(cmd.encode('utf-8'))

    if process.returncode != 0:
        logging.error(
            f"failed to run the script 'bazel_runner' for {project}, cmd: {cmd},  stdout: {stdout} error: {stderr}")
        raise Exception(f"failed to run experiment for {project}, stdout: {stdout} error: {stderr}")


def parse_action_dep_paths(action_deps):
    path_fragment_labels = {}
    path_fragment_parents = {}
    for frag in action_deps["pathFragments"]:
        path_fragment_labels[frag["id"]] = frag["label"]
        path_fragment_parents[frag["id"]] = frag["parentId"] if "parentId" in frag else None

    constructed_paths = {}

    def construct_path(path_id) -> str:
        label = path_fragment_labels[path_id]
        parent_id = path_fragment_parents[path_id]

        if parent_id is None:
            return label

        if parent_id in constructed_paths:
            return f"{constructed_paths[parent_id]}/{label}"

        constructed_paths[parent_id] = construct_path(parent_id)
        return f"{constructed_paths[parent_id]}/{label}"

    for frag in action_deps["pathFragments"]:
        if frag["id"] in constructed_paths:
            continue
        constructed_paths[frag["id"]] = construct_path(frag["id"])

    return constructed_paths


def parse_action_dep_artifacts(action_deps):
    paths = parse_action_dep_paths(action_deps)

    artifacts = {}
    for artifact in action_deps["artifacts"]:
        artifacts[artifact["id"]] = paths[artifact["pathFragmentId"]]

    return artifacts


def parse_action_dep_sets(action_deps):
    dep_sets = {}
    for dep_set in action_deps["depSetOfFiles"]:
        artifact_sets = set()
        if "directArtifactIds" in dep_set:
            artifact_sets.update(dep_set["directArtifactIds"])

        if "transitiveDepSetIds" in dep_set:
            for transitive_dep_set_id in dep_set["transitiveDepSetIds"]:
                artifact_sets.update(dep_sets[transitive_dep_set_id])

        dep_sets[dep_set["id"]] = artifact_sets

    return dep_sets


def parse_action_dep_targets(action_deps):
    targets = {}
    for target in action_deps["targets"]:
        targets[target["id"]] = target["label"]

    return targets


def resolve_input_file_path(project, path) -> (str, bool):
    resolved_path = f"/root/{project}/{path}"
    if os.path.exists(resolved_path):
        return resolved_path, os.path.isdir(resolved_path)

    resolved_path = f"/root/{project}/bazel-out/{path}"
    if os.path.exists(resolved_path):
        return resolved_path, os.path.isdir(resolved_path)

    resolved_path = f"/root/{project}/bazel-bin/{path}"
    if os.path.exists(resolved_path):
        return resolved_path, os.path.isdir(resolved_path)

    resolved_path = f"/root/.cache/bazel/_bazel_root"
    for p in os.listdir(resolved_path):
        if p.startswith(("cache", "install")):
            continue

        if os.path.exists(f"{resolved_path}/{p}/{path}"):
            return f"{resolved_path}/{p}/{path}", os.path.isdir(f"{resolved_path}/{p}/{path}")

    return None, None


def parse_action_deps(project, commit=""):
    if commit:
        path = f"/results/{project}/aquery_{project}_{commit}.json"
    else:
        path = f"/results/{project}/aquery_{project}.json"
    with open(f"{path}", "r+") as action_deps_file:
        action_deps = json.load(action_deps_file)

    artifacts = parse_action_dep_artifacts(action_deps)
    dep_sets = parse_action_dep_sets(action_deps)
    targets = parse_action_dep_targets(action_deps)

    results = []
    input_file_sizes = {}
    for action in action_deps["actions"]:
        if "inputDepSetIds" not in action:
            continue

        target = targets[action["targetId"]]
        input_file_paths = set()
        for input_dep_set_id in action["inputDepSetIds"]:
            for artifact_id in dep_sets[input_dep_set_id]:
                input_file_paths.add(artifacts[artifact_id])

        for input_file_path in input_file_paths:
            if "runfiles" in input_file_path or input_file_path.endswith(".params") or input_file_path.startswith("bazel-out/"):
                continue

            if input_file_path in input_file_sizes:
                results.append({"target": target, "path": input_file_path, "size": input_file_sizes[input_file_path]})

            resolved_input_file_path, is_dir = resolve_input_file_path(project, input_file_path)
            if not resolved_input_file_path:
                logging.warning(f"failed to resolve input file path for {project} {target} {input_file_path}")
                continue

            if is_dir:
                for root, dirs, files in os.walk(resolved_input_file_path):
                    for file in files:
                        resolved_file_path = f"{root}/{file}"
                        with open(resolved_file_path, "rb") as input_file:
                            input_file_size = sum(1 for _ in input_file)
                            input_file_sizes[resolved_file_path] = input_file_size
                            results.append({"target": target, "path": resolved_file_path, "size": input_file_size})

                continue
            with open(resolved_input_file_path, "rb") as input_file:
                input_file_size = sum(1 for _ in input_file)
                input_file_sizes[resolved_input_file_path] = input_file_size
                results.append({"target": target, "path": resolved_input_file_path, "size": input_file_size})

    return results


def perform_experiment(org, project, build_target, commit):
    logging.info(f"start running experiment for {project}")

    generate_dag_and_action_deps(org, project, build_target, commit)
    results = parse_action_deps(project, commit)
    if commit:
        path = f"/results/{project}/dep_results_{commit}.json"
    else:
        path = f"/results/{project}/dep_results.json"

    with open(f"{path}", "w+") as results_file:
        csv_writer = csv.DictWriter(results_file, fieldnames=["target", "path", "size"])
        csv_writer.writeheader()

        for result in results:
            csv_writer.writerow(result)

    logging.info(f"finished running experiment for {project}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')

    parser = ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--org", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()

    perform_experiment(args.org, args.project, args.target, args.commit)