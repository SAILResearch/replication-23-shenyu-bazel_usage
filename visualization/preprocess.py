import os.path
import re

import pandas as pd

from utils import fileutils

build_argument_matchers = {"bazel": re.compile(r"^(--\w+(=[^\s]+)?\s+)"), "maven": re.compile(r"^(-\w+(=[^\s]+)?\s+)")}
build_subcommands = {"bazel": ["build", "test", "run", "coverage"],
                     "maven": ["clean", "compile", "test", "package", "integration-test", "install", "deploy"]}


def preprocess_data(data_dir: str):
    processed_data_dir = os.path.join(data_dir, "processed")
    if not fileutils.exists(processed_data_dir):
        os.mkdir(processed_data_dir)

    parent_dir_name_and_build_tool = {"bazel-projects": "bazel", "maven-large-projects": "maven",
                                      "maven-small-projects": "maven"}
    for parent_dir_name, build_tool in parent_dir_name_and_build_tool.items():
        preprocess_ci_tools(os.path.join(data_dir, parent_dir_name),
                            processed_data_dir, build_tool, parent_dir_name)


def preprocess_ci_tools(source_dir: str, target_dir: str, build_tool: str, target_filename_prefix=""):
    source_data_file_path = os.path.join(source_dir, "build_commands.csv")
    target_processed_file_path = os.path.join(target_dir, f"{target_filename_prefix}-build_tools.csv")

    build_commands = pd.read_csv(source_data_file_path, sep="#")
    build_commands = build_commands[build_commands["build_tool"] == build_tool]
    build_commands["subcommands"] = build_commands.apply(lambda row: label_subcommand(row, build_tool), axis=1)
    build_commands = build_commands[build_commands["subcommands"] != ""]

    build_commands = build_commands.drop_duplicates(subset=["project", "ci_tool", "subcommands"], keep="first")
    build_commands = build_commands.drop(
        columns=["raw_arguments", "build_tool", "local_cache", "remote_cache", "parallelism", "cores"])
    build_commands.to_csv(target_processed_file_path, encoding="utf-8", index=False)


def label_subcommand(row, build_tool) -> str:
    args = row["raw_arguments"]

    args = args.lstrip()
    while match := build_argument_matchers[build_tool].search(args):
        matched_argument = match.group(1)
        args = args.replace(matched_argument, "")

    subcommands = []
    # TODO make codes more graceful
    while True:
        matched = False
        for subcommand in build_subcommands[build_tool]:
            if args.startswith(f"{subcommand} ") or (" " not in args and args == subcommand):
                subcommands.append(subcommand)
                args = args.replace(f"{subcommand}", "")
                args = args.lstrip()
                matched = True
                break

        if not matched:
            break

    return ",".join(subcommands)
