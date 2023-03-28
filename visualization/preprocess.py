import os.path
import re

import pandas as pd

from utils import fileutils

build_argument_matchers = {"bazel": re.compile(r"^(--[\w_]+(=[^\s]+)?\s+)"),
                           "maven": re.compile(r"^(--?[-\w]+( ?[^-\s]+)?\s+)")}
build_subcommands = {"bazel": ["build", "test", "run", "coverage"],
                     "maven": ["clean", "compile", "test", "package", "integration-test", "install", "verify",
                               "deploy"]}

manually_checked_remote_cahce_projects = ["tensorflow_tensorflow", "bazelbuild_bazel-toolchains",
                                          "rabbitmq_rabbitmq-server", "grpc_grpc", "dfinity_ic", "scionproto_scion",
                                          "buildbuddy-io_rules_xcodeproj", "angular_components",
                                          "buildbuddy-io_buildbuddy", "formatjs_formatjs", "dropbox_dbx_build_tools",
                                          "mwitkow_go-proto-validators", "vaticle_biograkn", "bazelbuild_rules_docker",
                                          "rabbitmq_ra", "kythe_kythe", "tweag_rules_haskell", "aspect-build_rules_js",
                                          "bazelbuild_rules_jvm_external", "elastic_kibana", "envoyproxy_nighthawk",
                                          "iree-org_iree", "pixie-io_pixie", "pingcap_tidb", "istio_proxy",
                                          "lewish_asciiflow", "angular_angular", "android_testing-samples",
                                          "tensorflow_federated", "carbon-language_carbon-lang",
                                          "CodeIntelligenceTesting_jazzer", "tensorflow_tfjs", "brendanhay_amazonka",
                                          "lowRISC_opentitan", "tensorflow_runtime", "envoyproxy_envoy",
                                          "GoogleCloudPlatform_esp-v2", "angular_angular-cli", "dataform-co_dataform"]


def preprocess_data(data_dir: str):
    processed_data_dir = os.path.join(data_dir, "processed")
    if not fileutils.exists(processed_data_dir):
        os.mkdir(processed_data_dir)

    parent_dir_name_and_build_tool = {"bazel-projects": "bazel", "maven-large-projects": "maven",
                                      "maven-small-projects": "maven"}
    for parent_dir_name, build_tool in parent_dir_name_and_build_tool.items():
        source_dir = os.path.join(data_dir, parent_dir_name)

        preprocess_ci_tools(source_dir, processed_data_dir, build_tool, parent_dir_name)
        preprocess_feature_usage(source_dir, processed_data_dir, build_tool, parent_dir_name)
        preprocess_build_rules(source_dir, processed_data_dir, build_tool, parent_dir_name)
        preprocess_script_usage(source_dir, processed_data_dir, build_tool, parent_dir_name)
        preprocess_arg_size(source_dir, processed_data_dir, build_tool, parent_dir_name)
        preprocess_project_data(source_dir, processed_data_dir, build_tool, parent_dir_name)


def preprocess_build_rules(source_dir: str, processed_data_dir: str, build_tool: str, target_filename_prefix=""):
    build_rules_data_path = os.path.join(source_dir, "build_rules.csv")
    target_processed_file_path = os.path.join(processed_data_dir, f"{target_filename_prefix}-build_rules.csv")

    build_rules = pd.read_csv(build_rules_data_path)
    build_rules.to_csv(target_processed_file_path, encoding="utf-8", index=False)


def preprocess_project_data(source_dir, processed_data_dir, build_tool, target_filename_prefix=""):
    project_data_path = os.path.join(source_dir, "projects.csv")
    target_processed_file_path = os.path.join(processed_data_dir, f"{target_filename_prefix}-projects.csv")

    projects = pd.read_csv(project_data_path, sep="#")
    projects.to_csv(target_processed_file_path, encoding="utf-8", index=False)


def preprocess_ci_tools(source_dir: str, target_dir: str, build_tool: str, target_filename_prefix=""):
    build_commands_data_path = os.path.join(source_dir, "build_commands.csv")
    ci_tool_usage_data_path = os.path.join(source_dir, "ci_tool_usage.csv")
    target_processed_file_path = os.path.join(target_dir, f"{target_filename_prefix}-build_tools.csv")

    build_commands = pd.read_csv(build_commands_data_path, sep="#")
    ci_tool_usages = pd.read_csv(ci_tool_usage_data_path).drop_duplicates()
    build_commands = build_commands[build_commands["build_tool"] == build_tool]
    build_commands["subcommands"] = build_commands.apply(lambda row: label_subcommand(row, build_tool)[0], axis=1)
    build_commands["skip_tests"] = build_commands.apply(lambda row: label_subcommand(row, build_tool)[1], axis=1)

    # TODO some maven projects specify their default goals in the maven pom,
    # so we need to read the default goals from the pom file as the subcommands.
    # for example: https://github.com/apache/commons-codec/blob/master/pom.xml
    build_commands = build_commands[build_commands["subcommands"] != ""]

    build_commands = build_commands.drop_duplicates(subset=["project", "ci_tool", "subcommands"], keep="first")
    build_commands = build_commands.drop(
        columns=["raw_arguments", "build_tool", "local_cache", "remote_cache", "parallelism", "cores",
                 "invoked_by_script"])
    build_commands["use_build_tool"] = True
    for _, row in ci_tool_usages.iterrows():
        if build_commands.loc[(build_commands["project"] == row["project"]) &
                              (build_commands["ci_tool"] == row["ci_tool"])].any().all():
            continue

        build_commands = build_commands.append(
            {"project": row["project"], "ci_tool": row["ci_tool"], "use_build_tool": False, "subcommands": ""},
            ignore_index=True)

    build_commands.to_csv(target_processed_file_path, encoding="utf-8", index=False)


def preprocess_feature_usage(source_dir: str, target_dir: str, build_tool: str, target_filename_prefix=""):
    source_data_file_path = os.path.join(source_dir, "build_commands.csv")
    target_processed_file_path = os.path.join(target_dir, f"{target_filename_prefix}-feature_usage.csv")

    build_commands = pd.read_csv(source_data_file_path, sep="#")
    build_commands = build_commands[build_commands["build_tool"] == build_tool]
    build_commands["use_parallelization"] = build_commands.apply(lambda row: int(row["parallelism"]) > 1, axis=1)

    build_commands["remote_cache"] = build_commands.apply(
        lambda row: row["remote_cache"] or row["project"] in manually_checked_remote_cahce_projects, axis=1)

    build_commands = build_commands.drop(
        columns=["raw_arguments", "build_tool", "parallelism", "cores", "invoked_by_script"])
    build_commands = build_commands.drop_duplicates()
    build_commands.to_csv(target_processed_file_path, encoding="utf-8", index=False)


def label_subcommand(row, build_tool) -> (str, bool):
    args = row["raw_arguments"]

    skipTest = "-DskipTests" in args or "-Dmaven.test.skip" in args
    # skipITs = "-DskipITs" in args or "-DskipIT" in args
    args = args.lstrip()
    while match := build_argument_matchers[build_tool].search(args):
        matched_argument = match.group(1)
        for subcommand in build_subcommands[build_tool]:
            if matched_argument.rstrip().endswith(f" {subcommand}"):
                matched_argument = matched_argument.rstrip().replace(subcommand, "")

        args = args.replace(matched_argument, "")

    subcommands = []
    # TODO replace the while True!!!
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

    return ",".join(subcommands), skipTest


def preprocess_script_usage(source_dir: str, target_dir: str, build_tool: str, target_filename_prefix=""):
    source_data_file_path = os.path.join(source_dir, "build_commands.csv")
    target_processed_file_path = os.path.join(target_dir, f"{target_filename_prefix}-script_usage.csv")

    build_commands = pd.read_csv(source_data_file_path, sep="#")
    build_commands = build_commands[build_commands["build_tool"] == build_tool]
    build_commands = build_commands.drop(
        columns=["raw_arguments", "build_tool", "parallelism", "cores", "local_cache", "remote_cache"])
    build_commands = build_commands.drop_duplicates()
    build_commands.to_csv(target_processed_file_path, encoding="utf-8", index=False)


def preprocess_arg_size(source_dir, processed_data_dir, build_tool, parent_dir_name):
    source_data_file_path = os.path.join(source_dir, "build_command_arg_size.csv")
    target_processed_file_path = os.path.join(processed_data_dir, f"{parent_dir_name}-arg_size.csv")

    build_command_arg_size = pd.read_csv(source_data_file_path, sep="#")
    build_command_arg_size = build_command_arg_size[build_command_arg_size["build_tool"] == build_tool]
    build_command_arg_size = build_command_arg_size.drop(columns=["build_tool"])
    build_command_arg_size = build_command_arg_size.drop_duplicates()
    build_command_arg_size.to_csv(target_processed_file_path, encoding="utf-8", index=False)


if __name__ == "__main__":
    label_subcommand({"raw_arguments": "--batch-mode --update-snapshots verify"}, "maven")
