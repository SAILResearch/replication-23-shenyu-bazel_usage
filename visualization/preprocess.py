import os.path
import re
import shutil
import subprocess

import pandas as pd

from utils import fileutils

build_subcommands = {"bazel": ["build", "test", "run", "coverage", "version", "sync", "query", "fetch", "info", "clean",
                               "shutdown", "help", "mobile-install", "aquery", "cquery", "config", "dump",
                               "analyze-profile"],
                     "maven": ["validate", "initialize", "generate-sources", "process-sources", "generate-resources",
                               "process-resources", "compile", "process-classes", "generate-test-sources",
                               "process-test-sources", "generate-test-resources", "process-test-resources",
                               "test-compile", "process-test-classes", "test", "prepare-package", "package",
                               "pre-integration-test", "integration-test", "post-integration-test", "verify",
                               "install", "deploy", "pre-clean", "clean", "post-clean", "pre-site", "site",
                               "post-site", "site-deploy"]}

maven_plugin_goal_phase_mappings = \
    {"clean:clean": "clean", "resources:resources": "process-resources", "compiler:compile": "compile",
     "resources:testResources": "process-test-resources", "compiler:testCompile": "test-compile",
     "surefire:test": "test", "ejb:ejb": "package", "ejb3:ejb3": "package", "jar:jar": "package", "par:par": "package",
     "rar:rar": "package", "war:war": "package", "install:install": "install",
     "ear:generate-application-xml": "generate-resources",
     "ear:ear": "package", "plugin:descriptor": "generate-resources",
     "plugin:addPluginArtifactMetadata": "package", "deploy:deploy": "deploy", "site:site": "site",
     "site:deploy": "site-deploy"}

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
    preprocess_parallelization_experiments(data_dir, processed_data_dir)
    preprocess_cache_experiments(data_dir, processed_data_dir)


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
    manually_added_build_commands_path = os.path.join(source_dir, "build_commands_manually_added.csv")
    ci_tool_usage_data_path = os.path.join(source_dir, "ci_tool_usage.csv")
    target_processed_file_path = os.path.join(target_dir, f"{target_filename_prefix}-build_tools.csv")

    build_commands = pd.read_csv(build_commands_data_path, sep="#")
    build_commands = pd.concat([build_commands, pd.read_csv(manually_added_build_commands_path, sep="#")])

    ci_tool_usages = pd.read_csv(ci_tool_usage_data_path).drop_duplicates()
    build_commands = build_commands[build_commands["build_tool"] == build_tool]
    build_commands["subcommands"] = build_commands.apply(lambda row: label_subcommand(row, build_tool)[0], axis=1)
    build_commands["skip_tests"] = build_commands.apply(lambda row: label_subcommand(row, build_tool)[1], axis=1)
    build_commands["build_tests_only"] = build_commands.apply(lambda row: label_subcommand(row, build_tool)[2], axis=1)

    # TODO some maven projects specify their default goals in the maven pom,
    # so we need to read the default goals from the pom file as the subcommands.
    # for example: https://github.com/apache/commons-codec/blob/master/pom.xml
    build_commands = build_commands[build_commands["subcommands"] != ""]

    build_commands = build_commands.drop_duplicates(subset=["project", "ci_tool", "subcommands"], keep="first")
    build_commands = build_commands.drop(
        columns=["raw_arguments", "build_tool", "local_cache", "remote_cache", "parallelism", "cores",
                 "invoker"])
    build_commands["use_build_tool"] = True

    lst = []
    for _, row in ci_tool_usages.iterrows():
        if build_commands.loc[(build_commands["project"] == row["project"]) &
                              (build_commands["ci_tool"] == row["ci_tool"])].any().all():
            continue

        lst.append({"project": row["project"], "ci_tool": row["ci_tool"], "use_build_tool": False, "subcommands": ""})

    build_commands_extended = pd.DataFrame(lst)
    build_commands = pd.concat([build_commands, build_commands_extended])

    build_commands.to_csv(target_processed_file_path, encoding="utf-8", index=False)


def preprocess_feature_usage(source_dir: str, target_dir: str, build_tool: str, target_filename_prefix=""):
    source_data_file_path = os.path.join(source_dir, "build_commands.csv")
    manually_added_build_commands_path = os.path.join(source_dir, "build_commands_manually_added.csv")
    target_processed_file_path = os.path.join(target_dir, f"{target_filename_prefix}-feature_usage.csv")

    build_commands = pd.read_csv(source_data_file_path, sep="#")
    build_commands = pd.concat([build_commands, pd.read_csv(manually_added_build_commands_path, sep="#")])
    build_commands = build_commands[build_commands["build_tool"] == build_tool]
    build_commands["use_parallelization"] = build_commands.apply(lambda row: int(row["parallelism"]) > 1, axis=1)

    build_commands["remote_cache"] = build_commands.apply(
        lambda row: row["remote_cache"] or row["project"] in manually_checked_remote_cahce_projects, axis=1)

    build_commands = build_commands.drop(
        columns=["raw_arguments", "build_tool", "parallelism", "cores", "invoker"])
    build_commands = build_commands.drop_duplicates()
    build_commands.to_csv(target_processed_file_path, encoding="utf-8", index=False)


def label_subcommand(row, build_tool) -> (str, bool, bool):
    # skipITs = "-DskipITs" in args or "-DskipIT" in args
    skipTest = "-DskipTests" in row["raw_arguments"] or "-Dmaven.test.skip" in row["raw_arguments"]
    # TODO build_test_only
    build_tests_only = "--build_tests_only" in row["raw_arguments"]

    args = row["raw_arguments"].strip().split()
    if len(args) == 0:
        return "", skipTest, build_tests_only

    maven_plugin_goal_matcher = re.compile(r"(\w+:[\w@]+)(:\d+\.\d+\.\d+)?")
    subcommands = []
    for arg in args:
        if arg in build_subcommands[build_tool] or maven_plugin_goal_matcher.match(arg):
            # we convert the maven plugin goal to the corresponding maven phase according
            # to the Maven built-in lifecycle bindings
            if arg in maven_plugin_goal_phase_mappings:
                arg = maven_plugin_goal_phase_mappings[arg]

            subcommands.append(arg)

    return ",".join(subcommands), skipTest, build_tests_only


def preprocess_script_usage(source_dir: str, target_dir: str, build_tool: str, target_filename_prefix=""):
    source_data_file_path = os.path.join(source_dir, "build_commands.csv")
    manually_added_build_commands_path = os.path.join(source_dir, "build_commands_manually_added.csv")
    target_processed_file_path = os.path.join(target_dir, f"{target_filename_prefix}-script_usage.csv")

    build_commands = pd.read_csv(source_data_file_path, sep="#")
    build_commands = pd.concat([build_commands, pd.read_csv(manually_added_build_commands_path, sep="#")])
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


def preprocess_parallelization_experiments(data_dir, processed_data_dir):
    experiments_data_file_path = os.path.join(data_dir, "parallel-experiments", "parallelization-experiments.csv")
    cache_experiments_data_file_path = os.path.join(data_dir, "cache-experiments", "cache-experiments.csv")
    target_processed_file_path = os.path.join(processed_data_dir, "parallelization-experiments.csv")

    project_data_path = os.path.join(data_dir, "bazel_projects_manually_examined.csv")
    project_data = pd.read_csv(project_data_path, sep=",")

    experiments = pd.read_csv(experiments_data_file_path, sep=",")
    cache_experiments_data = pd.read_csv(cache_experiments_data_file_path, sep=",")

    experiments = experiments.drop_duplicates()

    projects_to_be_dropped = []

    parallelisms = [1, 2, 4, 8, 16]
    for project in experiments["project"].unique():
        _, project_name = project.split("_", maxsplit=1)
        if cache_experiments_data.loc[(cache_experiments_data["project"] == project_name) & (
                cache_experiments_data["status"] == "success")].shape[0] == 0:
            projects_to_be_dropped.append((project, "build"))
            projects_to_be_dropped.append((project, "test"))
            continue

        for subcommand in ["build"]:
            for parallelism in parallelisms:
                cnt = experiments.query(
                    f"project == '{project}' and subcommand == '{subcommand}' and parallelism == {parallelism}").shape[
                    0]
                if cnt == 0:
                    print(f"Project to drop: project: {project}, subcommand: {subcommand}, parallelism: {parallelism}")
                    projects_to_be_dropped.append((project, subcommand))
        # print(f"project and subcommand: {project} {subcommand}")
        experiments.loc[experiments["project"] == project, "commits"] = \
            project_data[project_data["project"] == project]["commits"].values[0]
    for project, subcommand in projects_to_be_dropped:
        experiments = experiments.drop(
            experiments[(experiments["project"] == project) & (experiments["subcommand"] == subcommand)].index)

    experiments.to_csv(target_processed_file_path, encoding="utf-8", index=False)


# TODO get the commit size in experiments
class LocalGitRepository:
    def __init__(self, org: str, project: str):
        self.org = org
        self.project = project

    def __enter__(self):
        proc = subprocess.run(["git", "clone", f"https://github.com/{self.org}/{self.project}.git"])
        if proc.returncode != 0:
            raise Exception(f"Failed to clone {self.org}/{self.project}")

        self.repo_path = f"{self.project}/"

        proc = subprocess.run(["cp", "git-commit-size.sh", self.repo_path])
        if proc.returncode != 0:
            raise Exception(f"Failed to copy git-commit-size.sh to {self.repo_path}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.repo_path)


insertion_matcher = re.compile(r"(\d+) insertion[s]?\(\+\)")
deletion_matcher = re.compile(r"(\d+) deletion[s]?\(-\)")


def preprocess_cache_experiments(data_dir, processed_data_dir):
    experiments_data_file_path = os.path.join(data_dir, "cache-experiments", "cache-experiments.csv")
    target_processed_file_path = os.path.join(processed_data_dir, "cache-experiments.csv")

    project_data_path = os.path.join(data_dir, "bazel_projects_manually_examined.csv")
    project_data = pd.read_csv(project_data_path, sep=",")

    experiments = pd.read_csv(experiments_data_file_path, sep=",")
    projects_to_be_dropped = []

    for project_name in experiments["project"].unique():
        for full_project_name in project_data["project"].unique():
            if full_project_name.endswith(project_name):
                experiments.loc[experiments["project"] == project_name, "project"] = full_project_name
                project = full_project_name
                break

        skip = True
        for cache_type in ["remote", "local", "external", "no_cache"]:
            if experiments.loc[(experiments["project"] == project) & (experiments["cache_type"] == cache_type)].shape[
                0] == 0:
                print(f"Project has no correspondent experiments: project: {project}, cache_type: {cache_type}")
                projects_to_be_dropped.append(project)
                skip = False
                break
        if not skip:
            continue

        org = project.split("_")[0]
        with LocalGitRepository(org, project_name) as repo:
            for commit in experiments[experiments["project"] == project]["commit"].unique():
                proc = subprocess.run([f"cd {repo.repo_path} && ./git-commit-size.sh {commit}"],
                                      capture_output=True, text=True, shell=True)
                if proc.returncode != 0:
                    raise Exception(f"Failed to get number of lines changed for commit {project}")

                size = 0
                if match := insertion_matcher.search(proc.stdout):
                    size += int(match.group(1))
                if match := deletion_matcher.search(proc.stdout):
                    size += int(match.group(1))

                print(f"project: {project}, commit: {commit}, size: {size}")

                experiments.loc[(experiments["project"] == project) & (experiments["commit"] == commit), "size"] = size

        experiments.loc[experiments["project"] == project, "commits"] = \
            project_data[project_data["project"] == project]["commits"].values[0]

    for project in projects_to_be_dropped:
        experiments = experiments.drop(experiments[experiments["project"] == project].index)

    experiments["commits"] = experiments["commits"].astype(int, errors='ignore')
    experiments["size"] = experiments["size"].astype(int, errors='ignore')
    experiments["processes"] = experiments["processes"].astype(int, errors='ignore')
    experiments.to_csv(target_processed_file_path, encoding="utf-8", index=False)


if __name__ == "__main__":
    print(label_subcommand({"raw_arguments": '-B -V -e "-Dstyle.color=always" -Prun-it verify'}, "maven"))
