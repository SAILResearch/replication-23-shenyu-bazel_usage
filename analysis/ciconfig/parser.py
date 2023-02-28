import logging
import os
import re
from abc import abstractmethod
from enum import Enum

import yaml

from utils import fileutils


class CIToolType(Enum):
    GITHUB_ACTIONS = "GitHub Actions"
    CIRCLE_CI = "CircleCI"
    BUILDKITE = "Buildkite"


class BuildCommand:
    def __init__(self, build_tool: str, raw_arguments: str, local_cache: bool = False, local_cache_paths=None,
                 cores: int = 2):
        if local_cache_paths is None:
            local_cache_paths = []
        self.build_tool = build_tool
        self.raw_arguments = raw_arguments
        self.local_cache = local_cache
        self.local_cache_paths = local_cache_paths
        self.cores = cores


class CIConfig:
    def __init__(self, ci_tool: str):
        self.ci_tool = ci_tool
        self.build_commands = []


class CIConfigParser:
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.ci_file_pattern = re.compile(self._ci_file_pattern())
        self.sh_scripts = list(fileutils.scan_tree(project_dir, re.compile(r".*\.sh"), match_path=False))
        self.bazel_command_matcher = re.compile(".*bazel (.+)")
        self.maven_command_matcher = re.compile(".*mvn (.*)")

    def parse(self) -> [CIConfig]:
        ci_configs = []
        for ci_file in self._scan_ci_files():
            with open(ci_file.path, "r") as f:
                ci_config_str = f.read()
            if not ci_config_str:
                continue

            ci_config = self.parse_ci_file(ci_config_str)
            if not ci_config:
                continue
            ci_configs.append(ci_config)
        return ci_configs

    @abstractmethod
    def parse_ci_file(self, ci_file_str) -> CIConfig:
        pass

    @abstractmethod
    def ci_tool_type(self) -> str:
        pass

    @abstractmethod
    def _ci_file_pattern(self) -> str:
        pass

    def _scan_ci_files(self) -> [os.DirEntry]:
        possible_ci_files = []
        for f in fileutils.scan_tree(self.project_dir, self.ci_file_pattern):
            possible_ci_files.append(f)
        return possible_ci_files

    def _parse_commands(self, command) -> [BuildCommand]:
        cmds = []
        command_matchers = {"maven": self.maven_command_matcher, "bazel": self.bazel_command_matcher}
        for build_tool, matcher in command_matchers.items():
            if match := matcher.match(command):
                cmds.append(
                    BuildCommand(build_tool, match.group(1)))
            else:
                # if this step invoke a script, we look at that script to check if they run any Bazel command within
                for script_file in self.sh_scripts:
                    if script_file.name in command:
                        with open(script_file.path, "r") as f:
                            raw_script = f.read()
                            for match in matcher.finditer(raw_script):
                                cmds.append(
                                    BuildCommand(build_tool, match.group(1)))

        filtered = []
        for cmd in cmds:
            if cmd.build_tool == "bazel":
                bazel_sub_cmds = ["build", "test", "run", "coverage"]
                if not any(x in cmd.raw_arguments for x in bazel_sub_cmds):
                    continue

            filtered.append(cmd)

        return filtered


class GitHubActionConfigParser(CIConfigParser):
    def __init__(self, project_dir: str):
        super().__init__(project_dir)
        self.runner_cores_pattern = re.compile(r".*(\d+)-?cores?")

    def parse_ci_file(self, ci_config_str: str) -> CIConfig:
        gha_cfg = CIConfig(self.ci_tool_type())

        try:
            cfgs = yaml.full_load(ci_config_str)
        except yaml.error.YAMLError as e:
            logging.error(f"error when load yaml {ci_config_str}, reason {e}")
            return None
        if "jobs" not in cfgs:
            return gha_cfg

        for job in cfgs["jobs"].values():
            if "steps" not in job:
                continue

            # we won't consider the self-hosted runners here as they can only be used in private GitHub repository.
            # the default number of cores for linux/windows runner is 2
            cores = 2
            if "runs-on" not in job:
                continue

            # TODO log or save the runners used by projects so that we can find custom runners and look at the hardware resources manually
            runners = []
            if type(job["runs-on"]) is str:
                runners.append(job["runs-on"])
            else:
                runners.extend(job["runs-on"])

            # for workflows with multiple runners, we use the last runner
            for runner in runners:
                # the default number of cores for macos runner is 3
                # https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources
                if runner.startswith("macos"):
                    cores = 3

                # if the workflow uses larger runners that have more resources, the name of the runner may reflect
                # the number of cores given to the runner
                if match := self.runner_cores_pattern.match(runner):
                    cores = int(match.group(1))

            local_cache_enable = False
            local_cache_paths = None

            for step in job["steps"]:
                # this step uses cache action
                if "uses" in step and step["uses"].startswith("actions@cache"):
                    if "with" in step and "path" in step["with"]:
                        local_cache_enable = True
                        local_cache_paths = step["with"]["path"].splitlines()
                    continue

                # this step runs command
                if "run" in step:
                    command = step["run"]
                    for cmd in self._parse_commands(command):
                        cmd.cores = cores
                        cmd.local_cache = local_cache_enable
                        cmd.local_cache_paths = local_cache_paths
                        gha_cfg.build_commands.append(cmd)

        return gha_cfg

    def ci_tool_type(self) -> str:
        return "github_actions"

    def _ci_file_pattern(self) -> str:
        project_dir_regex_literal = re.escape(str.rstrip(self.project_dir, "/"))
        return rf"^{project_dir_regex_literal}/\.github/workflows/.*\.ya?ml$"


class CircleCIConfigParser(CIConfigParser):
    def __init__(self, project_dir):
        super().__init__(project_dir)

    def parse_ci_file(self, ci_config_str: str) -> CIConfig:
        cc_cfg = CIConfig(self.ci_tool_type())

        # TODO extract the yaml loading to parent class
        try:
            cfgs = yaml.full_load(ci_config_str)
        except yaml.error.YAMLError as e:
            logging.error(f"error when load yaml {ci_config_str}, reason {e}")
            return None

        if not cfgs:
            return None

        if "jobs" not in cfgs:
            return cc_cfg

        for job in cfgs["jobs"].values():
            if "steps" not in job:
                continue

            cores = self._convert_res_class_to_cores(job)
            if not cores:
                continue

            local_cache_enable = False
            local_cache_paths = []
            local_cache_keys = []
            bazel_cmds = []

            for step in job["steps"]:
                if type(step) is str:
                    continue
                if "restore_cache" in step and "keys" in step["restore_cache"]:
                    local_cache_keys.extend(step["restore_cache"]["keys"])
                    local_cache_enable = True

                if "save_cache" in step and "key" in step["save_cache"] and "paths" in step and step["save_cache"][
                    "key"] in local_cache_keys:
                    local_cache_paths.extend(step["save_cache"]["paths"])

                if "run" in step:
                    if type(step["run"]) is str:
                        command = step["run"]
                    else:
                        command = step["run"]["command"]

                    bazel_cmds.extend(self._parse_commands(command))

            for cmd in bazel_cmds:
                cmd.cores = cores
                cmd.local_cache = local_cache_enable
                if local_cache_enable:
                    cmd.local_cache_paths = local_cache_paths
                cc_cfg.build_commands.append(cmd)

        return cc_cfg

    def _convert_res_class_to_cores(self, job) -> int:
        res_class = job["resource_class"] if "resource_class" in job else "medium"

        if "docker" in job:
            return self._docker_or_linux_res_class_to_cores(res_class)
        if "macos" in job:
            return self._macos_res_class_to_cores(res_class)
        if "machine" in job:
            machine = job["machine"]
            if (type(machine) is bool and machine) or machine == "true":
                return self._docker_or_linux_res_class_to_cores(res_class)
            if "image" not in machine:
                logging.warning(f"unsupported executor {machine}, skip it")
                return None
            img = machine["image"]
            if img.startswith("windows-"):
                return self._windows_res_class_to_cores(res_class)
            return self._docker_or_linux_res_class_to_cores(res_class)
        if "executor" in job and "name" in job["executor"] and job["executor"]["name"].startswith("win"):
            size = job["executor"]["size"] if "size" in job["executor"] else "medium"
            return self._windows_res_class_to_cores(size)

        logging.warning(f"unsupported executor for job {job}")
        return None

    def _docker_or_linux_res_class_to_cores(self, res_class: str) -> int:
        res_class = res_class.removeprefix("arm.")

        match res_class:
            case "small":
                return 1
            case "medium":
                return 2
            case "medium+":
                return 3
            case "large":
                return 4
            case "xlarge":
                return 8
            case "2xlarge":
                return 16
            case "2xlarge":
                return 20
            case _:
                # maybe throw an error here?
                logging.warning(
                    f"unmatched resource class {res_class} for CircleCI linux or docker executor, fall back to use 2")
                return 2

    def _macos_res_class_to_cores(self, res_class: str) -> int:
        match res_class:
            case "medium" | "macos.x86.medium.gen2":
                return 4
            case "large":
                return 8
            case "macos.x86.metal.gen1":
                return 12
            case _:
                logging.warning(
                    f"unmatched resource class {res_class} for CircleCI macos executor, fall back to use 4")
                return 4

    def _windows_res_class_to_cores(self, res_class: str) -> int:
        res_class = res_class.removeprefix("windows.")

        match res_class:
            case "medium":
                return 4
            case "large":
                return 8
            case "xlarge":
                return 16
            case "2xlarge":
                return 32
            case _:
                logging.warning(
                    f"unmatched resource class {res_class} for CircleCI windows executor, fall back to use 4")
                return 4

    def ci_tool_type(self) -> str:
        return "circle_ci"

    def _ci_file_pattern(self) -> str:
        project_dir_regex_literal = re.escape(str.rstrip(self.project_dir, "/"))
        return rf"^{project_dir_regex_literal}/\.circleci/.*\.ya?ml"


class BuildkiteConfigParser(CIConfigParser):
    def __init__(self, project_dir: str):
        super().__init__(project_dir)
        # TODO add default build flags
        self.default_bazelci_build_flags = []
        self.default_bazelci_test_flags = []

    def parse_ci_file(self, ci_config_str: str) -> CIConfig:
        bb_cfg = CIConfig(self.ci_tool_type())
        ci_config_str.removeprefix("---\n")

        try:
            cfgs = yaml.full_load(ci_config_str)
        except yaml.error.YAMLError as e:
            logging.error(f"error when load yaml {ci_config_str}, reason {e}")
            return None

        if "steps" in cfgs:
            local_cache = False
            loca_cache_paths = []
            bazel_cmds = []
            for step in cfgs["steps"]:
                command_key = None
                if "command" in step:
                    command_key = "command"
                elif "commands" in step:
                    command_key = "commands"

                if command_key:
                    commands = [step[command_key]] if type(step[command_key]) is str else step[command_key]
                    for command in commands:
                        bazel_cmds.extend(self._parse_commands(command))
                if "plugins" in step:
                    plugins = step["plugins"]

                    plugin_dict = {}
                    # if the plugins are defined both by string literal and key-value maps in the yaml,
                    # the parsed structure in python is a list.
                    # We extract the plugin name and its properties from the list here.
                    if type(plugins) is list:
                        for plugin_wrap in plugins:
                            if type(plugin_wrap) is not dict:
                                continue
                            for plugin_name, plugin in plugin_wrap.items():
                                plugin_dict[plugin_name] = plugin
                    elif type(plugins) is dict:
                        plugin_dict = plugins

                    for plugin_name, plugin in plugin_dict.items():
                        if plugin_name.startswith("gencer/cache"):
                            local_cache = True
                            loca_cache_paths.extend(plugin["paths"])

            for cmd in bazel_cmds:
                cmd.local_cache = local_cache
                cmd.local_cache_paths = loca_cache_paths
                bb_cfg.build_commands.append(cmd)

            return bb_cfg

        # Bazel development team's DSL of Buildkite pipeline
        if "platform" in cfgs or "tasks" in cfgs:
            tasks = cfgs["platforms"].values() if "platform" in cfgs else cfgs["tasks"].values()
            for task in tasks:
                if "build_targets" in task:
                    build_flags = task["build_flags"] if "build_flags" in task else []
                    build_flags.extend(self.default_bazelci_build_flags)
                    bb_cfg.build_commands.append(BuildCommand("bazel", "build " + " ".join(build_flags)))
                if "test_targets" in task:
                    test_flags = task["test_flags"] if "test_flags" in task else []
                    test_flags.extend(self.default_bazelci_test_flags)
                    bb_cfg.build_commands.append(BuildCommand("bazel", "test " + " ".join(test_flags)))

            return bb_cfg


    def ci_tool_type(self) -> str:
        return "buildkite"

    def _ci_file_pattern(self) -> str:
        project_dir_regex_literal = re.escape(str.rstrip(self.project_dir, "/"))
        return rf"^{project_dir_regex_literal}/\.(buildkite|bazelci)/.*\.ya?ml$"
