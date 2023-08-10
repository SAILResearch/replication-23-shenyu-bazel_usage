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
                 cores: int = 2, bazelci_project=False, invoker="ci", non_expended_arg_size=None,
                 expanded_arg_size=None):
        if local_cache_paths is None:
            local_cache_paths = []
        self.build_tool = build_tool
        self.raw_arguments = raw_arguments
        self.local_cache = local_cache
        self.local_cache_paths = local_cache_paths
        self.cores = cores
        self.bazelci_project = bazelci_project
        self.invoker = invoker
        self.non_expended_arg_size = non_expended_arg_size
        self.expanded_arg_size = expanded_arg_size


class CIConfig:
    def __init__(self, ci_tool: str):
        self.ci_tool = ci_tool
        self.build_commands = []


class CIConfigParser:
    total_analyzed_ci_file_count = 0
    total_analyzed_script_count = 0
    total_analyzed_commands_count = 0
    total_analyzed_makefile_count = 0

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.ci_file_pattern = re.compile(self._ci_file_pattern())
        self.sh_scripts = list(fileutils.scan_tree(project_dir, re.compile(r".*\.sh"), match_path=False))
        self.bazelrc_configs = self._extract_bazelrc_configs()
        self.make_targets = self._extract_make_targets()
        self.bazel_command_matcher = re.compile(".*bazel[\"w]? (.+)")
        self.bazelisk_command_matcher = re.compile(".*bazelisk (.+)")
        self.maven_command_matcher = re.compile(".*mvnw? (.*)")
        self.make_command_matcher = re.compile(".*make (.*)")

        self.analyzed_makefile = False
        self.analyzed_scripts = set()

    def parse(self) -> [CIConfig]:
        ci_configs = []
        for ci_file in self._scan_ci_files():
            with open(ci_file.path, "r") as f:
                ci_config_str = f.read()
            if not ci_config_str:
                continue
            CIConfigParser.total_analyzed_ci_file_count += 1
            ci_config = self.parse_ci_file(ci_config_str)
            if not ci_config:
                continue
            ci_configs.append(ci_config)
        CIConfigParser.total_analyzed_makefile_count += 1 if self.analyzed_makefile else 0
        CIConfigParser.total_analyzed_script_count += len(self.analyzed_scripts)
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

    def _extract_bazelrc_configs(self) -> dict[str:str]:
        bazelrc_configs = {}
        bazelrc_file_path = os.path.join(self.project_dir, ".bazelrc")

        if not fileutils.exists(bazelrc_file_path):
            return bazelrc_configs
        with open(bazelrc_file_path, "r") as bazelrc_file:
            for line in bazelrc_file:
                line = line.rstrip()
                if not line or line.lstrip().startswith("#"):
                    continue

                line = self._remove_comments(line).strip()
                if " " not in line:
                    continue

                (command, rest) = line.split(maxsplit=1)
                # we only concern about the build, test, coverage and run commands.
                if command.startswith(("build", "test", "coverage", "run")):
                    bazelrc_configs[command] = rest if command not in bazelrc_configs else bazelrc_configs[
                                                                                               command] + " " + rest

        # in .bazelrc, some commands inherit configs from other commands, so we resolve these inherited config here.
        inheritances_graph = {"test": ["build"], "run": ["build"], "coverage": ["test", "build"]}
        resolved_cfgs = {}
        for bazelrc_cmd, bazelrc_cfg in bazelrc_configs.items():
            for inherited_cmd, ancestors in inheritances_graph.items():
                if not bazelrc_cmd.startswith(inherited_cmd):
                    continue
                for ancestor in ancestors:
                    ancestor_cmd = bazelrc_cmd.replace(inherited_cmd, ancestor)
                    if ancestor_cmd not in bazelrc_configs:
                        continue
                    resolved_cfgs[bazelrc_cmd] = bazelrc_cfg + " " + bazelrc_configs[ancestor_cmd]

        for bazelrc_cmd, bazelrc_cfg in bazelrc_configs.items():
            if bazelrc_cmd not in resolved_cfgs:
                resolved_cfgs[bazelrc_cmd] = bazelrc_cfg

        heirs_graph = {"test": ["coverage"], "build": ["test", "run", "coverage"]}
        heirs_cfgs = {}
        for bazelrc_cmd, bazelrc_cfg in resolved_cfgs.items():
            for ancestor, heirs in heirs_graph.items():
                if not bazelrc_cmd.startswith(ancestor):
                    continue

                for heir in heirs:
                    heir_cmd = bazelrc_cmd.replace(ancestor, heir)
                    if heir_cmd in resolved_cfgs:
                        continue

                    if heir_cmd in heirs_cfgs:
                        heirs_cfgs[heir_cmd] = heirs_cfgs[heir_cmd] + " " + bazelrc_cfg
                    else:
                        heirs_cfgs[heir_cmd] = bazelrc_cfg
        for cmd, cfg in heirs_cfgs.items():
            resolved_cfgs[cmd] = cfg

        return resolved_cfgs

    def _extract_make_targets(self) -> dict[str:str]:
        make_targets = {}
        targets_relation_graph = {}
        makefile_path = os.path.join(self.project_dir, "Makefile")

        if not fileutils.exists(makefile_path):
            makefile_path = os.path.join(self.project_dir, "makefile")
            if not fileutils.exists(makefile_path):
                return make_targets

        makefile_command_matcher = re.compile(r"^(\S+):(.*)$")
        curr_cmd, curr_cmd_cfg = "", ""
        with open(makefile_path, "r") as makefile:
            for line in makefile:
                line = self._remove_comments(line)
                if not line.strip():
                    continue
                if line.startswith(".PHONY"):
                    continue

                if match := makefile_command_matcher.search(line):
                    if curr_cmd:
                        make_targets[curr_cmd] = curr_cmd_cfg
                        curr_cmd_cfg = ""
                    curr_cmd = match.group(1).strip()
                    dependencies = [t.strip() for t in match.group(2).split()]
                    if len(dependencies) > 0:
                        targets_relation_graph[curr_cmd] = dependencies
                    continue
                if line.startswith("\t") and curr_cmd:
                    curr_cmd_cfg += line.strip() + "\n"
            if curr_cmd:
                make_targets[curr_cmd] = curr_cmd_cfg

        resolved_make_targets = {}

        def resolve_dependencies_commands(build_target, visited):
            # return if the target is already resolved
            if build_target in resolved_make_targets:
                return resolved_make_targets[build_target]

            # avoid infinite loop
            if build_target in visited or build_target not in make_targets:
                return ""
            visited.add(build_target)

            dependency_commands = ""
            if build_target in targets_relation_graph:
                for dep in targets_relation_graph[build_target]:
                    dependency_commands += resolve_dependencies_commands(dep, visited) + "\n"
            resolved_make_targets[build_target] = dependency_commands + "\n" + make_targets[build_target]
            return resolved_make_targets[build_target]

        for target in make_targets:
            resolved_make_targets[target] = resolve_dependencies_commands(target, set()).strip()

        return resolved_make_targets

    def _scan_ci_files(self) -> [os.DirEntry]:
        possible_ci_files = []
        for f in fileutils.scan_tree(self.project_dir, self.ci_file_pattern):
            possible_ci_files.append(f)
        return possible_ci_files

    def _parse_commands(self, command, convert_make=True) -> [BuildCommand]:
        CIConfigParser.total_analyzed_commands_count += 1

        cmds = []
        command_matchers = {"maven": self.maven_command_matcher, "bazel": self.bazel_command_matcher,
                            "bazelisk": self.bazelisk_command_matcher, "make": self.make_command_matcher}
        for build_tool, matcher in command_matchers.items():
            if build_tool == "bazelisk":
                build_tool = "bazel"

            if matcher.search(command):
                for match in matcher.finditer(command):
                    if match.group().startswith("#"):
                        continue

                    cmd = BuildCommand(build_tool, match.group(1))
                    if not convert_make:
                        cmd.invoker = "make"
                    cmds.append(cmd)
            else:
                # if this step invoke a script, we look at that script to check if they run any Bazel command within
                for script_file in self.sh_scripts:
                    if script_file.name in command:
                        self.analyzed_scripts.add(script_file.path)
                        with open(script_file.path, "r") as f:
                            raw_script = f.read()
                            for match in matcher.finditer(raw_script):
                                if match.group().startswith("#"):
                                    continue
                                cmd = BuildCommand(build_tool, match.group(1), invoker="shell")
                                if not convert_make:
                                    cmd.invoker = "make"

                                cmds.append(cmd)
        if convert_make:
            cmds = self._convert_make_targets(cmds)

        cmds = self._filter_cmds(cmds)

        return cmds

    def _convert_make_targets(self, cmds):
        new_cmds = []

        # some projects use make to run other build tools, so we examine the make targets and convert them to other build tools
        for cmd in cmds:
            if cmd.build_tool != "make":
                new_cmds.append(cmd)
                continue

            self.analyzed_makefile = True

            make_targets = []
            make_args = cmd.raw_arguments.split()
            if not make_args:
                # if no target is specified, we check if "all" is defined in Makefile, is so, use it as the default target otherwise use "default"
                make_targets.append("all" if "all" in self.make_targets else "default")
            else:
                make_targets = [arg for arg in make_args if arg.startswith("-") is False]

            for make_target in make_targets:
                if make_target in self.make_targets:
                    new_cmds.extend(self._parse_commands(self.make_targets[make_target], convert_make=False))

        return new_cmds

    def _filter_cmds(self, cmds):
        filtered = []
        for cmd in cmds:
            if cmd.build_tool == "make":
                continue

            non_expanded_arg_size = len([arg_name for arg_name in cmd.raw_arguments.split() if
                                         arg_name.startswith("--") or arg_name.startswith("-")])
            cmd.non_expended_arg_size = non_expanded_arg_size

            if cmd.build_tool == "bazel":
                # we only analyze the following commands.
                # Also, we add a space to each command to try to improve the precision of the results.
                bazel_sub_cmds = ["build ", "test ", "run ", "coverage "]
                if not any(x in cmd.raw_arguments for x in bazel_sub_cmds):
                    continue

            cmd.raw_arguments = self._remove_comments(cmd.raw_arguments)
            if cmd.raw_arguments == "":
                continue

            cmd.raw_arguments = self._apply_default_bazelrc_configs(cmd.raw_arguments)

            expanded_arg_size = len([arg_name for arg_name in cmd.raw_arguments.split() if
                                     arg_name.startswith("--") or arg_name.startswith("-")])
            cmd.expanded_arg_size = expanded_arg_size

            filtered.append(cmd)
        return filtered

    def _remove_comments(self, x: str) -> str:
        comments_matcher = re.compile(r"(#.*)")
        if match := comments_matcher.search(x):
            x = x.replace(match.group(1), "")
        return x

    def _apply_default_bazelrc_configs(self, raw_arguments: str) -> str:
        bazel_group_config_regex = re.compile(r"--config=(.+)")

        group_config_tag = ""
        if match := bazel_group_config_regex.search(raw_arguments):
            group_config_tag = match.group(1)

        for bazelrc_cmd, bazelrc_cfg in self.bazelrc_configs.items():
            # command with group tag - like build:ci
            if ":" in bazelrc_cmd:
                if not bazelrc_cmd.endswith(f":{group_config_tag}"):
                    continue

                segs = bazelrc_cmd.split(":", maxsplit=1)
                if len(segs) < 2:
                    continue
                cmd = segs[0]
            else:
                cmd = bazelrc_cmd

            if f"{cmd} " in raw_arguments or cmd == "common":
                raw_arguments = f"{raw_arguments} {bazelrc_cfg}"

        return raw_arguments


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
            # if this is an action file
            if "runs" in cfgs and "steps" in cfgs["runs"]:
                self._analyze_job(2, gha_cfg, cfgs["runs"])
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
                if match := self.runner_cores_pattern.search(runner):
                    cores = int(match.group(1))

            self._analyze_job(cores, gha_cfg, job)

        return gha_cfg

    def _analyze_job(self, cores, gha_cfg, job):
        local_cache_enable = False
        local_cache_paths = []
        local_maven_cache = False
        for step in job["steps"]:
            # this step uses cache action
            if "uses" in step:
                if step["uses"].startswith("actions/cache"):
                    if "with" in step and "path" in step["with"]:
                        local_cache_enable = True
                        local_cache_paths = step["with"]["path"].splitlines()
                    continue

                if step["uses"].startswith("github/codeql-action/autobuild"):
                    if "strategy" in job and "matrix" in job["strategy"] and "language" in job["strategy"]["matrix"]:
                        if "java" in job["strategy"]["matrix"]["language"]:
                            for cmd in self._parse_commands(
                                    'mvn clean package -f "pom.xml" -B -V -e -Dfindbugs.skip -Dcheckstyle.skip -Dpmd.skip=true -Dspotbugs.skip -Denforcer.skip -Dmaven.javadoc.skip -DskipTests -Dmaven.test.skip.exec -Dlicense.skip=true -Drat.skip=true -Dspotless.check.skip=true'):
                                cmd.cores = cores
                                cmd.local_cache = (
                                                              cmd.build_tool == "maven" and local_maven_cache) or local_cache_enable
                                cmd.local_cache_paths = local_cache_paths
                                gha_cfg.build_commands.append(cmd)

                if step["uses"].startswith("actions/setup-java"):
                    if "with" in step and "cache" in step["with"] and step["with"]["cache"] == "maven":
                        local_maven_cache = True
                        local_cache_paths.append("~/.m2/repository")

            # this step runs command
            if "run" in step:
                command = step["run"]
                for cmd in self._parse_commands(command):
                    cmd.cores = cores
                    cmd.local_cache = (cmd.build_tool == "maven" and local_maven_cache) or local_cache_enable
                    cmd.local_cache_paths = local_cache_paths
                    gha_cfg.build_commands.append(cmd)

    def ci_tool_type(self) -> str:
        return "github_actions"

    def _ci_file_pattern(self) -> str:
        project_dir_regex_literal = re.escape(str.rstrip(self.project_dir, "/"))
        return rf"^{project_dir_regex_literal}/\.github/(workflows|actions)/.*\.ya?ml$"


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

        custom_executors = {}
        if "executors" in cfgs:
            for executor_name, executor_cfg in cfgs["executors"].items():
                executor_cores = self._convert_res_class_to_cores(executor_cfg)
                if executor_cores:
                    custom_executors[executor_name] = executor_cores

        reusable_steps = {}
        if "commands" in cfgs:
            for cmd_name, cmd_cfg in cfgs["commands"].items():
                if "steps" in cmd_cfg:
                    reusable_steps[cmd_name] = cmd_cfg["steps"]

        for job in cfgs["jobs"].values():
            if "steps" not in job:
                continue

            if "executor" in job and type(job["executor"]) is str and job["executor"] in custom_executors:
                cores = custom_executors[job["executor"]]
            else:
                cores = self._convert_res_class_to_cores(job)

            if not cores:
                cores = -1

            local_cache_enable = False
            local_cache_paths = []
            local_cache_keys = []
            bazel_cmds = []

            for step in job["steps"]:
                # the step may be expanded to multiple steps
                steps_to_run = []

                if type(step) is str:
                    if step in reusable_steps:
                        steps_to_run.extend(reusable_steps[step])
                    else:
                        continue

                # the step may use a reusable step with parameters, in this case, the step is a dict with one key.
                if type(step) is dict and len(step) == 1 and list(step.keys())[0] in reusable_steps:
                    reuse_steps_name = list(step.keys())[0]
                    steps_to_run.extend(reusable_steps[reuse_steps_name])
                    if type(step[reuse_steps_name]) is dict:
                        for _, value in step[reuse_steps_name].items():
                            if type(value) is str:
                                bazel_cmds.extend(self._parse_commands(value))

                else:
                    steps_to_run.append(step)

                for step in steps_to_run:
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
        return rf"^{project_dir_regex_literal}/\.circleci/.*\.ya?ml$"


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
        if "platforms" in cfgs or "tasks" in cfgs:
            tasks = cfgs["platforms"].values() if "platforms" in cfgs else cfgs["tasks"].values()
            for task in tasks:
                if "build_targets" in task:
                    targets = task["build_targets"]

                    build_flags = task["build_flags"] if "build_flags" in task else []
                    build_flags.extend(self.default_bazelci_build_flags)
                    non_expanded_arg_size = len(build_flags)

                    raw_arguments = self._apply_default_bazelrc_configs(
                        "build " + " ".join(build_flags) + " " + " ".join(targets))

                    expanded_arg_size = len(
                        [arg for arg in raw_arguments if arg.startswith("--") or arg.startswith("-")])
                    bb_cfg.build_commands.append(
                        BuildCommand("bazel", raw_arguments, bazelci_project=True,
                                     non_expended_arg_size=non_expanded_arg_size, expanded_arg_size=expanded_arg_size))
                if "test_targets" in task:
                    targets = task["test_targets"]

                    test_flags = task["test_flags"] if "test_flags" in task else []
                    test_flags.extend(self.default_bazelci_test_flags)
                    non_expanded_arg_size = len(test_flags)

                    raw_arguments = self._apply_default_bazelrc_configs(
                        "test " + " ".join(test_flags) + " " + " ".join(targets))
                    expanded_arg_size = len(
                        [arg for arg in raw_arguments if arg.startswith("--") or arg.startswith("-")])

                    bb_cfg.build_commands.append(
                        BuildCommand("bazel", raw_arguments, bazelci_project=True,
                                     non_expended_arg_size=non_expanded_arg_size, expanded_arg_size=expanded_arg_size))

            return bb_cfg

    def ci_tool_type(self) -> str:
        return "buildkite"

    def _ci_file_pattern(self) -> str:
        project_dir_regex_literal = re.escape(str.rstrip(self.project_dir, "/"))
        return rf"^{project_dir_regex_literal}/\.(buildkite|bazelci)/.*\.ya?ml$"


class TravisCIConfigParser(CIConfigParser):
    def __init__(self, project_dir: str):
        super().__init__(project_dir)

    def parse_ci_file(self, ci_config_str: str) -> CIConfig:
        tc_cfg = CIConfig(self.ci_tool_type())

        try:
            cfgs = yaml.full_load(ci_config_str)
        except yaml.error.YAMLError as e:
            logging.error(f"error when load yaml {ci_config_str}, reason {e}")
            return None

        local_cache = False
        local_cache_paths = []
        build_cmds = []
        if "script" in cfgs:
            build_cmds.extend(self._parse_script(cfgs["script"]))

        if "jobs" in cfgs or "matrix" in cfgs:
            jobs = cfgs["jobs"] if "jobs" in cfgs else cfgs["matrix"]
            stages = []
            if "include" in jobs:
                stages = jobs["include"]
            elif "exclude" in jobs:
                stages = stages.extend(jobs["exclude"]) if stages else jobs["exclude"]

            for stage in stages:
                if "script" in stage:
                    build_cmds.extend(self._parse_script(stage["script"]))

        if "install" in cfgs and (type(cfgs["install"]) is list or type(cfgs["install"]) is str):
            install_cmds = cfgs["install"]
            if type(install_cmds) is str:
                install_cmds = [install_cmds]
            for cmd in install_cmds:
                if type(cmd) is not str:
                    continue
                build_cmds.extend(self._parse_commands(cmd))

        if "deploy" in cfgs:
            deployments = []
            if type(cfgs["deploy"]) is list:
                deployments.extend(cfgs["deploy"])
            elif type(cfgs["deploy"]) is dict:
                deployments.append(cfgs["deploy"])

            for deployment in deployments:
                if "script" in deployment:
                    build_cmds.extend(self._parse_script(deployment["script"]))

        if "cache" in cfgs:
            cache = cfgs["cache"]
            if "directories" in cache:
                local_cache = True
                local_cache_paths = cache["directories"]

        if "language" in cfgs and cfgs["language"] == "java":
            if "script" not in cfgs and "jobs" not in cfgs:
                # if not script is defined, and it is a java project, travis will run mvn test -B by default
                build_cmds.extend(self._parse_commands("mvn test -B"))

            if "install" not in cfgs:
                # if not install is defined, and it is a java project, travis will run mvn install -DskipTests=true -Dmaven.javadoc.skip=true -B -V by default
                build_cmds.extend(self._parse_commands("mvn install -DskipTests=true -Dmaven.javadoc.skip=true -B -V"))

        for cmd in build_cmds:
            cmd.local_cache = local_cache
            cmd.local_cache_paths = local_cache_paths
            tc_cfg.build_commands.append(cmd)

        return tc_cfg

    def _parse_script(self, script) -> list[BuildCommand]:
        if type(script) is str:
            return self._parse_commands(script)
        elif type(script) is list:
            bazel_cmds = []
            for cmd in script:
                bazel_cmds.extend(self._parse_commands(cmd))
            return bazel_cmds
        else:
            return []

    def ci_tool_type(self) -> str:
        return "travis_ci"

    def _ci_file_pattern(self) -> str:
        project_dir_regex_literal = re.escape(str.rstrip(self.project_dir, "/"))
        return rf"^{project_dir_regex_literal}/\.travis.ya?ml$"
