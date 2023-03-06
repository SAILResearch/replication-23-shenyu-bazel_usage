import logging
import os
import re

from analysis.ciconfig.parser import GitHubActionConfigParser, CIConfig, CircleCIConfigParser, \
    BuildkiteConfigParser


class CIConfigAnalyzer:
    def __init__(self):
        pass

    def _analyze_project_ci_configs(self, project_dir: str) -> [CIConfig]:
        ci_config_parsers = [GitHubActionConfigParser(project_dir), CircleCIConfigParser(project_dir),
                             BuildkiteConfigParser(project_dir)]
        ci_configs = []
        for p in ci_config_parsers:
            ci_configs.extend(p.parse())
        return ci_configs

    def analyze_ci_configs(self, project_base_dir: str, output_dir: str):
        bazel_job_regex = re.compile(r"--jobs=(\d+)")
        maven_parallel_options_regex = re.compile(r"-T (\d+(\.\d+)?C?)")

        with open(f"{output_dir}/build_commands.csv", "w") as bzl_cmd_file:
            # we use '#' as the separator of the csv file
            bzl_cmd_file.write("project#ci_tool#build_tool#raw_arguments#local_cache#remote_cache#parallelism#cores\n")
            for entry in os.scandir(project_base_dir):
                if not entry.is_dir():
                    continue

                logging.info(f"starting to analyze ci/cd configurations for project {entry.name}")
                for ci_config in self._analyze_project_ci_configs(entry.path):
                    for cmd in ci_config.build_commands:
                        use_local_cache = False
                        use_remote_cache = False

                        cores = cmd.cores
                        parallelism = -1
                        if cmd.build_tool == "bazel":
                            if cmd.local_cache and any(path.endswith(".cache/bazel") for path in cmd.local_cache_paths):
                                use_local_cache = True
                            if "--remote_cache" in cmd.raw_arguments:
                                use_remote_cache = True
                            if match := bazel_job_regex.match(cmd.raw_arguments):
                                parallelism = float(match.group(1))
                        elif cmd.build_tool == "maven":
                            # TODO, We need to examine the pom file to know if the maven project uses cache.
                            if match := maven_parallel_options_regex.match(cmd.raw_arguments):
                                threads = match.group(1)
                                if threads.endswith("C"):
                                    parallelism = float(threads.removesuffix("C")) * cores
                                else:
                                    parallelism = threads
                        else:
                            logging.error(f"unsupported build tool {cmd.build_tool} in project {entry.name}, skip it")
                            continue
                        # we use '#' as the separator of the csv file
                        bzl_cmd_file.write(
                            f"{entry.name}#{ci_config.ci_tool}#{cmd.build_tool}#{cmd.raw_arguments}#{use_local_cache}#{use_remote_cache}#{parallelism}#{cmd.cores}\n")


if __name__ == "__main__":
    results = CIConfigAnalyzer()._analyze_project_ci_configs(
        "/Users/zhengshenyu/PycharmProjects/how-do-developers-use-bazel/repos/bazel/apache_rocketmq")
    print(results)
