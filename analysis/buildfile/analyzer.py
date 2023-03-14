import logging
import os

from analysis.buildfile.parser import BuildFileConfig, BazelBuildFileParser, MavenBuildFileParser


class BuildFileAnalyzer:
    def __init__(self):
        pass

    def _analyze_project_build_files(self, project_dir: str):
        build_file_parsers = [BazelBuildFileParser(project_dir), MavenBuildFileParser(project_dir)]
        build_configs = []
        for p in build_file_parsers:
            build_configs.extend(p.parse())
        if len(build_configs) == 0:
            logging.warning(f"no build rules found for project {project_dir}")
        return build_configs

    def analyze_build_files(self, project_base_dir: str, output_dir: str):
        with open(f"{output_dir}/build_rules.csv", "w") as build_config_file:
            build_config_file.write("project,build_tool,name,category\n")
            for entry in os.scandir(project_base_dir):
                if not entry.is_dir():
                    continue

                logging.info(f"starting to analyze ci/cd configurations for project {entry.name}")
                for build_config in self._analyze_project_build_files(entry.path):
                    for rule in build_config.rules:
                        build_config_file.write(f"{entry.name},{build_config.build_tool},{rule.name},{rule.category}\n")
