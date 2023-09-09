import logging
import re

from analysis.buildfile.analyzer import BuildFileAnalyzer
from analysis.ciconfig.analyzer import CIConfigAnalyzer
from analysis.project.analyzer import ProjectAnalyzer

bazel_job_regex = re.compile(r"--jobs=(\d+)")


def analyze(project_root):
    projects_base_names = ["bazel", "maven-large", "maven-small"]
    for base_dir_name in projects_base_names:
        logging.info(f"starting to analyze ci/cd configuration files in the folder {project_root}/{base_dir_name}")
        ci_config_analyzer = CIConfigAnalyzer()
        ci_config_analyzer.analyze_ci_configs(f"{project_root}/{base_dir_name}",
                                                          f"data/{base_dir_name}-projects")
        #
        # logging.info(f"starting to analyze build files in the folder {project_root}/{base_dir_name}")
        # build_config_analyzer = BuildFileAnalyzer()
        # build_config_analyzer.analyze_build_files(f"{project_root}/{base_dir_name}",
        #                                           f"data/{base_dir_name}-projects")
        #
        # logging.info(f"starting to analyze source files in the folder {project_root}/{base_dir_name}")
        # project_analyzer = ProjectAnalyzer()
        # project_analyzer.analyze_projects(f"{project_root}/{base_dir_name}",
        #                                   f"data/{base_dir_name}-projects")
