import re

from analysis.buildfile.analyzer import BuildFileAnalyzer
from analysis.ciconfig.analyzer import CIConfigAnalyzer

bazel_job_regex = re.compile(r"--jobs=(\d+)")


def analyze(project_root):
    projects_base_names = ["bazel", "maven-large", "maven-small"]
    for base_dir_name in projects_base_names:
        ci_config_analyzer = CIConfigAnalyzer()
        ci_config_analyzer.analyze_ci_configs(f"{project_root}/{base_dir_name}",
                                                          f"data/{base_dir_name}-projects")
        #
        # build_config_analyzer = BuildFileAnalyzer()
        # build_config_analyzer.analyze_build_files(f"{project_root}/{base_dir_name}",
        #                                           f"data/{base_dir_name}-projects")
