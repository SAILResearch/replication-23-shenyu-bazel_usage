import re

from analysis.buildfile.analyzer import BuildFileAnalyzer
from analysis.ciconfig.analyzer import CIConfigAnalyzer

bazel_job_regex = re.compile(r"--jobs=(\d+)")


def analyze(projects_base_dir):
    # ci_config_analyzer = CIConfigAnalyzer()
    # ci_config_analyzer.analyze_ci_configs(projects_base_dir, "data/bazel-projects")

    build_config_analyzer = BuildFileAnalyzer()
    build_config_analyzer.analyze_build_files(projects_base_dir, "data/bazel-projects")
