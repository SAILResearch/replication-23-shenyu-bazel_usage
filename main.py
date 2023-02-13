import logging

import analysis.analyze
from projects import project

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')
    projects = project.retrieve_projects()

    # analysis.analyze.analyze("/Users/zhengshenyu/GolandProjects/bazel-testing-practices/repos")
