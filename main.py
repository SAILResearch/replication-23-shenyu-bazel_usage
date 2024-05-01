import logging

from analysis import analysis
from projects import project
from visualization.preprocess import preprocess_data
from visualization.visualize import visualize_data

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')
    # projects = project.retrieve_projects()
    # project.retrieve_projects_rebuttal_ver()

    # analysis.analyze("./repos")
    # preprocess_data("./data")
    visualize_data("./data")
