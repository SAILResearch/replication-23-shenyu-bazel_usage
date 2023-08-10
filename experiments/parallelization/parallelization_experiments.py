import csv
import logging
import multiprocessing
import os
import re
import subprocess

project_names_path = "bazel_projects.csv"
error_projects_path = "error_projects.log"
experiment_results_path = "./results.csv"

experiment_times = 10
parallelization_configs = [1, 2]

elapsed_time_matcher = re.compile(r"Elapsed time: (\d+\.\d+)s")
critical_path_matcher = re.compile(r"Critical Path: (\d+\.\d+)s")


class Experiment:
    def __init__(self, project, build_target="//:all", test_target="//..."):
        self.project = project
        self.build_target = build_target if build_target else "//:all"
        self.test_target = test_target if test_target else "//..."


class ExperimentResult:
    def __init__(self, project, subcommand, elapsed_time, critical_path, parallelism, target):
        self.project = project
        self.subcommand = subcommand
        self.elapsed_time = elapsed_time
        self.critical_path = critical_path
        self.parallelism = parallelism
        self.target = target


def run_experiment_in_docker(L, i, project_git_url, target, subcommand, parallelism):
    logging.info(
        f"Running {i} docker run --cpus={parallelism} -e PROJECT_URL={project_git_url} -e TARGET={target} -e SUBCOMMAND={subcommand} -e CMD_ARGS='--jobs=256' cizezsy/bazel-parallellization-experiment-runner")
    proc = subprocess.run(
        [
            f"docker run --rm --cpus={parallelism} -e PROJECT_URL={project_git_url} -e TARGET={target} -e SUBCOMMAND={subcommand} -e CMD_ARGS='--jobs=256' cizezsy/bazel-parallellization-experiment-runner"],
        capture_output=True,
        text=True,
        shell=True
    )

    logging.info(f"stdout: {proc.stderr}")

    if proc.returncode != 0:
        raise Exception(f"{proc.stderr}")

    if match := elapsed_time_matcher.search(proc.stderr):
        elapsed_time = float(match.group(1))
    else:
        raise Exception(f"unable to parse elapsed time from {proc.stderr}")

    if match := critical_path_matcher.search(proc.stderr):
        critical_path = float(match.group(1))
    else:
        raise Exception(f"unable to parse critical path from {proc.stderr}")

    L.append(ExperimentResult(project_git_url, subcommand, elapsed_time, critical_path, parallelism, target))


def run_experiment(experiment, subcommand) -> [ExperimentResult]:
    org, project = experiment.project.split("_", maxsplit=1)
    project_git_url = f"https://github.com/{org}/{project}.git"

    results = []
    with multiprocessing.Manager() as manager:
        for parallelism in parallelization_configs:
            L = manager.list()
            pool = multiprocessing.Pool(int(16 / parallelism))

            for i in range(experiment_times):
                target = experiment.test_target if subcommand == 'test' else experiment.build_target
                pool.apply_async(run_experiment_in_docker,
                                 args=(L, i, project_git_url, target, subcommand, parallelism))

            pool.close()
            pool.join()
            results = results + list(L)

    return results


def run_experiments():
    if not os.path.exists(experiment_results_path):
        with open(experiment_results_path, "w") as experiment_results_file:
            experiment_results_file.write("project,subcommand,elapsed_time,critical_path,parallelism,target\n")

    with open(project_names_path, "r+") as project_names_file:
        project_reader = csv.DictReader(project_names_file, delimiter=",")
        pending_experiments = [Experiment(row["project"], row["build_target"], row["test_target"]) for row in
                               project_reader]

        with open(experiment_results_path, "r") as experiment_results_file:
            results_reader = csv.DictReader(experiment_results_file, delimiter=",")
            completed_projects = set([row["project"] for row in results_reader])

        for experiment in pending_experiments:
            if experiment.project in completed_projects:
                continue

            with open(experiment_results_path, "a", buffering=1) as experiment_results_file, \
                    open(error_projects_path, "a", buffering=1) as error_projects_file:
                logging.info(f"start running experiment for {experiment.project}")
                try:
                    results = run_experiment(experiment, "build")
                    for result in results:
                        experiment_results_file.write(
                            f"{result.project},{result.subcommand},{result.elapsed_time},{result.critical_path},{result.parallelism},{result.target}\n")
                    results = run_experiment(experiment, "test")
                    for result in results:
                        experiment_results_file.write(
                            f"{result.project},{result.subcommand},{result.elapsed_time},{result.critical_path},{result.parallelism},{result.target}\n")
                except Exception as e:
                    error_projects_file.write(f"error when run experiment for {experiment.project}, reason {e}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')
    run_experiments()