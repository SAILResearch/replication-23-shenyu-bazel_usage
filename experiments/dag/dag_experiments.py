import csv
import json
import logging
import os
import re
import subprocess
import urllib.request
import datetime

project_names_path = "bazel_projects.csv"
error_projects_path = "error_projects.log"
experiment_results_dir = "./results"

github_api_token = os.environ["GITHUB_API_TOKEN"]
history_commits = True


class Experiment:
    def __init__(self, project, build_target="//:all"):
        self.project = project
        self.build_target = build_target if build_target else "//:all"


def run_experiment(experiment):
    logging.info(f"start running experiment for {experiment.project}")

    org, project = experiment.project.split("_", maxsplit=1)
    if history_commits:
        end_time = datetime.datetime.strptime("2023-07-31", "%Y-%m-%d")
        start_time = end_time + datetime.timedelta(days=-1000)
        commits = retrieve_commits_within_time_range(org, project, start_time.isoformat(), end_time.isoformat())

        commits = commits[:100]
        commits.reverse()
        for commit in commits:
            cmd = f"docker run -v /dag-results:/results --rm -e PROJECT={project} -e ORG={org} -e BUILD_TARGET={experiment.build_target} -e COMMIT={commit} cizezsy/bazel-dag-experiment-runner"
            logging.info(f"running {cmd}")

            process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            stdout, stderr = process.communicate(cmd.encode('utf-8'))
            if process.returncode != 0:
                logging.error(
                    f"failed to run experiment for {project}, stdout: {stdout} error: {stderr}")
                raise Exception(f"failed to run experiment for {project}, stdout: {stdout} error: {stderr}")

            logging.info(
                f"finished running experiment for {experiment.project}, commit {commit}, stdout: {stdout} error: {stderr}")
    else:
        cmd = f"docker run -v /dag-results:/results --rm -e PROJECT={project} -e ORG={org} -e BUILD_TARGET={experiment.build_target} cizezsy/bazel-dag-experiment-runner"
        logging.info(f"running {cmd}")

        process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        stdout, stderr = process.communicate(cmd.encode('utf-8'))
        if process.returncode != 0:
            logging.error(
                f"failed to run experiment for {project}, stdout: {stdout} error: {stderr}")
            raise Exception(f"failed to run experiment for {project}, stdout: {stdout} error: {stderr}")

        logging.info(f"finished running experiment for {experiment.project}, stdout: {stdout} error: {stderr}")


def run_experiments():
    with open(project_names_path, "r+") as project_names_file:
        project_reader = csv.DictReader(project_names_file, delimiter=",")
        pending_experiments = [Experiment(row["project"], row["build_target"]) for row in
                               project_reader]

        for experiment in pending_experiments:
            run_experiment(experiment)


def retrieve_commits_within_time_range(org, project, start_time, end_time):
    request = urllib.request.Request(
        f"https://api.github.com/repos/{org}/{project}/commits?since={start_time}&until={end_time}&per_page=100",
        headers={"Authorization": f"token {github_api_token}"})

    with urllib.request.urlopen(request) as response:
        data = json.loads(response.read().decode())
        commits = [commit["sha"] for commit in data]

        link = response.getheader("Link")
        if link:
            last_page = int(re.search(r'page=(\d+)>; rel="last"', link).group(1))
            for page in range(2, last_page + 1):
                request = urllib.request.Request(
                    f"https://api.github.com/repos/{org}/{project}/commits?since={start_time}&until={end_time}&per_page=100&page={page}",
                    headers={"Authorization": f"token {github_api_token}"})

                with urllib.request.urlopen(request) as response:
                    data = json.loads(response.read().decode())
                    commits = commits + [commit["sha"] for commit in data]

    return commits


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')
    run_experiments()
