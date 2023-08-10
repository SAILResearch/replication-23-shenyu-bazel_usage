import abc
import csv
import json
import logging
import multiprocessing
import os
import re
import subprocess
import urllib
import urllib.request
from datetime import date, timedelta

project_names_path = "bazel_projects.csv"
error_projects_path = "error_projects.log"
experiment_results_path = "./results.csv"

experiment_times = 5
cpu_quota = 2
# github_api_token = os.environ["GITHUB_API_TOKEN"]
github_api_token = "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"

elapsed_time_matcher = re.compile(r"Elapsed time: (\d+\.\d+)s")
critical_path_matcher = re.compile(r"Critical Path: (\d+\.\d+)s")
remote_cache_hit_matcher = re.compile(r" (\d+) remote cache hit,")
disk_cache_hit_matcher = re.compile(r" (\d+) disk cache hit,")
total_processes_matcher = re.compile(r"INFO: (\d+) processes:")

repository_path = "/data/repo"


class Experiment:
    def __init__(self, project, build_target="//:all", test_target="//..."):
        self.project = project
        self.build_target = build_target if build_target else "//:all"
        self.test_target = test_target if test_target else "//..."


class ExperimentResult:
    def __init__(self, project, subcommand, id, commit, cache_type, elapsed_time, critical_path, target, processes,
                 cache_hit, build_log, expr_id, status="success"):
        self.project = project
        self.subcommand = subcommand
        self.commit = commit
        self.id = id
        self.cache_type = cache_type
        self.elapsed_time = elapsed_time
        self.critical_path = critical_path
        self.processes = processes
        self.cache_hit = cache_hit
        self.target = target
        self.build_log = build_log
        self.expr_id = expr_id
        self.status = status


class DockerResource:
    def __init__(self):
        pass

    def __enter__(self):
        return self.create_res()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.remove_res()

    @abc.abstractmethod
    def create_res(self):
        return

    @abc.abstractmethod
    def remove_res(self):
        return


class GitRepository(DockerResource):
    def __init__(self, project_url):
        super().__init__()
        self.project_url = project_url

    def create_res(self):
        proc = subprocess.run(
            [
                f"docker run --rm -v /data/repo:/repo alpine/git clone {self.project_url} /repo/{os.path.splitext(os.path.basename(self.project_url))[0]}"],
            capture_output=True,
            text=True,
            shell=True)

        if proc.returncode != 0:
            raise Exception(f"error when cloning {self.project_url}, {proc.stderr}")

        return self

    def remove_res(self):
        pass

class DockerService(DockerResource):
    def create_res(self):
        # create a container running bazel remote cache server and setup the network configs.
        pass

    def remove_res(self):
        pass


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


def run_experiment_in_docker(L, i, project, target, subcommand, cache_type):
    try:
        logging.warning(f"running experiment for {project} {subcommand} {i} {target} {cache_type}")

        org, project = project.split("_", maxsplit=1)
        project_git_url = f"https://github.com/{org}/{project}.git"

        start_time = (date.today() + timedelta(days=-1000)).isoformat()
        end_time = date.today().isoformat()
        commits = retrieve_commits_within_time_range(org, project, start_time, end_time)

        # we only want first 100 commits
        commits = commits[:100]
        commits.reverse()

        prefix = f"{project}_{subcommand}_{i}"

        bazel_cmd_args = f"--jobs=256"
        if cache_type != "no_cache":
            if cache_type == "external":
                bazel_cmd_args += f" --repository_cache=/bazel-cache/{prefix}_repository-cache"
            if cache_type == "local":
                bazel_cmd_args += f" --repository_cache=/bazel-cache/{prefix}_repository-cache"
                bazel_cmd_args += f" --disk_cache=/bazel-cache/{prefix}_disk-cache"
            elif cache_type == "remote":
                bazel_cmd_args += f" --remote_cache=cache-server:port"
                bazel_cmd_args += f" --experimental_remote_downloader=grpc://cache-server:port"
                pass
        else:
            # for baseline builds, we build only one commit every 5 commits
            commits = commits[::5]

        for idx, commit in enumerate(commits):
            logging.warning(f"running experiment for {project} {subcommand} {i} {commit}")
            proc = subprocess.run(
                [f"docker run --privileged --cap-add SYS_ADMIN --device /dev/fuse --rm --cpus={cpu_quota} "
                 f"-e PROJECT_URL={project_git_url} "
                 f"-e TARGET={target} "
                 f"-e SUBCOMMAND={subcommand} "
                 f"-e SHA={commit} "
                 f"-e CMD_ARGS='{bazel_cmd_args}' "
                 f"-v /data/repo:/repo "
                 f"-v /cache_storage:/bazel-cache "
                 f"cizezsy/bazel-cache-experiment-runner"],
                capture_output=True,
                text=True,
                shell=True)

            build_log = proc.stderr

            elapsed_time = None
            if match := elapsed_time_matcher.search(build_log):
                elapsed_time = float(match.group(1))

            critical_path = None
            if match := critical_path_matcher.search(proc.stderr):
                critical_path = float(match.group(1))

            processes = None
            if match := total_processes_matcher.search(proc.stderr):
                processes = int(match.group(1))

            if cache_type == "remote":
                if match := remote_cache_hit_matcher.search(proc.stderr):
                    cache_hit = int(match.group(1))
                else:
                    cache_hit = 0
            else:
                if match := disk_cache_hit_matcher.search(proc.stderr):
                    cache_hit = int(match.group(1))
                else:
                    cache_hit = 0

            result = ExperimentResult(project, subcommand, idx, commit, cache_type, elapsed_time, critical_path,
                                      target, processes, cache_hit, build_log, i)

            if proc.returncode != 0:
                result.status = "failed"
            elif elapsed_time is None or critical_path is None or processes is None:
                result.status = "failed"

            L.append(result)

    except Exception as e:
        logging.error(f"error when running experiment for {project} {subcommand} {i} {commit}")
        logging.error(e)


def run_experiment(experiment, subcommand) -> [ExperimentResult]:
    results = []
    with multiprocessing.Manager() as manager:
        # for cache_type in ['local', 'remote', "external"]:
        for cache_type in ['local', 'external']:
            L = manager.list()
            # we run five experiments for each project
            pool = multiprocessing.Pool(int(10 / cpu_quota))

            target = experiment.test_target if subcommand == 'test' else experiment.build_target
            for i in range(experiment_times):
                pool.apply_async(run_experiment_in_docker,
                                 args=(L, i, experiment.project, target, subcommand, cache_type))

            pool.close()
            pool.join()
            results = results + list(L)
    return results


def run_experiments():
    if not os.path.exists(experiment_results_path):
        with open(experiment_results_path, "w") as experiment_results_file:
            experiment_results_file.write(
                "project,subcommand,id,commit,cache_type,elapsed_time,critical_path,target,processes,cache_hit,status\n")

    if not os.path.exists("build_logs"):
        os.mkdir("build_logs")

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

        org, project = experiment.project.split("_", maxsplit=1)

        with open(experiment_results_path, "a", buffering=1) as experiment_results_file, \
                open(error_projects_path, "a", buffering=1) as error_projects_file:
            logging.info(f"start running experiment for {experiment.project}")
            try:
                with GitRepository(f"https://github.com/{org}/{project}.git"):
                    results = run_experiment(experiment, "build")
                    # results.extend(run_experiment(experiment, "test", repo_cache_vol, local_cache_vol))

                    for result in results:
                        experiment_results_file.write(
                            f"{result.project},{result.subcommand},{result.id},{result.commit},{result.cache_type},{result.elapsed_time},{result.critical_path},{result.target},{result.processes},{result.cache_hit},{result.status}\n")
                        with(open(
                                f"build_logs/{experiment.project}_{result.subcommand}_{result.expr_id}_{result.commit}_{result.id}_{result.cache_type}_{result.status}.log",
                                "w")) as log_file:
                            log_file.write(result.build_log)
            except Exception as e:
                error_projects_file.write(f"error when run experiment for {experiment.project}, reason {e}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')
    run_experiments()
