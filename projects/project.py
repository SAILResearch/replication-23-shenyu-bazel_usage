import json
import logging
import os
import shutil
import subprocess
import time
import urllib
from datetime import datetime

import pandas as pd
import numpy as np
import requests
from git import Repo
from github import Github
from github import GithubException

from utils import fileutils


class Project:
    def __init__(self, org: str, name: str):
        self.org = org
        self.name = name
        self.meta = {}


def search_projects(file_name_pattern: str, build_tool: str) -> [Project]:
    logging.info(f"starting to search project for build tool {build_tool} with file name pattern {file_name_pattern}")
    proc = subprocess.run(
        [f"src search -json 'select:repo (file:{file_name_pattern}) count:10000'"],
        capture_output=True,
        text=True,
        shell=True)

    if proc.returncode != 0:
        raise Exception(f"error when search project for build tool {build_tool}, reason {proc.stderr}")

    results = json.loads(proc.stdout)
    if "Results" not in results:
        raise Exception(f"unable to parse search results {proc.stdout}, expected 'Results' in the returned results")

    projects = []
    for repo in results["Results"]:
        repo_name = repo["name"]
        # we only want GitHub projects
        if not repo_name.startswith("github.com"):
            continue

        paths = repo_name.removeprefix("github.com/").split("/")

        projects.append(Project(paths[0], paths[1]))
    logging.info(f"found {len(projects)} projects")
    return projects


def filter_projects(projects: [Project]) -> [Project]:
    with open(".github_token", "r") as f:
        token = f.read()
    g = Github(token)

    filtered = []
    for p in projects:
        logging.info(f"checking the stars and commits of project {p.org}/{p.name}")
        try:
            time.sleep(3)
            repo = g.get_repo(f"{p.org}/{p.name}")
        except GithubException as e:
            logging.warning(f"error when get repository info for project {p.org}/{p.name}, reason: {e}")
            continue
        if repo.stargazers_count < 100:
            continue
        commits = commit_count(p.org, p.name, token)
        if commits < 100:
            continue
        p.meta["stars"] = repo.stargazers_count
        p.meta["commits"] = commits
        filtered.append(p)
    return filtered


def get_file_language(project_dir: str, file_name: str) -> str:
    path = os.path.join(project_dir, file_name)

    proc = subprocess.run(
        [f"github-linguist {path} --json"],
        capture_output=True,
        text=True,
        shell=True)
    if proc.returncode != 0:
        raise Exception(
            f"error when examining the build file {file_name} for project {project_dir}, reason {proc.stderr}")
    results = json.loads(proc.stdout)
    if path not in results:
        raise Exception(f"unable to examine the build file {file_name}, the output of linguist {proc.stdout}")

    return results[path]["language"]


def remove_non_bazel_projects(project_base: dir):
    for entry in os.scandir(project_base):
        if not entry.is_dir():
            continue

        if fileutils.exists(os.path.join(entry.path, "BUILD")) and get_file_language(entry.path, "BUILD") == "Starlark":
            continue

        if fileutils.exists(os.path.join(entry.path, "BUILD.bazel")) and get_file_language(entry.path,
                                                                                           "BUILD.bazel") == "Starlark":
            continue

        logging.info(f"{entry.path} is not a valid Bazel projects, will delete it")
        shutil.rmtree(entry.path)


def commit_count(org: str, project: str, token: str) -> int:
    url = f'https://api.github.com/repos/{org}/{project}/commits'
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'token {token}',
    }
    params = {
        'per_page': 1,
    }
    resp = requests.request('GET', url, params=params, headers=headers, timeout=30)
    if (resp.status_code // 100) != 2:
        raise Exception(f'invalid github response: {resp.content}')
    # check the resp count, just in case there are 0 commits
    count = len(resp.json())
    last_page = resp.links.get('last')
    # if there are no more pages, the count must be 0 or 1
    if last_page:
        # extract the query string from the last page url
        qs = urllib.parse.urlparse(last_page['url']).query
        # extract the page number from the query string
        count = int(dict(urllib.parse.parse_qsl(qs))['page'])
    return count


def clone_projects(base_dir: str, projects: [Project]):
    for p in projects:
        logging.info(f"cloning repository {p.org}/{p.name}")
        # shallow clone the repository
        data_collection_date = datetime.fromisoformat("2023-02-23T00:00:00Z")
        Repo.clone_from(f"https://github.com/{p.org}/{p.name}.git", f"{base_dir}/{p.org}_{p.name}")

        proc = subprocess.run(
            [f"git -C {base_dir}/{p.org}_{p.name} checkout $(git -C {base_dir}/{p.org}_{p.name} rev-list -n 1 --before='2023-02-23' HEAD)"],
            capture_output=True,
            text=True,
            shell=True)

        if proc.returncode != 0:
            logging.warning(f"error when checking out the repository {p.org}/{p.name}, reason {proc.stderr}")
            continue

        logging.info(f"cloned repository {p.org}/{p.name}")



def retrieve_bazel_projects(project_base_dir: str):
    projects = search_projects("^BUILD(.bazel)?$", "bazel")
    projects = filter_projects(projects)
    clone_projects("./repos/bazel", projects)
    remove_non_bazel_projects("/Users/zhengshenyu/PycharmProjects/how-do-developers-use-bazel/repos/bazel")

    filtered_projects = set()
    for f in os.scandir(project_base_dir):
        if not f.is_dir():
            continue
        filtered_projects.add(f.name)

    with open("data/bazel_projects.csv", "w") as f:
        f.write("project,stars,commits,build_tool\n")
        for p in projects:
            if f"{p.org}_{p.name}" not in filtered_projects:
                continue
            f.write(f"{p.org}_{p.name},{p.meta['stars']},{p.meta['commits']},bazel\n")


def retrieve_maven_projects(project_base_dir: str):
    projects = search_projects("^pom.xml$", "maven")
    projects = filter_projects(projects)
    with open("data/maven_projects.csv", "w") as f:
        f.write("project,stars,commits,build_tool\n")
        for p in projects:
            f.write(f"{p.org}_{p.name},{p.meta['stars']},{p.meta['commits']},maven\n")


def sample_maven_projects():
    mvn_projects = pd.read_csv("data/maven_projects.csv")
    bazel_projects = pd.read_csv("data/bazel_projects.csv")

    mvn_projects = mvn_projects.sort_values(by="commits").reset_index(drop=True)
    subsets = np.array_split(mvn_projects, 2)
    small_maven_projects = subsets[0].iloc[np.random.choice(len(subsets[0]), len(bazel_projects), replace=False)]
    large_maven_projects = subsets[1].iloc[np.random.choice(len(subsets[1]), len(bazel_projects), replace=False)]
    small_maven_projects.to_csv("data/small_maven_projects_subset.csv", index=False)
    large_maven_projects.to_csv("data/large_maven_projects_subset.csv", index=False)

    clone_maven_projects("./repos/maven-small", small_maven_projects)
    clone_maven_projects("./repos/maven-large", large_maven_projects)


def clone_maven_projects(base_dir: str, projects_raw: pd.DataFrame):
    projects = []
    for _, row in projects_raw.iterrows():
        segs = row["project"].split("_")
        projects.append(Project(segs[0], "_".join(segs[1:])))

    clone_projects(base_dir, projects)


def retrieve_projects():
    project_base_dir = "repos/"
    retrieve_bazel_projects(f"{project_base_dir}/bazel")
    retrieve_maven_projects(f"{project_base_dir}/maven")
    sample_maven_projects()


def sample_maven_projects_rebuttal_ver():
    # bazel_projects = pd.read_csv("data/bazel_projects_manually_examined.csv")
    mvn_projects = pd.read_csv("data/maven_projects.csv")
    print(f"median commits: {np.median(mvn_projects['commits'])}")
    #
    # # some projects were removed from GitHub since the last data collection
    # projects_to_be_removed = ["liuht777_Taroco", "microsoft_Availability-Monitor-for-Kafka", "Jerusalem01_tesco-mall"]
    # pattern = '|'.join(projects_to_be_removed)
    # mvn_projects = mvn_projects.loc[~mvn_projects["project"].str.contains(pattern, case=False)]
    #
    # mvn_projects = mvn_projects.sort_values(by="commits").reset_index(drop=True)
    #
    # bazel_commits = bazel_projects["commits"]
    # medians = []
    # for i in range(100):
    #     sampled_mvn_projects = mvn_projects.sample(n=len(bazel_projects), random_state=i+3)
    #     maven_commits = sampled_mvn_projects["commits"]
    #
    #     all_commits = np.concatenate([bazel_commits, maven_commits])
    #     medians.append(np.median(all_commits))
    #
    # median = np.median(medians)
    # print(f"median: {median}")
    #
    # small_maven_projects = mvn_projects[mvn_projects["commits"] < median]
    # large_maven_projects = mvn_projects[mvn_projects["commits"] >= median]
    #
    #
    # small_maven_projects = small_maven_projects.sample(n=len(bazel_projects))
    # large_maven_projects = large_maven_projects.sample(n=len(bazel_projects))
    #
    # small_maven_projects.to_csv("data/small_maven_projects_subset_rebuttal_ver.csv", index=False)
    # large_maven_projects.to_csv("data/large_maven_projects_subset_rebuttal_ver.csv", index=False)

    # small_maven_projects = pd.read_csv("data/small_maven_projects_subset_rebuttal_ver.csv")
    # large_maven_projects = pd.read_csv("data/large_maven_projects_subset_rebuttal_ver.csv")

    # clone_maven_projects("./repos/maven-small_rebuttal_ver", small_maven_projects)
    # clone_maven_projects("./repos/maven-large_rebuttal_ver", large_maven_projects)

    # bazel_projects = pd.read_csv("data/bazel_projects_manually_examined.csv")
    # clone_maven_projects("./repos/bazel", bazel_projects)



def retrieve_projects_rebuttal_ver():
    project_base_dir = "repos/"
    sample_maven_projects_rebuttal_ver()
