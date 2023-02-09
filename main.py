# --test_sharding_strategy=
import os
import urllib

import pandas as pd
import requests
from github import Github

import visualization.visualize
from buildfile.parser import BuildFileParser
from ciconfig.parser import CIConfigParser
from stats.stats import analyze_build_targets

g = Github("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")


def commit_count(org, project, token=None):
    """
    Return the number of commits to a project
    """
    token = token or os.environ.get('GITHUB_API_TOKEN')
    url = f'https://api.github.com/repos/{org}/{project}/commits'
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'token {token}',
    }
    params = {
        'per_page': 1,
    }
    resp = requests.request('GET', url, params=params, headers=headers)
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


def get_project_info(projects_dir):
    projects = []
    for project_dir in os.scandir(projects_dir):
        if project_dir.is_dir():
            org, name = project_dir.name.split("_", 1)
            projects.append((org, name))
    with open("data/projects.csv", "w") as project_file:
        project_file.write("project,language,size,stars,commits\n")
        for org, name in projects:
            print(f"Processing {org}_{name}")
            repo = g.get_repo(f"{org}/{name}")
            commit_cnt = commit_count(org, name, "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")
            project_file.write(f"{org}_{name},{repo.language},{repo.size},{repo.stargazers_count},{commit_cnt}\n")


class Project:
    def __init__(self, name, path):
        self.name = name
        self.path = path


def get_projects(projects_dir):
    project_info = pd.read_csv("data/projects.csv")
    projects = []
    for p in project_info.iterrows():
        if p[1]["commits"] < 100:
            continue
        projects.append(Project(p[1]["project"], os.path.join(projects_dir, p[1]["project"])))

    return projects


def parse_build_files(projects_dir: str) -> None:
    parser = BuildFileParser()

    with open("data/build_targets.csv", "w") as build_target_file, open("data/test_targets.csv",
                                                                        "w") as test_target_file:
        build_target_file.write("project,name,category\n")
        for project in get_projects(projects_dir):
            print(f"Processing {project.name}")
            build_file_stats = parser.parse(project.path)
            for rule in build_file_stats.rules:
                build_target_file.write(f"{project.name},{rule.name},{rule.category}\n")
                if rule.name.endswith("_test"):
                    test_target_file.write(
                        f"{project.name},{rule.name},{rule.category},"
                        f"{rule.attrs['shard_count'] if 'shard_count' in rule.attrs else ''},"
                        f"{rule.attrs['timeout'] if 'timeout' in rule.attrs else ''},"
                        f"{rule.attrs['size'] if 'size' in rule.attrs else ''},"
                        f"{rule.attrs['flaky'] if 'flaky' in rule.attrs else ''}\n")


def parse_ci_configs(projects_dir: str) -> None:
    parser = CIConfigParser()

    with open("data/bazel_commands.csv", "w") as command_file, open("data/project_tools.csv", "w") as project_tool_file:
        command_file.write("project,tool,command\n")
        project_tool_file.write("project,tool,use_bazel,use_test\n")
        for project in get_projects(projects_dir):
            print(f"Processing {project.name}")
            ci_config_stats = parser.parse(project.path)
            tool_test_enable_map = {"github-actions": False, "circleci": False, "buildkite": False}
            tools = set()
            for command in ci_config_stats.commands:
                command_file.write(f"{project.name},{command.tool},{command.command}\n")

                if "test" in command.command:
                    tool_test_enable_map[command.tool] = True
                tools.add(command.tool)
            for tool in ci_config_stats.tools:
                project_tool_file.write(f"{project.name},{tool},{tool in tools},{tool_test_enable_map[tool]}\n")


def main():
    projects_dir = "/Users/zhengshenyu/GolandProjects/bazel-testing-practices/repos"
    # get_project_info(projects_dir)
    # parse_build_files(projects_dir)
    # parse_ci_configs(projects_dir)
    # visualization.visualize.visualize_data()
    # visualization.visualize.visualize_project_tools()
    # visualization.visualize.visualize_project_tools_use_test()
    # visualization.visualize.draw_test_parallelization_usage()
    # visualization.visualize.draw_bazel_cache()
    # visualization.visualize.visualize_test_suite_types()
    # visualization.visualize.draw_flaky_tests()
    # visualization.visualize.visualize_build_rule_categories()
    analyze_build_targets()


if __name__ == "__main__":
    main()
