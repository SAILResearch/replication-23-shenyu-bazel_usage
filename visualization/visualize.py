import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from visualization.preprocess import *


def visualize_data(data_dir: str):
    sns.set_style("whitegrid")

    data_dir = os.path.join(data_dir, "processed")
    visualize_ci_tools(data_dir)
    visualize_parallelization_usage(data_dir)
    visualize_cache_usage(data_dir)
    visualize_test_usage(data_dir)


def visualize_ci_tools(data_dir: str):
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 10), tight_layout=True, sharey=True)
    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    idx = 0
    for parent_dir_name, correspondent_ci_tool in parent_dir_names.items():
        df = pd.read_csv(os.path.join(data_dir, f"{parent_dir_name}-build_tools.csv")).drop(
            columns=["subcommands"]).drop_duplicates()

        build_tool_usage = pd.DataFrame({"Local": [0, 0, 0, 0, 0, 0, 0], "CI": [0, 0, 0, 0, 0, 0, 0],
                                         "CI/CD Services": ["github_actions", "circle_ci", "buildkite",
                                                            "buildkite+github_actions", "buildkite+circle_ci",
                                                            "circle_ci+github_actions",
                                                            "buildkite+circle_ci+github_actions"]})

        for project in df["project"].unique():
            ci_tools = sorted(df[df["project"] == project]["ci_tool"].unique())
            all_true = True
            some_true = False
            for ci_tool in ci_tools:
                used_in_tool = df.query(f"project == '{project}' and ci_tool == '{ci_tool}'")["use_build_tool"].iloc[0]
                all_true = all_true and used_in_tool
                some_true = some_true or used_in_tool

            if len(ci_tools) > 1:
                # TODO a project may use Bazel in GitHub Actions but not in CirCleCI, currentl this is counted as local, we may need to change the strategy
                build_tool_usage.loc[
                    build_tool_usage["CI/CD Services"] == "+".join(ci_tools), "CI" if all_true else "Local"] += 1
            else:
                build_tool_usage.loc[
                    build_tool_usage["CI/CD Services"] == ci_tools[0], "CI" if some_true else "Local"] += 1

        build_tool_usage = build_tool_usage.melt(id_vars="CI/CD Services")
        ax = sns.histplot(data=build_tool_usage, x="CI/CD Services", hue="variable", weights="value", discrete=True,
                          multiple="stack", ax=axs[idx])
        idx += 1
        for c in ax.containers:
            ax.bar_label(c, label_type='center')

        ax.set_title(f"{parent_dir_name} usage in CI/CD services")
        ax.set_xlabel("CI/CD Service")
        ax.set_ylabel("Number of Projects")

    fig.autofmt_xdate()
    plt.savefig("./images/ci_tool_usage")
    plt.show()


def visualize_parallelization_usage(data_dir: str):
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 10), tight_layout=True, sharey=True)

    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    idx = 0
    for parent_dir_name, correspondent_ci_tool in parent_dir_names.items():
        parallelization_usage = {"Parallel": [], "Serial": [],
                                 "CI/CD Services": ["GitHub Actions", "CirCleCI", "Buildkite"]}

        df = pd.read_csv(os.path.join(data_dir, f"{parent_dir_name}-feature_usage.csv")).drop(
            columns=["local_cache", "remote_cache"])
        gha_parallel = df.loc[(df["ci_tool"] == "github_actions") & df["use_parallelization"]]
        gha_serial = df.loc[(df["ci_tool"] == "github_actions") & ~df["use_parallelization"]]
        parallelization_usage["Parallel"].append(len(gha_parallel))
        parallelization_usage["Serial"].append(len(gha_serial))

        circleci_parallel = df.loc[(df["ci_tool"] == "circle_ci") & df["use_parallelization"]]
        circleci_serial = df.loc[(df["ci_tool"] == "circle_ci") & ~df["use_parallelization"]]
        parallelization_usage["Parallel"].append(len(circleci_parallel))
        parallelization_usage["Serial"].append(len(circleci_serial))

        buildkite_parallel = df.loc[(df["ci_tool"] == "buildkite") & df["use_parallelization"]]
        buildkite_serial = df.loc[(df["ci_tool"] == "buildkite") & ~df["use_parallelization"]]
        parallelization_usage["Parallel"].append(len(buildkite_parallel))
        parallelization_usage["Serial"].append(len(buildkite_serial))

        parallelization_usage = pd.DataFrame(parallelization_usage)
        parallelization_usage = parallelization_usage.melt(id_vars="CI/CD Services")
        ax = sns.histplot(data=parallelization_usage, x="CI/CD Services", hue="variable", weights="value",
                          discrete=True, multiple="stack", ax=axs[idx])
        idx += 1

        for c in ax.containers:
            ax.bar_label(c, label_type='center')

        ax.set_title(f"{parent_dir_name} parallelization usage in CI/CD services")
        ax.set_xlabel("CI/CD Service")
        ax.set_ylabel("Number of Projects")

    fig.autofmt_xdate()
    plt.savefig("./images/parallelization_usage")
    plt.show()


def visualize_cache_usage(data_dir: str):
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 10), tight_layout=True, sharey=True)

    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    idx = 0
    for parent_dir_name, correspondent_ci_tool in parent_dir_names.items():
        cache_usage = {"Disk Cache": [], "Remote Cache": [], "No Cache": [],
                       "CI/CD Services": ["GitHub Actions", "CirCleCI", "Buildkite"]}

        df = pd.read_csv(os.path.join(data_dir, f"{parent_dir_name}-feature_usage.csv")).drop(
            columns=["use_parallelization"])
        gha_disk_cache = df.loc[(df["ci_tool"] == "github_actions") & df["local_cache"]]
        gha_remote_cache = df.loc[(df["ci_tool"] == "github_actions") & df["remote_cache"]]
        gha_no_cache = df.loc[(df["ci_tool"] == "github_actions") & ~df["local_cache"] & ~df["remote_cache"]]
        cache_usage["Disk Cache"].append(len(gha_disk_cache))
        cache_usage["Remote Cache"].append(len(gha_remote_cache))
        cache_usage["No Cache"].append(len(gha_no_cache))

        circleci_disk_cache = df.loc[(df["ci_tool"] == "circle_ci") & df["local_cache"]]
        circleci_remote_cache = df.loc[(df["ci_tool"] == "circle_ci") & df["remote_cache"]]
        circleci_no_cache = df.loc[(df["ci_tool"] == "circle_ci") & ~df["local_cache"] & ~df["remote_cache"]]
        cache_usage["Disk Cache"].append(len(circleci_disk_cache))
        cache_usage["Remote Cache"].append(len(circleci_remote_cache))
        cache_usage["No Cache"].append(len(circleci_no_cache))

        buildkite_disk_cache = df.loc[(df["ci_tool"] == "buildkite") & df["local_cache"]]
        buildkite_remote_cache = df.loc[(df["ci_tool"] == "buildkite") & df["remote_cache"]]
        buildkite_no_cache = df.loc[(df["ci_tool"] == "buildkite") & ~df["local_cache"] & ~df["remote_cache"]]
        cache_usage["Disk Cache"].append(len(buildkite_disk_cache))
        cache_usage["Remote Cache"].append(len(buildkite_remote_cache))
        cache_usage["No Cache"].append(len(buildkite_no_cache))

        cache_usage = pd.DataFrame(cache_usage)
        cache_usage = cache_usage.melt(id_vars="CI/CD Services")
        ax = sns.histplot(data=cache_usage, x="CI/CD Services", hue="variable", weights="value", discrete=True,
                          multiple="stack", ax=axs[idx])
        idx += 1

        for c in ax.containers:
            ax.bar_label(c, label_type='center')

        ax.set_title(f"{parent_dir_name} cache usage in CI/CD services")
        ax.set_xlabel("CI/CD Service")
        ax.set_ylabel("Number of Projects")

    fig.autofmt_xdate()
    plt.savefig("./images/cache_usage")
    plt.show()


def subcommand_filter(ci_tool, target_subcommand):
    def filter_func(row):
        if row["ci_tool"] == ci_tool and intersect(row["subcommands"].split(","), target_subcommand):
            return True
        else:
            return False

    return filter_func


def visualize_test_usage(data_dir: str):
    test_goals = ["test", "integration-test", "package", "verify", "install", "deploy"]

    figs, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 10), tight_layout=True, sharey=True)

    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    idx = 0
    for parent_dir_name, correspondent_ci_tool in parent_dir_names.items():
        test_usage = {"Execute Tests": [], "Not Execute Tests": [],
                      "CI/CD Services": ["GitHub Actions", "CirCleCI", "Buildkite"]}

        df = pd.read_csv(os.path.join(data_dir, f"{parent_dir_name}-build_tools.csv"))
        df = df.loc[df["use_build_tool"]]

        gha_test = df.loc[df.apply(subcommand_filter("github_actions", test_goals), axis=1)]
        total_gha_projects = len(df.loc[df["ci_tool"] == "github_actions"]["project"].unique())
        gha_test_projects = len(gha_test["project"].unique())

        test_usage["Execute Tests"].append(gha_test_projects)
        test_usage["Not Execute Tests"].append(total_gha_projects - gha_test_projects)

        circleci_test = df.loc[df.apply(subcommand_filter("circle_ci", test_goals), axis=1)]
        total_circleci_projects = len(df.loc[df["ci_tool"] == "circle_ci"]["project"].unique())
        circleci_test_projects = len(circleci_test["project"].unique())

        test_usage["Execute Tests"].append(circleci_test_projects)
        test_usage["Not Execute Tests"].append(total_circleci_projects - circleci_test_projects)

        buildkite_test = df.loc[df.apply(subcommand_filter("buildkite", test_goals), axis=1)]
        total_buildkite_projects = len(df.loc[df["ci_tool"] == "buildkite"]["project"].unique())
        buildkite_test_projects = len(buildkite_test["project"].unique())

        test_usage["Execute Tests"].append(buildkite_test_projects)
        test_usage["Not Execute Tests"].append(total_buildkite_projects - buildkite_test_projects)

        test_usage = pd.DataFrame(test_usage)
        test_usage = test_usage.melt(id_vars="CI/CD Services")
        ax = sns.histplot(data=test_usage, x="CI/CD Services", hue="variable", weights="value", discrete=True,
                          hue_order=["Not Execute Tests", "Execute Tests"], multiple="stack", ax=axs[idx])
        idx += 1

        for c in ax.containers:
            ax.bar_label(c, label_type='center')

        ax.set_title(f"{parent_dir_name} test usage in CI/CD services")
        ax.set_xlabel("CI/CD Service")
        ax.set_ylabel("Number of Projects")

    figs.autofmt_xdate()
    plt.savefig("./images/test_usage")
    plt.show()


def intersect(lst1, lst2):
    if type(lst1) is not list:
        print(lst1)
        return False
    for x in lst1:
        if x in lst2:
            return True
    return False
