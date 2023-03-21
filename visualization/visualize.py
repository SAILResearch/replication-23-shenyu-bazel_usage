import copy

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, ticker
import matplotlib.ticker as mticker

from visualization.preprocess import *


def visualize_data(data_dir: str):
    sns.set_style("whitegrid")

    data_dir = os.path.join(data_dir, "processed")
    # visualize_ci_tools(data_dir)
    visualize_subcommand_usage(data_dir)
    # visualize_parallelization_usage(data_dir)
    # visualize_cache_usage(data_dir)
    # visualize_build_rule_categories(data_dir)
    # visualize_script_usage(data_dir)
    # visualize_arg_size(data_dir)


def visualize_ci_tools(data_dir: str):
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 10), sharex=True)
    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    idx = 0
    for parent_dir_name, correspondent_build_tool in parent_dir_names.items():
        df = pd.read_csv(os.path.join(data_dir, f"{parent_dir_name}-build_tools.csv")).drop(
            columns=["subcommands"]).drop_duplicates()

        build_tool_usage = pd.DataFrame({"Local": [0, 0, 0, 0], "CI": [0, 0, 0, 0],
                                         "CI/CD Services": ["github_actions", "circle_ci", "buildkite", "travis_ci"]})

        for project in df["project"].unique():
            ci_tools = sorted(df[df["project"] == project]["ci_tool"].unique())
            for ci_tool in ci_tools:
                used_in_tool = df.query(f"project == '{project}' and ci_tool == '{ci_tool}'")["use_build_tool"].iloc[0]
                build_tool_usage.loc[
                    build_tool_usage["CI/CD Services"] == ci_tool, "CI" if used_in_tool else "Local"] += 1

        build_tool_usage["total"] = build_tool_usage["Local"] + build_tool_usage["CI"]
        build_tool_usage["CI_percentage"] = build_tool_usage["CI"] / build_tool_usage["total"]
        build_tool_usage["Local_percentage"] = build_tool_usage["Local"] / build_tool_usage["total"]

        ax = sns.histplot(data=build_tool_usage, weights="CI_percentage", x="CI/CD Services",
                          shrink=.8, ax=axs[0][idx], color="#66c2a5")

        ax.set_title(f"{correspondent_build_tool} ({parent_dir_name})")

        ax.set(ylim=(0, 1))
        if idx == 0:
            ax.set_ylabel("Percentage of projects using the build tool in CI")
        else:
            ax.set_ylabel("")

        for c in ax.containers:
            labels = []
            for p, bar_idx in zip(c.patches, range(len(c.patches))):
                if p.get_height() == 0:
                    labels.append("")
                else:
                    labels.append(f"{int(p.get_height() * 10000) / 100}% ({build_tool_usage['CI'][bar_idx]})")

            ax.bar_label(c, labels=labels, padding=1)

        ax = sns.histplot(data=build_tool_usage, weights="Local_percentage", x="CI/CD Services", shrink=.8,
                          ax=axs[1][idx], color="#fc8d62")
        ax.set(ylim=(0, 1))
        if idx == 0:
            ax.set_ylabel("Percentage of projects only using the build tool locally")
            yticks = ax.get_yticklabels()
            yticks[-1].set_visible(False)
        else:
            ax.set_ylabel(None)

        ax.invert_yaxis()

        for c in ax.containers:
            labels = []
            for p, bar_idx in zip(c.patches, range(len(c.patches))):
                if p.get_height() == 0:
                    labels.append("0%")
                else:
                    labels.append(f"{int(p.get_height() * 10000) / 100}% ({build_tool_usage['Local'][bar_idx]})")

            ax.bar_label(c, labels=labels, padding=1)
        idx += 1

    for ax in axs.flat:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))

    plt.suptitle("Build tool usage in CI/CD services")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig("./images/ci_tool_usage")
    plt.show()


def visualize_parallelization_usage(data_dir: str):
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 10), tight_layout=True, sharey=True)

    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    idx = 0
    for parent_dir_name, correspondent_build_tool in parent_dir_names.items():
        parallelization_usage = {"Parallel": [], "Serial": [],
                                 "CI/CD Services": ["GitHub Actions", "CirCleCI", "Buildkite", "TravisCI"]}

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

        travisci_parallel = df.loc[(df["ci_tool"] == "travis_ci") & df["use_parallelization"]]
        travisci_serial = df.loc[(df["ci_tool"] == "travis_ci") & ~df["use_parallelization"]]
        parallelization_usage["Parallel"].append(len(travisci_parallel))
        parallelization_usage["Serial"].append(len(travisci_serial))

        parallelization_usage = pd.DataFrame(parallelization_usage)
        parallelization_usage = parallelization_usage.melt(id_vars="CI/CD Services")
        ax = sns.histplot(data=parallelization_usage, x="CI/CD Services", hue="variable", weights="value",
                          palette="Set2",
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
    for parent_dir_name, correspondent_build_tool in parent_dir_names.items():
        cache_usage = {"Disk Cache": [], "Remote Cache": [], "No Cache": [],
                       "CI/CD Services": ["GitHub Actions", "CirCleCI", "Buildkite", "TravisCI"]}

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

        travisci_disk_cache = df.loc[(df["ci_tool"] == "travis_ci") & df["local_cache"]]
        travisci_remote_cache = df.loc[(df["ci_tool"] == "travis_ci") & df["remote_cache"]]
        travisci_no_cache = df.loc[(df["ci_tool"] == "travis_ci") & ~df["local_cache"] & ~df["remote_cache"]]
        cache_usage["Disk Cache"].append(len(travisci_disk_cache))
        cache_usage["Remote Cache"].append(len(travisci_remote_cache))
        cache_usage["No Cache"].append(len(travisci_no_cache))

        cache_usage = pd.DataFrame(cache_usage)
        cache_usage = cache_usage.melt(id_vars="CI/CD Services")
        ax = sns.histplot(data=cache_usage, x="CI/CD Services", hue="variable", weights="value", discrete=True,
                          palette="Set2",
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


def visualize_subcommand_usage(data_dir: str):
    figs, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 10), tight_layout=True, sharey=True)

    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    build_tool_subcommand_names = {"bazel": ["build", "test"],
                                   "maven": ["compile", "test", "package", "verify", "install", "deploy"]}

    idx = 0
    for parent_dir_name, correspondent_build_tool in parent_dir_names.items():
        subcommand_usage = {}

        df = pd.read_csv(os.path.join(data_dir, f"{parent_dir_name}-build_tools.csv"))
        df = df.loc[df["use_build_tool"]]
        count_unique_subcommand_usage(correspondent_build_tool, build_tool_subcommand_names, df, subcommand_usage)

        subcommand_usage = pd.DataFrame(subcommand_usage.items(), columns=["Subcommand", "Count"])
        total_projects = df["project"].unique().size
        subcommand_usage["Usage Percentage"] = subcommand_usage["Count"] / total_projects
        subcommand_usage["Subcommand"] = pd.Categorical(subcommand_usage["Subcommand"],
                                                        categories=build_tool_subcommand_names[correspondent_build_tool] + ["Other Subcommands"])

        ax = sns.histplot(data=subcommand_usage, x="Subcommand", weights="Usage Percentage", shrink=.8, ax=axs[idx], color="#66c2a5")

        idx += 1

        for c in ax.containers:
            ax.bar_label(c, labels=[f"{round(p.get_height() * 100, 2)}%" for p in c.patches], label_type='edge')

        ax.set_title(f"{correspondent_build_tool} ({parent_dir_name})")
        ax.set_xlabel("CI/CD Service")
        ax.set_ylabel("Percentage of Projects Using Subcommand")

    for ax in axs:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))

    plt.suptitle("Subcommands usage in CI/CD services")
    plt.tight_layout()
    figs.autofmt_xdate()
    plt.savefig("./images/command_usage")
    plt.show()


# TODO we need to check the -Dmaven.test.skip=true flag or -DskipTests flag in the build commands!
def count_unique_subcommand_usage(correspondent_build_tool, build_tool_subcommand_names, gha_projects,
                                  subcommand_usage):
    for project in gha_projects["project"].unique():
        project_df = gha_projects.loc[gha_projects["project"] == project]
        subcommands = set()
        project_df.apply(lambda row: subcommands.update(row["subcommands"].split(",")), axis=1)

        if correspondent_build_tool == "bazel":
            subcommands = set(
                [subcommand if subcommand in build_tool_subcommand_names["bazel"] else "Other Subcommands" for
                 subcommand in
                 subcommands])
        elif correspondent_build_tool == "maven":
            new_subcommands = set()
            for used_subcommand in subcommands:
                if used_subcommand not in build_tool_subcommand_names["maven"]:
                    new_subcommands.add("Other Subcommands")
                    continue

                for subcommand in build_tool_subcommand_names["maven"]:
                    new_subcommands.add(subcommand)

                    if subcommand == used_subcommand:
                        break

            subcommands = new_subcommands

        for subcommand in subcommands:
            if subcommand not in subcommand_usage:
                subcommand_usage[subcommand] = 0

            subcommand_usage[subcommand] += 1


def visualize_build_rule_categories(data_dir: str):
    build_rule_categories = None

    fig = plt.figure(figsize=(20, 10))

    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    for parent_dir_name in parent_dir_names:
        df = pd.read_csv(f"{data_dir}/{parent_dir_name}-build_rules.csv").drop(columns=["name"])
        df["count"] = df.groupby(by=["project", "category"])["category"].transform("count")
        df["dataset"] = parent_dir_name
        df = df.drop_duplicates()

        if build_rule_categories is None:
            build_rule_categories = df
        else:
            build_rule_categories = pd.concat([build_rule_categories, df])

        print(f"mean of {parent_dir_name} is {df['count'].mean()}")
        print(f"median of {parent_dir_name} is {df['count'].median()}")
        print("------------------")

        print(f"total number of projects in {parent_dir_name} is {df['project'].unique().size}")
        print(
            f"number of projects that use custom rule in {parent_dir_name} is {df.loc[df['category'] == 'custom']['project'].unique().size}")
        print("------------------")

        print(f"mean of custom rules in {parent_dir_name} is {df.loc[df['category'] == 'custom']['count'].mean()}")
        print(f"median of custom rules in {parent_dir_name} is {df.loc[df['category'] == 'custom']['count'].median()}")
        print("------------------")

    # g = sns.catplot(data=build_rule_categories, x="dataset", y="count", hue="category", kind="boxen", palette="Set2", scale="exponential",
    #                 showfliers=False, k_depth="trustworthy", legend=False, dodge=True)

    build_rule_categories["count_log"] = np.log10(build_rule_categories["count"])
    g = sns.catplot(data=build_rule_categories, x="dataset", y="count_log", hue="category", palette="Set2",
                    scale="count",
                    kind="violin", inner="box", scale_hue=True, legend=False)

    g.set_xlabels("Dataset")
    g.set_ylabels("Number of Build Rules (log10)")
    g.despine(left=True)
    plt.legend(loc='upper right', title="Category")

    ax = g.ax
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    ymin, ymax = ax.get_ylim()
    tick_range = np.arange(0, ymax)
    ax.yaxis.set_ticks(tick_range)
    ax.yaxis.set_ticks([np.log10(x) for p in tick_range for x in np.linspace(10 ** p, 10 ** (p + 1), 10)],
                       minor=True)

    g.ax.set_title("The Number of Build Rules per project by category")

    plt.tight_layout()
    fig.autofmt_xdate()

    plt.savefig("./images/build_rule_categories")
    plt.show()


def visualize_script_usage(data_dir: str):
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 10), tight_layout=True, sharey=True)

    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    idx = 0
    for parent_dir_name in parent_dir_names:
        script_usage = {"Used": [], "Not Used": [],
                        "CI/CD Services": ["GitHub Actions", "CirCleCI", "Buildkite", "TravisCI"]}

        df = pd.read_csv(f"{data_dir}/{parent_dir_name}-script_usage.csv")
        df["script"] = df.apply(lambda row: sum(
            df.loc[(df["project"] == row["project"]) & (df["invoked_by_script"])]["invoked_by_script"]) != 0,
                                axis=1)

        df = df.drop(columns=["invoked_by_script"])
        df = df.drop_duplicates()

        total_gha_projects = df.loc[df["ci_tool"] == "github_actions"]["project"].unique().size
        used_script_gha_projects = df.loc[(df["ci_tool"] == "github_actions") & df["script"]]["project"].unique().size
        script_usage["Used"].append(used_script_gha_projects)
        script_usage["Not Used"].append(total_gha_projects - used_script_gha_projects)

        total_circleci_projects = df.loc[df["ci_tool"] == "circle_ci"]["project"].unique().size
        used_script_circleci_projects = df.loc[(df["ci_tool"] == "circle_ci") & df["script"]]["project"].unique().size
        script_usage["Used"].append(used_script_circleci_projects)
        script_usage["Not Used"].append(total_circleci_projects - used_script_circleci_projects)

        total_buildkite_projects = df.loc[df["ci_tool"] == "buildkite"]["project"].unique().size
        used_script_buildkite_projects = df.loc[(df["ci_tool"] == "buildkite") & df["script"]]["project"].unique().size
        script_usage["Used"].append(used_script_buildkite_projects)
        script_usage["Not Used"].append(total_buildkite_projects - used_script_buildkite_projects)

        total_travisci_projects = df.loc[df["ci_tool"] == "travis_ci"]["project"].unique().size
        used_script_travisci_projects = df.loc[(df["ci_tool"] == "travis_ci") & df["script"]]["project"].unique().size
        script_usage["Used"].append(used_script_travisci_projects)
        script_usage["Not Used"].append(total_travisci_projects - used_script_travisci_projects)

        script_usage = pd.DataFrame(script_usage)
        script_usage = script_usage.melt(id_vars="CI/CD Services")

        ax = sns.histplot(data=script_usage, x="CI/CD Services", hue="variable", weights="value", discrete=True,
                          palette="Set2", multiple="stack", ax=axs[idx], legend=0)
        idx += 1
        for c in ax.containers:
            ax.bar_label(c, label_type='center')

        ax.set_title(f"{parent_dir_name} Script Usage in CI/CD services")
        ax.set_xlabel("CI/CD Service")
        ax.set_ylabel("Number of Projects")

    fig.legend(title="Whether use shell script to run build systems", labels=["Yes", "No"], loc="center right",
               bbox_to_anchor=(1, 0.9))

    fig.autofmt_xdate()
    plt.savefig("./images/script_usage")
    plt.show()


def visualize_arg_size(data_dir: str):
    build_arg_size = None

    fig = plt.figure(figsize=(20, 10))

    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    for parent_dir_name in parent_dir_names:
        df = pd.read_csv(f"{data_dir}/{parent_dir_name}-arg_size.csv").drop(columns=["expanded_command_size"])
        df = df.drop_duplicates()
        df["dataset"] = parent_dir_name

        if build_arg_size is None:
            build_arg_size = df
        else:
            build_arg_size = build_arg_size.append(df)

    g = sns.catplot(data=build_arg_size, x="dataset", y="non_expanded_command_size", hue="ci_tool", palette="Set2",
                    kind="violin", inner="box", scale_hue=True, legend=False)

    g.set_xlabels("Dataset")
    g.set_ylabels("Count of Arguments in Build Commands")
    g.despine(left=True)
    plt.legend(loc='upper right', title="CI Tool")

    g.ax.set_title("Count of Arguments in Build Commands by CI Tool")

    plt.tight_layout()
    fig.autofmt_xdate()
    plt.savefig("./images/build_arg_size")
    plt.show()
