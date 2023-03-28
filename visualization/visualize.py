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
    visualize_ci_tools(data_dir)
    visualize_subcommand_usage(data_dir)
    visualize_parallelization_usage(data_dir)
    visualize_cache_usage(data_dir)
    visualize_build_rule_categories(data_dir)
    visualize_script_usage(data_dir)
    visualize_arg_size(data_dir)


def visualize_ci_tools(data_dir: str):
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 10), sharex=True)
    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    idx = 0
    for parent_dir_name, correspondent_build_tool in parent_dir_names.items():
        df = pd.read_csv(os.path.join(data_dir, f"{parent_dir_name}-build_tools.csv")).drop(
            columns=["subcommands", "skip_tests"]).drop_duplicates()

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
    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    parallelization_usage = {"Parallel": [], "Serial": [], "Dataset": []}
    for parent_dir_name, correspondent_build_tool in parent_dir_names.items():
        df = pd.read_csv(os.path.join(data_dir, f"{parent_dir_name}-feature_usage.csv")).drop(
            columns=["local_cache", "remote_cache", "ci_tool"]).drop_duplicates()

        total_projects = df["project"].nunique()
        parallelized_projects = df.loc[(df["use_parallelization"])]["project"].nunique()
        serial_projects = total_projects - parallelized_projects

        parallelization_usage["Dataset"].append(parent_dir_name)
        parallelization_usage["Parallel"].append(parallelized_projects)
        parallelization_usage["Serial"].append(serial_projects)

    parallelization_usage = pd.DataFrame(parallelization_usage)
    parallelization_usage = parallelization_usage.melt(id_vars="Dataset")
    ax = sns.histplot(data=parallelization_usage, x="Dataset", hue="variable", weights="value",
                      palette="Set2", shrink=.8, multiple="fill")

    for c in ax.containers:
        labels = [f"{p.get_height() * 100:.2f}%" for p in c.patches]
        ax.bar_label(c, labels=labels, label_type='center')

    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_title(f"Parallelization usage of build tools in CI/CD services")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Percentage of Projects")
    sns.move_legend(ax, loc="upper left", title="How the build tool is used", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig("./images/parallelization_usage")
    plt.show()


def visualize_cache_usage(data_dir: str):
    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    cache_usage = {"Remote Cache": [], "Local Cache": [], "No Cache": [], "Dataset": []}
    for parent_dir_name, build_tool in parent_dir_names.items():
        df = pd.read_csv(os.path.join(data_dir, f"{parent_dir_name}-feature_usage.csv")).drop(
            columns=["use_parallelization", "ci_tool"]).drop_duplicates()

        total_projects = df["project"].nunique()
        cache_projects = df.loc[(df["local_cache"]) | (df["remote_cache"])]["project"].nunique()
        remote_cache_projects = df.loc[(df["remote_cache"])]["project"].nunique()
        non_cache_projects = total_projects - cache_projects

        cache_usage["Dataset"].append(parent_dir_name)
        cache_usage["Remote Cache"].append(remote_cache_projects)
        cache_usage["Local Cache"].append(cache_projects - remote_cache_projects)
        cache_usage["No Cache"].append(non_cache_projects)

    cache_usage = pd.DataFrame(cache_usage)
    cache_usage = cache_usage.melt(id_vars="Dataset")
    ax = sns.histplot(data=cache_usage, x="Dataset", hue="variable", weights="value", palette="Set2",
                      shrink=0.8, multiple="fill")

    for c in ax.containers:
        labels = [f"{p.get_height() * 100:.2f}%" for p in c.patches]
        ax.bar_label(c, labels=labels, label_type='center')

    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_title(f"Cache usage of build tools in CI/CD services")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Percentage of Projects")
    sns.move_legend(ax, loc="upper left", title="How the build tool is used", bbox_to_anchor=(1, 1))

    plt.tight_layout()
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
                                                        categories=build_tool_subcommand_names[
                                                                       correspondent_build_tool] + [
                                                                       "Other Subcommands"])

        ax = sns.histplot(data=subcommand_usage, x="Subcommand", weights="Usage Percentage", shrink=.8, ax=axs[idx],
                          color="#66c2a5")

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

        skipTest = False not in project_df["skip_tests"].unique()

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

        if skipTest and "test" in subcommands:
            subcommands.remove("test")

        for subcommand in subcommands:
            if subcommand not in subcommand_usage:
                subcommand_usage[subcommand] = 0

            subcommand_usage[subcommand] += 1


def visualize_build_rule_categories(data_dir: str):
    build_rule_categories = None

    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 10))

    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    for parent_dir_name in parent_dir_names:
        df = pd.read_csv(f"{data_dir}/{parent_dir_name}-build_rules.csv").drop(columns=["name"])
        df["count"] = df.groupby(by=["project", "category"])["category"].transform("count")
        df["dataset"] = parent_dir_name
        df = df.drop_duplicates()

        project_data = pd.read_csv(f"{data_dir}/{parent_dir_name}-projects.csv")
        df = df.merge(project_data, on="project", how="inner")

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

    build_rule_categories["total_count"] = build_rule_categories.apply(
        lambda row: sum(build_rule_categories.loc[build_rule_categories["project"] == row["project"]]["count"]),
        axis=1)

    total_build_rules = build_rule_categories.drop(columns=["category", "count"]).drop_duplicates()
    total_build_rules["total_count_per_source_file"] = total_build_rules["total_count"] / total_build_rules["num_files"]
    total_build_rules["total_count_per_line_of_code"] = total_build_rules["total_count"] / total_build_rules[
        "num_lines"]

    # in total 1127 projects, there is only 8 projects that has more than 6 total_count_per_source_file.
    # So, we removed these outliers.
    total_build_rules = total_build_rules[total_build_rules["total_count_per_source_file"] < 6].reset_index(drop=True)

    ax = sns.violinplot(data=total_build_rules, x="dataset", y="total_count_per_source_file", color="#e78ac3",
                        scale="count", inner="box", ax=axs[0])

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Number of Build Rules Per Source File")

    ax.set_title("The Number of Build Rules Per Source File in Projects")

    # plot percentage of projects that use custom build rules
    total_bazel_projects = build_rule_categories.loc[build_rule_categories["dataset"] == "bazel-projects"][
        "project"].nunique()
    custom_rule_bazel_projects = build_rule_categories.loc[(build_rule_categories["dataset"] == "bazel-projects") & (
            build_rule_categories["category"] == "custom")]["project"].nunique()

    total_maven_large_projects = build_rule_categories.loc[build_rule_categories["dataset"] == "maven-large-projects"][
        "project"].nunique()
    custom_rule_maven_large_projects = \
        build_rule_categories.loc[(build_rule_categories["dataset"] == "maven-large-projects") & (
                build_rule_categories["category"] == "custom")]["project"].nunique()

    total_maven_small_projects = build_rule_categories.loc[build_rule_categories["dataset"] == "maven-small-projects"][
        "project"].nunique()
    custom_rule_maven_small_projects = \
        build_rule_categories.loc[(build_rule_categories["dataset"] == "maven-small-projects") & (
                build_rule_categories["category"] == "custom")]["project"].nunique()

    custom_rule_percentages = pd.DataFrame(
        {"dataset": ["bazel-projects", "maven-large-projects", "maven-small-projects"], "percentage": [
            custom_rule_bazel_projects / total_bazel_projects,
            custom_rule_maven_large_projects / total_maven_large_projects,
            custom_rule_maven_small_projects / total_maven_small_projects
        ]})

    ax = sns.histplot(data=custom_rule_percentages, x="dataset", weights="percentage", color="#e78ac3", ax=axs[1])
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Percentage of Projects that Use Custom Build Rules")
    ax.set_title("Percentage of Projects that Use Custom Build Rules")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))

    for c in ax.containers:
        labels = [f"{p.get_height() * 100:.2f}%" for p in c.patches]

        ax.bar_label(c, labels=labels, label_type="center")

    # plot the distribution of percentage of build rule categories in projects
    build_category_percentages = build_rule_categories[["project", "dataset"]].copy().drop_duplicates().reset_index(
        drop=True)

    custom_rule_percentages = build_category_percentages.copy()
    custom_rule_percentages["category"] = "custom"
    native_rule_percentages = build_category_percentages.copy()
    native_rule_percentages["category"] = "native"
    external_rule_percentages = build_category_percentages.copy()
    external_rule_percentages["category"] = "external"

    build_category_percentages = pd.concat(
        [custom_rule_percentages, native_rule_percentages, external_rule_percentages],
        axis=0, ignore_index=True).reset_index(drop=True)
    build_category_percentages["percentage"] = build_category_percentages.apply(
        lambda row: calculate_build_rule_percentage_for_row(row, build_rule_categories), axis=1)

    build_category_percentages = build_category_percentages[build_category_percentages["percentage"] != 0]
    ax = sns.boxplot(data=build_category_percentages, x="dataset", y="percentage", hue="category", palette="Set2",
                     ax=axs[2], dodge=True)

    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Percentage of Build Rules")
    ax.set_title("Percentage of Build Rules in Projects")

    sns.move_legend(ax, loc="upper left", title="Category", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    fig.autofmt_xdate()

    plt.savefig("./images/build_rules")
    plt.show()


def calculate_build_rule_percentage_for_row(row, build_rule_categories):
    project = row["project"]
    dataset = row["dataset"]
    category = row["category"]

    total_count = build_rule_categories[(build_rule_categories["project"] == project) & (
            build_rule_categories["dataset"] == dataset)]["count"].sum()

    category_rows = build_rule_categories[(build_rule_categories["project"] == project) & (
            build_rule_categories["dataset"] == dataset) & (build_rule_categories["category"] == category)]
    if len(category_rows) == 0:
        category_count = 0
    else:
        category_count = category_rows["count"].sum()

    return category_count / total_count


def visualize_script_usage(data_dir: str):
    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    script_usage = {"Used": [], "Not Used": [], "Dataset": []}

    for parent_dir_name in parent_dir_names:
        df = pd.read_csv(f"{data_dir}/{parent_dir_name}-script_usage.csv")
        df["script"] = df.apply(lambda row: sum(
            df.loc[(df["project"] == row["project"]) & (df["invoked_by_script"])]["invoked_by_script"]) != 0,
                                axis=1)
        df = df.drop(columns=["invoked_by_script"])
        df = df.drop_duplicates()

        total_projects = df["project"].unique().size
        used_script_projects = df.loc[(df["script"])]["project"].nunique()
        not_used_script_projects = total_projects - used_script_projects

        script_usage["Dataset"].append(parent_dir_name)
        script_usage["Used"].append(used_script_projects)
        script_usage["Not Used"].append(not_used_script_projects)

    script_usage = pd.DataFrame(script_usage)
    script_usage = script_usage.melt(id_vars="Dataset")
    ax = sns.histplot(data=script_usage, x="Dataset", hue="variable", weights="value", multiple="fill",
                      palette="Set2", shrink=0.8)

    for c in ax.containers:
        labels = [f"{p.get_height() * 100:.2f}%" for p in c.patches]
        ax.bar_label(c, labels=labels, label_type='center')

    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Percentage of Projects")
    ax.set_title("Percentage of Projects that Use Shell Scripts to Run Build Systems")
    ax.set_ylim(0, 1)
    sns.move_legend(ax, loc="upper left", title="", bbox_to_anchor=(1, 1))

    plt.tight_layout()
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
