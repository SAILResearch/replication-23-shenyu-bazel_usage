import copy
import itertools

import math
from textwrap import wrap

import numpy as np
from cliffs_delta import cliffs_delta
import pandas as pd
import scipy as scipy
import scikit_posthocs as sp
import seaborn as sns
from matplotlib import pyplot as plt, ticker
from matplotlib.gridspec import GridSpec
from matplotlib_venn import venn2, venn2_circles

from visualization.preprocess import *


def visualize_data(data_dir: str):
    sns.set_style("whitegrid")

    data_dir = os.path.join(data_dir, "processed")
    visualize_ci_tools(data_dir)
    visualize_subcommand_usage(data_dir)
    visualize_subcommand_intersection(data_dir)
    visualize_parallelization_usage(data_dir)
    visualize_cache_usage(data_dir)
    visualize_build_rule_categories(data_dir)
    visualize_build_system_invoker(data_dir)
    # visualize_arg_size(data_dir)
    visualize_parallelization_experiments_by_commits(data_dir)
    visualize_parallelization_experiments_by_build_durations(data_dir)
    parallelism_confidence_levels(data_dir)
    visualize_parallelism_utilization()
    visualize_cache_experiments_change_size(data_dir)
    visualize_cache_speed_up(data_dir)
    cache_speedup_confidence_levels(data_dir)


def visualize_ci_tools(data_dir: str):
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 10), sharex=True)
    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    titles = ["Bazel Projects", "Large Maven Projects", "Small Maven Projects"]
    idx = 0
    for parent_dir_name, correspondent_build_tool in parent_dir_names.items():
        df = pd.read_csv(os.path.join(data_dir, f"{parent_dir_name}-build_tools.csv")).drop(
            columns=["subcommands", "skip_tests", "build_tests_only"]).drop_duplicates()

        build_tool_usage = pd.DataFrame({"Local": [0, 0, 0, 0], "CI": [0, 0, 0, 0],
                                         "CI/CD Services": ["github_actions", "circle_ci", "buildkite", "travis_ci"]})

        df["use_build_tool"] = df.apply(
            lambda row: df.loc[
                (df["project"] == row["project"]) & (df["ci_tool"] == row["ci_tool"]) & (
                    df["use_build_tool"])].any().all(),
            axis=1)
        df = df.drop_duplicates()

        for project in df["project"].unique():
            ci_tools = sorted(df[df["project"] == project]["ci_tool"].unique())
            for ci_tool in ci_tools:
                used_in_tool = df.query(f"project == '{project}' and ci_tool == '{ci_tool}'")["use_build_tool"].iloc[0]
                build_tool_usage.loc[
                    build_tool_usage["CI/CD Services"] == ci_tool, "CI" if used_in_tool else "Local"] += 1

        build_tool_usage["total"] = build_tool_usage["Local"] + build_tool_usage["CI"]
        build_tool_usage["CI_percentage"] = build_tool_usage["CI"] / build_tool_usage["total"]
        build_tool_usage["Local_percentage"] = build_tool_usage["Local"] / build_tool_usage["total"]

        overall_ci_tool_usage = build_tool_usage["CI"].sum() / build_tool_usage["total"].sum()
        print(
            f"{correspondent_build_tool} overall CI tool usage: {build_tool_usage['CI'].sum()}/{build_tool_usage['total'].sum()}={overall_ci_tool_usage}")

        ax = sns.histplot(data=build_tool_usage, weights="CI_percentage", x="CI/CD Services",
                          shrink=.8, ax=axs[0][idx], color="#66c2a5")

        ax.set_title(f"{titles[idx]}", fontsize=20, pad=20)
        ax.tick_params(labelsize=15)
        ax.set(ylim=(0, 1))
        if idx == 0:
            ax.set_ylabel("Using build systems in CI services", fontsize=15)
        else:
            ax.set_ylabel("")

        for c in ax.containers:
            labels = []
            for p, bar_idx in zip(c.patches, range(len(c.patches))):
                if p.get_height() == 0:
                    labels.append("")
                else:
                    labels.append(f"{int(p.get_height() * 10000) / 100}% ({build_tool_usage['CI'][bar_idx]})")

            ax.bar_label(c, labels=labels, fontsize=12, padding=1)

        ax = sns.histplot(data=build_tool_usage, weights="Local_percentage", x="CI/CD Services", shrink=.8,
                          ax=axs[1][idx], color="#fc8d62")
        ax.tick_params(labelsize=15)
        ax.set(ylim=(0, 1))
        if idx == 0:
            ax.set_ylabel("Not using build systems in CI services", fontsize=15)
        else:
            ax.set_ylabel(None)
        ax.set_xlabel("")
        ax.set_xticklabels(["Github Actions", "Circle CI", "Buildkite", "Travis CI"])

        ax.invert_yaxis()

        for c in ax.containers:
            labels = []
            for p, bar_idx in zip(c.patches, range(len(c.patches))):
                if p.get_height() == 0:
                    labels.append("0%")
                else:
                    labels.append(f"{int(p.get_height() * 10000) / 100}% ({build_tool_usage['Local'][bar_idx]})")

            ax.bar_label(c, labels=labels, fontsize=12, padding=1)
        idx += 1

    for ax in axs.flat:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))

    fig.autofmt_xdate()
    fig.supxlabel("CI/CD Services", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    savefig("./images/ci_tool_usage")
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
        ax.bar_label(c, labels=labels, label_type='center', fontsize=12)

    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_xlabel("")
    ax.set_xticklabels(["Bazel Projects", "Large Maven Projects", "Small Maven Projects"], fontsize=12)
    ax.set_ylabel("Percentage of Projects", fontsize=12)
    ax.tick_params(labelsize=12)
    sns.move_legend(ax, loc="upper left", title="", fontsize=12, bbox_to_anchor=(1, 1))

    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    savefig("./images/parallelization_usage")
    plt.show()


def visualize_cache_usage(data_dir: str):
    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    cache_usage = {"Build System-enabled Cache": [], "CI-enabled Cache": [], "No Cache": [], "Dataset": []}
    for parent_dir_name, build_tool in parent_dir_names.items():
        df = pd.read_csv(os.path.join(data_dir, f"{parent_dir_name}-feature_usage.csv")).drop(
            columns=["use_parallelization", "ci_tool"]).drop_duplicates()

        total_projects = df["project"].nunique()
        cache_projects = df.loc[(df["local_cache"]) | (df["remote_cache"])]["project"].nunique()
        remote_cache_projects = df.loc[(df["remote_cache"])]["project"].nunique()
        non_cache_projects = total_projects - cache_projects

        cache_usage["Dataset"].append(parent_dir_name)
        cache_usage["Build System-enabled Cache"].append(remote_cache_projects)
        cache_usage["CI-enabled Cache"].append(cache_projects - remote_cache_projects)
        cache_usage["No Cache"].append(non_cache_projects)

    cache_usage = pd.DataFrame(cache_usage)
    cache_usage = cache_usage.melt(id_vars="Dataset")
    ax = sns.histplot(data=cache_usage, x="Dataset", hue="variable", weights="value", palette="Set2",
                      shrink=0.8, multiple="fill")

    for c in ax.containers:
        labels = [f"{p.get_height() * 100:.2f}%" for p in c.patches]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=12)

    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_xlabel("")
    ax.set_xticklabels(["Bazel Projects", "Large Maven Projects", "Small Maven Projects"], fontsize=12)
    ax.set_ylabel("Percentage of Projects", fontsize=12)
    ax.tick_params(labelsize=12)
    labels = ["No Cache", "CI-Local Cache", "Remote Cache"]
    labels = ["\n".join(wrap(l, 15)) for l in labels]
    ax.legend(labels=labels)

    sns.move_legend(ax, loc="upper left", title="", bbox_to_anchor=(1, 1), fontsize=12)
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    savefig("./images/cache_usage")
    plt.show()


def visualize_subcommand_intersection(data_dir: str):
    figs, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 6), tight_layout=True)
    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    build_tool_subcommand_names = {"bazel": ["build", "test"],
                                   "maven": ["compile", "test", "package", "verify", "install", "deploy"]}

    idx = 0
    titles = ["Bazel Projects", "Large Maven Projects", "Small Maven Projects"]
    for parent_dir_name, correspondent_build_tool in parent_dir_names.items():
        # Build only, Test Only, Build and Test
        build_test_usage = [0, 0, 0]
        df = pd.read_csv(os.path.join(data_dir, f"{parent_dir_name}-build_tools.csv"))
        df = df.loc[df["use_build_tool"]]

        print(f"Total projects in {parent_dir_name}: {df['project'].nunique()}")

        for project in df["project"].unique():
            project_df = df.loc[df["project"] == project]

            unparsed_subcommands = project_df["subcommands"].tolist()
            subcommands = []
            for subcommand in unparsed_subcommands:
                subcommands.extend(subcommand.split(","))

            use_build, use_test = False, False
            for subcommand in subcommands:
                if correspondent_build_tool == "bazel":
                    if "build" == subcommand:
                        use_build = True
                    if "test" == subcommand:
                        use_test = True
                elif correspondent_build_tool == "maven":
                    for phase in build_tool_subcommand_names["maven"]:
                        if phase == "compile":
                            use_build = True
                        if phase == "test":
                            use_test = True

                        if phase == subcommand:
                            break
                    if False not in project_df["skip_tests"].unique():
                        use_test = False

            if use_build and use_test:
                build_test_usage[2] += 1
            elif use_build:
                build_test_usage[0] += 1
            elif use_test:
                build_test_usage[1] += 1

        set_labels = ["Build", "Test"] if idx == 0 else ["Compile", "Test"]
        venn2(subsets=build_test_usage, set_labels=set_labels, set_colors=["#66c2a5", "#fc8d62"], alpha=1,
              ax=axs[idx])
        axs[idx].set_title(titles[idx], fontsize=12)
        idx += 1
    plt.show()


def visualize_subcommand_usage(data_dir: str):
    figs, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 6), tight_layout=True, sharey=True)

    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    build_tool_subcommand_names = {"bazel": ["build", "test"],
                                   "maven": ["compile", "test", "package", "verify", "install", "deploy"]}
    titles = ["Bazel Projects", "Large Maven Projects", "Small Maven Projects"]

    idx = 0
    for parent_dir_name, correspondent_build_tool in parent_dir_names.items():
        subcommand_usage = {}

        df = pd.read_csv(os.path.join(data_dir, f"{parent_dir_name}-build_tools.csv"))
        df = df.loc[df["use_build_tool"]]
        total_projects = df["project"].unique().size

        count_unique_subcommand_usage(correspondent_build_tool, build_tool_subcommand_names, df, subcommand_usage,
                                      False)
        for subcommand, count in sorted(subcommand_usage.items(), key=lambda x: x[1]):
            print(
                f"{parent_dir_name} - Subcommand: {subcommand}, Count: {count}, Percentage: {count / total_projects:.4f}")

        subcommand_usage = {}
        count_unique_subcommand_usage(correspondent_build_tool, build_tool_subcommand_names, df, subcommand_usage)

        subcommand_usage = pd.DataFrame(subcommand_usage.items(), columns=["Subcommand", "Count"])
        subcommand_usage["Usage Percentage"] = subcommand_usage["Count"] / total_projects
        subcommand_usage["Subcommand"] = pd.Categorical(subcommand_usage["Subcommand"],
                                                        categories=build_tool_subcommand_names[
                                                                       correspondent_build_tool] + [
                                                                       "Other Subcommands"])

        ax = sns.histplot(data=subcommand_usage, x="Subcommand", weights="Usage Percentage", shrink=.8, ax=axs[idx],
                          color="#66c2a5")
        ax.set_title(f"{titles[idx]}", fontsize=20, pad=20)
        idx += 1

        for c in ax.containers:
            ax.bar_label(c, labels=[f"{round(p.get_height() * 100, 2)}%" for p in c.patches], label_type='edge',
                         fontsize=12)

        ax.set_xlabel("")
        ax.tick_params(labelsize=15)
        ax.set_ylabel("Percentage of Projects Using Subcommand", fontsize=15)

    for ax in axs:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))

    figs.supxlabel("Subcommands", fontsize=20)
    figs.autofmt_xdate()
    plt.tight_layout()
    savefig("./images/command_usage")
    plt.show()


def count_unique_subcommand_usage(correspondent_build_tool, build_tool_subcommand_names, gha_projects,
                                  subcommand_usage, other_category=True):
    for project in gha_projects["project"].unique():
        project_df = gha_projects.loc[gha_projects["project"] == project]
        subcommands = set()
        project_df.apply(lambda row: subcommands.update(row["subcommands"].split(",")), axis=1)

        skipTest = False not in project_df["skip_tests"].unique()

        if correspondent_build_tool == "bazel":
            if other_category:
                subcommands = set(
                    [subcommand if subcommand in build_tool_subcommand_names["bazel"] else "Other Subcommands" for
                     subcommand in
                     subcommands])
        elif correspondent_build_tool == "maven":
            new_subcommands = set()
            for used_subcommand in subcommands:
                if used_subcommand not in build_tool_subcommand_names["maven"]:
                    if other_category:
                        new_subcommands.add("Other Subcommands")
                    else:
                        new_subcommands.add(used_subcommand)
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

    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(20, 8))

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

    print(
        f"mean of total_count_per_source_file for bazel is {total_build_rules.loc[total_build_rules['dataset'] == 'bazel-projects']['total_count_per_source_file'].mean()}")
    print(
        f"median of total_count_per_source_file for bazel is {total_build_rules.loc[total_build_rules['dataset'] == 'bazel-projects']['total_count_per_source_file'].median()}")

    print(
        f"mean of total_count_per_source_file for maven is {total_build_rules.loc[total_build_rules['dataset'] == 'maven-large-projects']['total_count_per_source_file'].mean()}")
    print(
        f"median of total_count_per_source_file for maven is {total_build_rules.loc[total_build_rules['dataset'] == 'maven-large-projects']['total_count_per_source_file'].median()}")

    print(
        f"mean of total_count_per_source_file for maven is {total_build_rules.loc[total_build_rules['dataset'] == 'maven-small-projects']['total_count_per_source_file'].mean()}")
    print(
        f"median of total_count_per_source_file for maven is {total_build_rules.loc[total_build_rules['dataset'] == 'maven-small-projects']['total_count_per_source_file'].median()}")

    # in total 1127 projects, there is only 8 projects that has more than 6 total_count_per_source_file.
    # So, we removed these outliers.
    total_build_rules = total_build_rules[total_build_rules["total_count_per_source_file"] < 6].reset_index(drop=True)

    ax = sns.violinplot(data=total_build_rules, x="dataset", y="total_count_per_source_file", color="#66c2a5",
                        scale="count", inner="box", ax=axs[0])

    ax.set_xlabel("")
    ax.set_ylabel("Number of Build Rules Per Source File in Projects", fontsize=20)
    ax.set_title("The Number of Build Rules", fontsize=20)
    ax.tick_params(labelsize=15)

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

    ax = sns.histplot(data=custom_rule_percentages, x="dataset", weights="percentage", color="#66c2a5", ax=axs[1])
    ax.set_xlabel("")
    ax.set_ylabel("Percentage of Projects that Use Custom Build Rules", fontsize=20)
    ax.set_title("The Usage of Custom Build Rules", fontsize=20)
    ax.tick_params(labelsize=15)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))

    for c in ax.containers:
        labels = [f"{p.get_height() * 100:.2f}%" for p in c.patches]

        ax.bar_label(c, labels=labels, label_type="center", fontsize=15)

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
    print("------------------")
    print(
        f"the median of percentage of external build rules in bazel projects is {build_category_percentages.loc[(build_category_percentages['dataset'] == 'bazel-projects') & (build_category_percentages['category'] == 'external')]['percentage'].median()}")
    print(
        f"the median of percentage of external build rules in maven large projects is {build_category_percentages.loc[(build_category_percentages['dataset'] == 'maven-large-projects') & (build_category_percentages['category'] == 'external')]['percentage'].median()}")
    print(
        f"the median of percentage of external build rules in maven small projects is {build_category_percentages.loc[(build_category_percentages['dataset'] == 'maven-small-projects') & (build_category_percentages['category'] == 'external')]['percentage'].median()}")
    print("------------------")

    print(
        f"the median of percentage of native build rules in bazel projects is {build_category_percentages.loc[(build_category_percentages['dataset'] == 'bazel-projects') & (build_category_percentages['category'] == 'native')]['percentage'].median()}")
    print(
        f"the median of percentage of native build rules in maven large projects is {build_category_percentages.loc[(build_category_percentages['dataset'] == 'maven-large-projects') & (build_category_percentages['category'] == 'native')]['percentage'].median()}")
    print(
        f"the median of percentage of native build rules in maven small projects is {build_category_percentages.loc[(build_category_percentages['dataset'] == 'maven-small-projects') & (build_category_percentages['category'] == 'native')]['percentage'].median()}")

    # build_category_percentages = build_category_percentages[build_category_percentages["percentage"] != 0]
    ax = sns.boxplot(data=build_category_percentages, x="dataset", y="percentage", hue="category", palette="Set2",
                     ax=axs[2], dodge=True)

    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_xlabel("")
    ax.set_ylabel("Percentage of Build Rules", fontsize=20)
    ax.set_title("Percentage of Build Rules in Projects", fontsize=20)
    ax.tick_params(labelsize=15)

    sns.move_legend(ax, loc="upper left", title="Category", bbox_to_anchor=(1, 1), fontsize=15)

    plt.tight_layout()
    fig.autofmt_xdate()

    savefig("./images/build_rules")
    plt.show()

    native_rule_stats = scipy.stats.kruskal(build_category_percentages.loc[
                                                (build_category_percentages['dataset'] == 'bazel-projects') & (
                                                        build_category_percentages['category'] == 'native')][
                                                'percentage'],
                                            build_category_percentages.loc[
                                                (build_category_percentages['dataset'] == 'maven-large-projects') & (
                                                        build_category_percentages['category'] == 'native')][
                                                'percentage'],
                                            build_category_percentages.loc[
                                                (build_category_percentages['dataset'] == 'maven-small-projects') & (
                                                        build_category_percentages['category'] == 'native')][
                                                'percentage'])
    print(f"native rule stats: {native_rule_stats}")

    native_rule_stats_posthoc = sp.posthoc_dunn([build_category_percentages.loc[
                                                     (build_category_percentages['dataset'] == 'bazel-projects') & (
                                                             build_category_percentages['category'] == 'native')][
                                                     'percentage'],
                                                 build_category_percentages.loc[
                                                     (build_category_percentages[
                                                          'dataset'] == 'maven-large-projects') & (
                                                             build_category_percentages['category'] == 'native')][
                                                     'percentage'],
                                                 build_category_percentages.loc[
                                                     (build_category_percentages[
                                                          'dataset'] == 'maven-small-projects') & (
                                                             build_category_percentages['category'] == 'native')][
                                                     'percentage']], p_adjust='holm')
    print(f"native rule stats posthoc: {native_rule_stats_posthoc}")

    bazel_to_large_maven_effect_size = cliffs_delta(build_category_percentages.loc[
                                                        (build_category_percentages['dataset'] == 'bazel-projects') & (
                                                                build_category_percentages['category'] == 'native')][
                                                        'percentage'],
                                                    build_category_percentages.loc[
                                                        (build_category_percentages[
                                                             'dataset'] == 'maven-large-projects') & (
                                                                build_category_percentages['category'] == 'native')][
                                                        'percentage'])
    print(f"bazel to large maven effect size: {bazel_to_large_maven_effect_size}")

    bazel_to_small_maven_effect_size = cliffs_delta(build_category_percentages.loc[
                                                        (build_category_percentages['dataset'] == 'bazel-projects') & (
                                                                build_category_percentages['category'] == 'native')][
                                                        'percentage'],
                                                    build_category_percentages.loc[
                                                        (build_category_percentages[
                                                             'dataset'] == 'maven-small-projects') & (
                                                                build_category_percentages['category'] == 'native')][
                                                        'percentage'])
    print(f"bazel to small maven effect size: {bazel_to_small_maven_effect_size}")

    external_rule_stats = scipy.stats.kruskal(build_category_percentages.loc[
                                                  (build_category_percentages['dataset'] == 'bazel-projects') & (
                                                          build_category_percentages['category'] == 'external')][
                                                  'percentage'],
                                              build_category_percentages.loc[
                                                  (build_category_percentages['dataset'] == 'maven-large-projects') & (
                                                          build_category_percentages['category'] == 'external')][
                                                  'percentage'],
                                              build_category_percentages.loc[
                                                  (build_category_percentages['dataset'] == 'maven-small-projects') & (
                                                          build_category_percentages['category'] == 'external')][
                                                  'percentage'])
    print(f"external rule stats: {external_rule_stats}")

    external_rule_stats_posthoc = sp.posthoc_dunn([build_category_percentages.loc[
                                                       (build_category_percentages['dataset'] == 'bazel-projects') & (
                                                               build_category_percentages['category'] == 'external')][
                                                       'percentage'],
                                                   build_category_percentages.loc[
                                                       (build_category_percentages[
                                                            'dataset'] == 'maven-large-projects') & (
                                                               build_category_percentages['category'] == 'external')][
                                                       'percentage'],
                                                   build_category_percentages.loc[
                                                       (build_category_percentages[
                                                            'dataset'] == 'maven-small-projects') & (
                                                               build_category_percentages['category'] == 'external')][
                                                       'percentage']], p_adjust='holm')
    print(f"external rule stats posthoc: {external_rule_stats_posthoc}")


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


def visualize_build_system_invoker(data_dir: str):
    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    build_system_invoker = {"Dataset": []}
    build_system_invoke_methods = {"Dataset": [], "Direct": [0, 0, 0], "Indirect": [0, 0, 0]}

    idx = 0
    for parent_dir_name in parent_dir_names:
        build_system_invoker["Dataset"].append(parent_dir_name)
        build_system_invoke_methods["Dataset"].append(parent_dir_name)

        df = pd.read_csv(f"{data_dir}/{parent_dir_name}-script_usage.csv")

        invokers = df.invoker.value_counts()

        for invoker in invokers.index:
            if invoker not in build_system_invoker:
                build_system_invoker[invoker] = []
            build_system_invoker[invoker].append(invokers[invoker])

            if invoker == "ci":
                build_system_invoke_methods["Direct"][idx] += invokers[invoker]
            else:
                build_system_invoke_methods["Indirect"][idx] += invokers[invoker]
        idx += 1

        for invoker in build_system_invoker:
            if invoker not in invokers.index and invoker != "Dataset":
                build_system_invoker[invoker].append(0)

    print(build_system_invoker)
    build_system_invoke_methods = pd.DataFrame(build_system_invoke_methods)

    build_system_invoke_methods = build_system_invoke_methods.melt(id_vars="Dataset")
    ax = sns.histplot(data=build_system_invoke_methods, x="Dataset", hue="variable", weights="value", multiple="fill",
                      palette=["#fc8d62", "#66c2a5"], shrink=0.8, hue_order=["Indirect", "Direct"])

    for c in ax.containers:
        labels = []
        for p in c.patches:
            height = p.get_height()
            if height == 0:
                labels.append("")
            else:
                labels.append(f"{p.get_height() * 100:.2f}%")

        ax.bar_label(c, labels=labels, label_type='center', fontsize=12)

    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_xlabel("")
    ax.set_xticklabels(["Bazel Projects", "Large Maven Projects", "Small Maven Projects"], fontsize=12)
    ax.set_ylabel("Percentage of Projects", fontsize=12)
    ax.set_title("")
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=12)
    sns.move_legend(ax, loc="upper left", title="", bbox_to_anchor=(1, 1), fontsize=12)

    plt.tight_layout()
    savefig("./images/invoke_methods_usage")
    plt.show()

    build_system_invoker = pd.DataFrame(build_system_invoker)
    build_system_invoker = pd.DataFrame(build_system_invoker)
    build_system_invoker = build_system_invoker.melt(id_vars="Dataset")
    build_system_invoker = build_system_invoker.drop(
        build_system_invoker[build_system_invoker["variable"] == "ci"].index)
    build_system_invoker["variable"] = build_system_invoker["variable"].replace(
        {"shell": "Shell Script File", "make": "Make", "docker": "Docker", "yarn": "Yarn", "npm": "Npm",
         "python": "Python"})
    ax = sns.histplot(data=build_system_invoker, x="Dataset", hue="variable", weights="value", multiple="fill",
                      palette=["#ffd92f", "#a6d854", "#e78ac3", "#8da0cb", "#fc8d62", "#66c2a5"], shrink=0.8,
                      hue_order=["Python", "Npm", "Yarn", "Docker", "Make", "Shell Script File"])

    for c in ax.containers:
        labels = []
        for p in c.patches:
            height = p.get_height()
            if height == 0:
                labels.append("")
            else:
                labels.append(f"{p.get_height() * 100:.2f}%")
        ax.bar_label(c, labels=labels, label_type='center')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_xlabel("")
    ax.set_xticklabels(["Bazel Projects", "Large Maven Projects", "Small Maven Projects"])
    ax.set_ylabel("Percentage of Tools Used To Run Build Systems")
    ax.set_title("")
    ax.set_ylim(0, 1)
    sns.move_legend(ax, loc="upper left", title="", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    savefig("./images/invoker_usage")
    plt.show()


def visualize_parallelization_experiments_by_commits(data_dir):
    experiments = pd.read_csv(f"{data_dir}/parallelization-experiments.csv")
    experiments = experiments.drop(columns=["target", "critical_path"])
    experiments["commits"] = experiments["commits"].astype(int)
    experiments["median_elapsed_time"] = 0

    experiments = experiments.drop(experiments.loc[experiments["subcommand"] == "test"].index)

    # The results of bazelbuild_rules_foreign_cc seems weird, so we temporarily remove it until we figure out what's wrong
    experiments = experiments.drop(experiments[experiments["project"] == "bazelbuild_rules_foreign_cc"].index)
    experiments = experiments.drop(experiments[experiments["project"] == "mukul-rathi_bolt"].index)

    parallelisms = [1, 2, 4, 8, 16]
    for project in experiments["project"].unique():
        for subcommand in ["build", "test"]:
            for parallelism in parallelisms:
                median = experiments.loc[(experiments["project"] == project) & (
                        experiments["subcommand"] == subcommand) & (
                                                 experiments["parallelism"] == parallelism)]["elapsed_time"].median()

                experiments.loc[(experiments["project"] == project) & (
                        experiments["subcommand"] == subcommand) & (
                                        experiments["parallelism"] == parallelism), "median_elapsed_time"] = median

    experiments = experiments.drop(columns=["elapsed_time"]).drop_duplicates()
    experiments["speedup"] = 0
    for project in experiments["project"].unique():
        for subcommand in ["build", "test"]:
            baseline_times = experiments.loc[(experiments["project"] == project) & (
                    experiments["subcommand"] == subcommand) & (
                                                     experiments["parallelism"] == 1)]["median_elapsed_time"]
            if len(baseline_times) == 0:
                continue
            baseline_time = baseline_times.iloc[0]

            for parallelism in parallelisms:
                speedup = baseline_time / experiments.loc[(experiments["project"] == project) & (
                        experiments["subcommand"] == subcommand) & (experiments["parallelism"] == parallelism)][
                    "median_elapsed_time"].iloc[0]

                experiments.loc[(experiments["project"] == project) & (
                        experiments["subcommand"] == subcommand) & (
                                        experiments["parallelism"] == parallelism), "speedup"] = speedup

    commits = sorted(
        experiments.loc[(experiments['subcommand'] == "build")].filter(["project", "commits"]).drop_duplicates()[
            "commits"].tolist())

    small, medium, _ = np.array_split(commits, 3)

    experiments["label"] = experiments.apply(
        lambda row: "small project" if row["commits"] in small else (
            "medium project" if row["commits"] in medium else "large project"), axis=1)

    # ax = sns.histplot(
    #     experiments.loc[(experiments['subcommand'] == "build")].filter(["project", "commits"]).drop_duplicates(),
    #     x="commits")
    # ax.axvline(x=small[-1], color='r')
    # ax.axvline(x=medium[-1], color='r')
    # plt.show()
    #
    # ax = sns.histplot(
    #     experiments.loc[(experiments['subcommand'] == "build")].filter(["project", "commits"]).drop_duplicates(),
    #     x="commits", log_scale=True, bins=15)
    # ax.axvline(x=small[-1], color='r')
    # ax.axvline(x=medium[-1], color='r')
    # plt.show()
    #
    # print(f"small group 0-{small[-1]}")
    # print(f"medium group {small[-1]}-{medium[-1]}")

    for parallelism in parallelisms[1:]:
        print(
            f"the median speedup of small project with parallelism {parallelism} is {experiments.loc[(experiments['label'] == 'small project') & (experiments['parallelism'] == parallelism)]['speedup'].median()}")
        print(
            f"the median speedup of medium project with parallelism {parallelism} is {experiments.loc[(experiments['label'] == 'medium project') & (experiments['parallelism'] == parallelism)]['speedup'].median()}")
        print(
            f"the median speedup of large project with parallelism {parallelism} is {experiments.loc[(experiments['label'] == 'large project') & (experiments['parallelism'] == parallelism)]['speedup'].median()}")
        print(
            f"the median speedup of test with parallelism {parallelism} is {experiments.loc[(experiments['label'] == 'test') & (experiments['parallelism'] == parallelism)]['speedup'].median()}")

    experiments = experiments.drop(experiments[(experiments["parallelism"] == 1)].index)
    experiments = experiments.drop(columns=["subcommand"]).reset_index(drop=True)

    ax = sns.boxplot(data=experiments, x="parallelism", y="speedup",
                     hue_order=['small project', 'medium project', 'large project'], hue="label",
                     palette="Set2")
    ax.set_xlabel("Parallelism")
    ax.set_ylabel("Speedup")
    ax.set_title("")
    ax.axhline(2, ls="--", c="#b3b3b3")
    ax.text(3.53, 1.7, "2x", color="#b3b3b3")
    ax.axhline(4, ls="--", c="#b3b3b3")
    ax.text(3.53, 3.7, "4x", color="#b3b3b3")
    ax.axhline(8, ls="--", c="#b3b3b3")
    ax.text(3.53, 7.7, "8x", color="#b3b3b3")
    ax.axhline(16, ls="--", c="#b3b3b3")
    ax.text(3.53, 15.7, "16x", color="#b3b3b3")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%dx'))
    sns.move_legend(ax, loc="upper left")

    plt.tight_layout()
    savefig("./images/parallelization_experiments_by_commits")
    plt.show()

    small_projects = experiments.loc[experiments["label"] == "small project"]
    medium_projects = experiments.loc[experiments["label"] == "medium project"]
    large_projects = experiments.loc[experiments["label"] == "large project"]

    parallelism2 = scipy.stats.kruskal(small_projects.loc[small_projects["parallelism"] == 2]["speedup"],
                                       medium_projects.loc[medium_projects["parallelism"] == 2]["speedup"],
                                       large_projects.loc[large_projects["parallelism"] == 2]["speedup"])
    print(f"the p-value of speedup with parallelism 2 is {parallelism2}")

    parallelism2_posthoc = sp.posthoc_dunn([small_projects.loc[small_projects["parallelism"] == 2]["speedup"],
                                            medium_projects.loc[medium_projects["parallelism"] == 2]["speedup"],
                                            large_projects.loc[large_projects["parallelism"] == 2]["speedup"]],
                                           p_adjust="holm")

    print(parallelism2_posthoc)

    effect_size_small_large_2 = cliffs_delta(small_projects.loc[small_projects["parallelism"] == 2]["speedup"],
                                             large_projects.loc[large_projects["parallelism"] == 2]["speedup"])
    print(
        f"the effect size of speedup between small and large projects with parallelism 2 is {effect_size_small_large_2}")

    effect_size_medium_large_2 = cliffs_delta(medium_projects.loc[medium_projects["parallelism"] == 2]["speedup"],
                                              large_projects.loc[large_projects["parallelism"] == 2]["speedup"])
    print(
        f"the effect size of speedup between medium and large projects with parallelism 2 is {effect_size_medium_large_2}")

    print("--------------")

    parallelism4 = scipy.stats.kruskal(small_projects.loc[small_projects["parallelism"] == 4]["speedup"],
                                       medium_projects.loc[medium_projects["parallelism"] == 4]["speedup"],
                                       large_projects.loc[large_projects["parallelism"] == 4]["speedup"])
    print(f"the p-value of speedup with parallelism 4 is {parallelism4}")

    parallelism4_posthoc = sp.posthoc_dunn([small_projects.loc[small_projects["parallelism"] == 4]["speedup"],
                                            medium_projects.loc[medium_projects["parallelism"] == 4]["speedup"],
                                            large_projects.loc[large_projects["parallelism"] == 4]["speedup"]],
                                           p_adjust="holm")

    print(parallelism4_posthoc)

    effect_size_small_large_4 = cliffs_delta(small_projects.loc[small_projects["parallelism"] == 4]["speedup"],
                                             large_projects.loc[large_projects["parallelism"] == 4]["speedup"])
    print(
        f"the effect size of speedup between small and large projects with parallelism 4 is {effect_size_small_large_4}")

    effect_size_medium_large_4 = cliffs_delta(medium_projects.loc[medium_projects["parallelism"] == 4]["speedup"],
                                              large_projects.loc[large_projects["parallelism"] == 4]["speedup"])
    print(
        f"the effect size of speedup between medium and large projects with parallelism 4 is {effect_size_medium_large_4}")

    print("--------------")

    parallelism8 = scipy.stats.kruskal(small_projects.loc[small_projects["parallelism"] == 8]["speedup"],
                                       medium_projects.loc[medium_projects["parallelism"] == 8]["speedup"],
                                       large_projects.loc[large_projects["parallelism"] == 8]["speedup"])
    print(f"the p-value of speedup with parallelism 8 is {parallelism8}")

    parallelism8_posthoc = sp.posthoc_dunn([small_projects.loc[small_projects["parallelism"] == 8]["speedup"],
                                            medium_projects.loc[medium_projects["parallelism"] == 8]["speedup"],
                                            large_projects.loc[large_projects["parallelism"] == 8]["speedup"]],
                                           p_adjust="holm")

    print(parallelism8_posthoc)
    print("--------------")

    parallelism16 = scipy.stats.kruskal(small_projects.loc[small_projects["parallelism"] == 16]["speedup"],
                                        medium_projects.loc[medium_projects["parallelism"] == 16]["speedup"],
                                        large_projects.loc[large_projects["parallelism"] == 16]["speedup"])
    print(f"the p-value of speedup with parallelism 16 is {parallelism16}")

    parallelism16_posthoc = sp.posthoc_dunn([small_projects.loc[small_projects["parallelism"] == 16]["speedup"],
                                             medium_projects.loc[medium_projects["parallelism"] == 16]["speedup"],
                                             large_projects.loc[large_projects["parallelism"] == 16]["speedup"]],
                                            p_adjust="holm")

    print(parallelism16_posthoc)
    print("--------------")

    small_projects_stats = scipy.stats.kruskal(small_projects.loc[small_projects["parallelism"] == 2]["speedup"],
                                               small_projects.loc[small_projects["parallelism"] == 4]["speedup"],
                                               small_projects.loc[small_projects["parallelism"] == 8]["speedup"],
                                               small_projects.loc[small_projects["parallelism"] == 16]["speedup"])
    print(f"the p-value of speedup of small projects with parallelism 2, 4, 8, 16 is {small_projects_stats}")

    small_projects_stats_posthoc = sp.posthoc_dunn([small_projects.loc[small_projects["parallelism"] == 2]["speedup"],
                                                    small_projects.loc[small_projects["parallelism"] == 4]["speedup"],
                                                    small_projects.loc[small_projects["parallelism"] == 8]["speedup"],
                                                    small_projects.loc[small_projects["parallelism"] == 16]["speedup"]],
                                                   p_adjust="holm")

    print(small_projects_stats_posthoc)
    print(small_projects_stats_posthoc > 0.01)
    print("--------------")

    medium_projects_stats = scipy.stats.kruskal(medium_projects.loc[medium_projects["parallelism"] == 2]["speedup"],
                                                medium_projects.loc[medium_projects["parallelism"] == 4]["speedup"],
                                                medium_projects.loc[medium_projects["parallelism"] == 8]["speedup"],
                                                medium_projects.loc[medium_projects["parallelism"] == 16]["speedup"])
    print(f"the p-value of speedup of medium projects with parallelism 2, 4, 8, 16 is {medium_projects_stats}")

    medium_projects_stats_posthoc = sp.posthoc_dunn(
        [medium_projects.loc[medium_projects["parallelism"] == 2]["speedup"],
         medium_projects.loc[medium_projects["parallelism"] == 4]["speedup"],
         medium_projects.loc[medium_projects["parallelism"] == 8]["speedup"],
         medium_projects.loc[medium_projects["parallelism"] == 16]["speedup"]], p_adjust="holm")

    print(medium_projects_stats_posthoc)
    print(medium_projects_stats_posthoc > 0.01)
    print("--------------")

    large_projects_stats = scipy.stats.kruskal(large_projects.loc[large_projects["parallelism"] == 2]["speedup"],
                                               large_projects.loc[large_projects["parallelism"] == 4]["speedup"],
                                               large_projects.loc[large_projects["parallelism"] == 8]["speedup"],
                                               large_projects.loc[large_projects["parallelism"] == 16]["speedup"])
    print(f"the p-value of speedup of large projects with parallelism 2, 4, 8, 16 is {large_projects_stats}")

    large_projects_stats_posthoc = sp.posthoc_dunn([large_projects.loc[large_projects["parallelism"] == 2]["speedup"],
                                                    large_projects.loc[large_projects["parallelism"] == 4]["speedup"],
                                                    large_projects.loc[large_projects["parallelism"] == 8]["speedup"],
                                                    large_projects.loc[large_projects["parallelism"] == 16]["speedup"]],
                                                   p_adjust="holm")

    print(large_projects_stats_posthoc)
    print(large_projects_stats_posthoc > 0.01)
    print("--------------")


def visualize_parallelization_experiments_by_build_durations(data_dir):
    experiments = pd.read_csv(f"{data_dir}/parallelization-experiments.csv")
    experiments = experiments.drop(columns=["target"])
    experiments["median_elapsed_time"] = 0

    experiments = experiments.drop(experiments.loc[experiments["subcommand"] == "test"].index)

    # The results of bazelbuild_rules_foreign_cc seems weird, so we temporarily remove it until we figure out what's wrong
    experiments = experiments.drop(experiments[experiments["project"] == "bazelbuild_rules_foreign_cc"].index)
    experiments = experiments.drop(experiments[experiments["project"] == "mukul-rathi_bolt"].index)

    parallelisms = [1, 2, 4, 8, 16]
    for project in experiments["project"].unique():
        for subcommand in ["build", "test"]:
            for parallelism in parallelisms:
                median = experiments.loc[(experiments["project"] == project) & (
                        experiments["subcommand"] == subcommand) & (
                                                 experiments["parallelism"] == parallelism)]["elapsed_time"].median()

                experiments.loc[(experiments["project"] == project) & (
                        experiments["subcommand"] == subcommand) & (
                                        experiments["parallelism"] == parallelism), "median_elapsed_time"] = median

                median_critical_path = experiments.loc[(experiments["project"] == project) & (
                        experiments["subcommand"] == subcommand) & (
                                                               experiments["parallelism"] == parallelism)][
                    "critical_path"].median()
                experiments.loc[(experiments["project"] == project) & (
                        experiments["subcommand"] == subcommand) & (
                                        experiments[
                                            "parallelism"] == parallelism), "critical_path_ratio"] = median_critical_path / median

    experiments = experiments.drop(columns=["elapsed_time", "critical_path"]).drop_duplicates()
    experiments["speedup"] = 0
    for project in experiments["project"].unique():
        for subcommand in ["build", "test"]:
            baseline_times = experiments.loc[(experiments["project"] == project) & (
                    experiments["subcommand"] == subcommand) & (
                                                     experiments["parallelism"] == 1)]["median_elapsed_time"]
            if len(baseline_times) == 0:
                continue
            baseline_time = baseline_times.iloc[0]

            for parallelism in parallelisms:
                speedup = baseline_time / experiments.loc[(experiments["project"] == project) & (
                        experiments["subcommand"] == subcommand) & (experiments["parallelism"] == parallelism)][
                    "median_elapsed_time"].iloc[0]

                experiments.loc[(experiments["project"] == project) & (
                        experiments["subcommand"] == subcommand) & (
                                        experiments["parallelism"] == parallelism), "speedup"] = speedup

    base_build_durations = sorted(
        experiments.loc[(experiments["parallelism"] == 1) & (experiments["subcommand"] == "build")].filter(
            ["project", "median_elapsed_time"]).drop_duplicates()[
            "median_elapsed_time"].unique())

    small_project_durations = base_build_durations[len(base_build_durations) // 3]
    medium_project_durations = base_build_durations[len(base_build_durations) // 3 * 2]

    # ax = sns.histplot(
    #     experiments.loc[(experiments["parallelism"] == 1) & (experiments["subcommand"] == "build")].filter(
    #         ["project", "median_elapsed_time"]).drop_duplicates(),
    #     x="median_elapsed_time")
    # ax.axvline(x=small_project_durations, color='r')
    # ax.axvline(x=medium_project_durations, color='r')
    # plt.show()
    #
    # ax = sns.histplot(
    #     experiments.loc[(experiments["parallelism"] == 1) & (experiments["subcommand"] == "build")].filter(
    #         ["project", "median_elapsed_time"]).drop_duplicates(),
    #     x="median_elapsed_time", bins=15, log_scale=True)
    # ax.axvline(x=small_project_durations, color='r')
    # ax.axvline(x=medium_project_durations, color='r')
    # plt.show()
    #
    # print(f"short {small_project_durations}")
    # print(f"medium {medium_project_durations}")

    experiments["label"] = experiments.apply(
        lambda row: "short build duration" if
        experiments.loc[(experiments["project"] == row["project"]) & (experiments["parallelism"] == 1)][
            "median_elapsed_time"].iloc[0] <= small_project_durations else (
            "medium build duration" if
            experiments.loc[(experiments["project"] == row["project"]) & (experiments["parallelism"] == 1)][
                "median_elapsed_time"].iloc[0] < medium_project_durations else "long build duration"), axis=1)
    experiments.loc[experiments["subcommand"] == "test", "label"] = "test"

    for parallelism in parallelisms[1:]:
        print(
            f"the median speedup of small project with parallelism {parallelism} is {experiments.loc[(experiments['label'] == 'short build duration') & (experiments['parallelism'] == parallelism)]['speedup'].median()}")
        print(
            f"the median speedup of medium project with parallelism {parallelism} is {experiments.loc[(experiments['label'] == 'medium build duration') & (experiments['parallelism'] == parallelism)]['speedup'].median()}")
        print(
            f"the median speedup of large project with parallelism {parallelism} is {experiments.loc[(experiments['label'] == 'long build duration') & (experiments['parallelism'] == parallelism)]['speedup'].median()}")
        print(
            f"the median speedup of test with parallelism {parallelism} is {experiments.loc[(experiments['label'] == 'test') & (experiments['parallelism'] == parallelism)]['speedup'].median()}")

    print("----------------")
    baseline_data = experiments.loc[(experiments["parallelism"] == 1) & (experiments["subcommand"] == "build")]
    for label in ["short build duration", "medium build duration", "long build duration"]:
        print(f"the median baseline build duration of {label} is {baseline_data.loc[baseline_data['label'] == label]['median_elapsed_time'].median()}")
        print(f"the min baseline build duration of {label} is {baseline_data.loc[baseline_data['label'] == label]['median_elapsed_time'].min()}")
        print(f"the max baseline build duration of {label} is {baseline_data.loc[baseline_data['label'] == label]['median_elapsed_time'].max()}")
    ax = sns.boxplot(data=baseline_data, x="label", y="median_elapsed_time", palette="Set2")
    ax.set_yscale("log")
    ax.set_ylabel("Baseline Build Duration in Seconds (Log)")
    ax.set_xlabel("")
    plt.tight_layout()
    savefig("./images/parallelization_baseline_build_duration")
    plt.show()


    experiments = experiments.drop(experiments[(experiments["parallelism"] == 1)].index)
    experiments.drop(columns=["subcommand"], inplace=True)

    ax = sns.boxplot(data=experiments, x="parallelism", y="speedup",
                     hue_order=['short build duration', 'medium build duration', 'long build duration'],
                     hue="label",
                     palette="Set2")
    ax.set_xlabel("Parallelism")
    ax.set_ylabel("Speedup")
    ax.set_title("")
    ax.set_ylim(0, 20)
    ax.axhline(2, ls="--", c="#b3b3b3")
    ax.text(3.53, 1.7, "2x", color="#b3b3b3")
    ax.axhline(4, ls="--", c="#b3b3b3")
    ax.text(3.53, 3.7, "4x", color="#b3b3b3")
    ax.axhline(8, ls="--", c="#b3b3b3")
    ax.text(3.53, 7.7, "8x", color="#b3b3b3")
    ax.axhline(16, ls="--", c="#b3b3b3")
    ax.text(3.53, 15.7, "16x", color="#b3b3b3")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%dx'))

    plt.tight_layout()
    savefig("./images/parallelization_experiments_by_build_duration")
    plt.show()
    #
    #
    # ax = sns.relplot(data=experiments, kind="scatter", col="parallelism", x="critical_path_ratio", col_wrap=2,
    #                  y="speedup", hue="label", palette="Set2")
    # plt.show()

    short_build_duration = experiments.loc[experiments["label"] == "short build duration"]
    medium_build_duration = experiments.loc[experiments["label"] == "medium build duration"]
    long_build_duration = experiments.loc[experiments["label"] == "long build duration"]

    parallelism2 = scipy.stats.kruskal(short_build_duration.loc[short_build_duration["parallelism"] == 2]["speedup"],
                                       medium_build_duration.loc[medium_build_duration["parallelism"] == 2]["speedup"],
                                       long_build_duration.loc[long_build_duration["parallelism"] == 2]["speedup"])
    print(f"the p-value of parallelism 2 is {parallelism2}")

    parallelism2_posthoc = sp.posthoc_dunn(
        [short_build_duration.loc[short_build_duration["parallelism"] == 2]["speedup"],
         medium_build_duration.loc[medium_build_duration["parallelism"] == 2]["speedup"],
         long_build_duration.loc[long_build_duration["parallelism"] == 2]["speedup"]], p_adjust="holm")
    print(f"the posthoc p-values of parallelism 2 are {parallelism2_posthoc}")

    print("--------------")

    parallelism4 = scipy.stats.kruskal(short_build_duration.loc[short_build_duration["parallelism"] == 4]["speedup"],
                                       medium_build_duration.loc[medium_build_duration["parallelism"] == 4]["speedup"],
                                       long_build_duration.loc[long_build_duration["parallelism"] == 4]["speedup"])
    print(f"the p-value of parallelism 4 is {parallelism4}")

    parallelism4_posthoc = sp.posthoc_dunn(
        [short_build_duration.loc[short_build_duration["parallelism"] == 4]["speedup"],
         medium_build_duration.loc[medium_build_duration["parallelism"] == 4]["speedup"],
         long_build_duration.loc[long_build_duration["parallelism"] == 4]["speedup"]], p_adjust="holm")

    print(f"the posthoc p-values of parallelism 4 are {parallelism4_posthoc}")

    parallelism4_short_medium = cliffs_delta(
        short_build_duration.loc[short_build_duration["parallelism"] == 4]["speedup"],
        long_build_duration.loc[long_build_duration["parallelism"] == 4]["speedup"])
    print(
        f"the cliffs_delta effect size of short build time project to long build time project in parallelism 4 is {parallelism4_short_medium}")

    parallelism4_short_medium = cliffs_delta(
        medium_build_duration.loc[medium_build_duration["parallelism"] == 4]["speedup"],
        long_build_duration.loc[long_build_duration["parallelism"] == 4]["speedup"])
    print(
        f"the cliffs_delta effect size of medium build time project to long build time project in parallelism 4 is {parallelism4_short_medium}")

    print("--------------")

    parallelism8 = scipy.stats.kruskal(short_build_duration.loc[short_build_duration["parallelism"] == 8]["speedup"],
                                       medium_build_duration.loc[medium_build_duration["parallelism"] == 8]["speedup"],
                                       long_build_duration.loc[long_build_duration["parallelism"] == 8]["speedup"])
    print(f"the p-value of parallelism 8 is {parallelism8}")

    parallelism8_posthoc = sp.posthoc_dunn(
        [short_build_duration.loc[short_build_duration["parallelism"] == 8]["speedup"],
         medium_build_duration.loc[medium_build_duration["parallelism"] == 8]["speedup"],
         long_build_duration.loc[long_build_duration["parallelism"] == 8]["speedup"]], p_adjust="holm")

    print(f"the posthoc p-values of parallelism 8 are {parallelism8_posthoc}")

    parallelism8_short_medium = cliffs_delta(
        short_build_duration.loc[short_build_duration["parallelism"] == 8]["speedup"],
        long_build_duration.loc[long_build_duration["parallelism"] == 8]["speedup"])
    print(
        f"the cliffs_delta effect size of short build time project to long build time project in parallelism 8 is {parallelism8_short_medium}")

    parallelism8_short_medium = cliffs_delta(
        medium_build_duration.loc[medium_build_duration["parallelism"] == 8]["speedup"],
        long_build_duration.loc[long_build_duration["parallelism"] == 8]["speedup"])
    print(
        f"the cliffs_delta effect size of medium build time project to long build time project in parallelism 8 is {parallelism8_short_medium}")

    print("--------------")

    parallelism16 = scipy.stats.kruskal(short_build_duration.loc[short_build_duration["parallelism"] == 16]["speedup"],
                                        medium_build_duration.loc[medium_build_duration["parallelism"] == 16][
                                            "speedup"],
                                        long_build_duration.loc[long_build_duration["parallelism"] == 16]["speedup"])
    print(f"the p-value of parallelism 16 is {parallelism16}")

    parallelism16_posthoc = sp.posthoc_dunn(
        [short_build_duration.loc[short_build_duration["parallelism"] == 16]["speedup"],
         medium_build_duration.loc[medium_build_duration["parallelism"] == 16]["speedup"],
         long_build_duration.loc[long_build_duration["parallelism"] == 16]["speedup"]], p_adjust="holm")

    print(f"the posthoc p-values of parallelism 16 are {parallelism16_posthoc}")

    parallelism16_short_medium = cliffs_delta(
        short_build_duration.loc[short_build_duration["parallelism"] == 16]["speedup"],
        long_build_duration.loc[long_build_duration["parallelism"] == 16]["speedup"])
    print(
        f"the cliffs_delta effect size of short build time project to long build time project in parallelism 16 is {parallelism16_short_medium}")

    parallelism16_medium_long = cliffs_delta(
        medium_build_duration.loc[medium_build_duration["parallelism"] == 16]["speedup"],
        long_build_duration.loc[long_build_duration["parallelism"] == 16]["speedup"])
    print(
        f"the cliffs_delta effect size of medium build time project to long build time project in parallelism 16 is {parallelism16_medium_long}")

    print("--------------")

    short_build_duration_projects_stats = scipy.stats.kruskal(
        short_build_duration.loc[short_build_duration["parallelism"] == 2]["speedup"],
        short_build_duration.loc[short_build_duration["parallelism"] == 4]["speedup"],
        short_build_duration.loc[short_build_duration["parallelism"] == 8]["speedup"],
        short_build_duration.loc[short_build_duration["parallelism"] == 16]["speedup"])
    print(f"the p-value of short build duration projects is {short_build_duration_projects_stats}")

    short_build_duration_projects_posthoc = sp.posthoc_dunn(
        [short_build_duration.loc[short_build_duration["parallelism"] == 2]["speedup"],
         short_build_duration.loc[short_build_duration["parallelism"] == 4]["speedup"],
         short_build_duration.loc[short_build_duration["parallelism"] == 8]["speedup"],
         short_build_duration.loc[short_build_duration["parallelism"] == 16]["speedup"]], p_adjust="holm")
    print(f"the posthoc p-values of short build duration projects are {short_build_duration_projects_posthoc}")

    duration_parallelism_effect_sizes(short_build_duration, "short build duration")

    print("--------------")

    medium_build_duration_projects_stats = scipy.stats.kruskal(
        medium_build_duration.loc[medium_build_duration["parallelism"] == 2]["speedup"],
        medium_build_duration.loc[medium_build_duration["parallelism"] == 4]["speedup"],
        medium_build_duration.loc[medium_build_duration["parallelism"] == 8]["speedup"],
        medium_build_duration.loc[medium_build_duration["parallelism"] == 16]["speedup"])
    print(f"the p-value of medium build duration projects is {medium_build_duration_projects_stats}")

    medium_build_duration_projects_posthoc = sp.posthoc_dunn(
        [medium_build_duration.loc[medium_build_duration["parallelism"] == 2]["speedup"],
         medium_build_duration.loc[medium_build_duration["parallelism"] == 4]["speedup"],
         medium_build_duration.loc[medium_build_duration["parallelism"] == 8]["speedup"],
         medium_build_duration.loc[medium_build_duration["parallelism"] == 16]["speedup"]], p_adjust="holm")
    print(f"the posthoc p-values of medium build duration projects are {medium_build_duration_projects_posthoc}")

    duration_parallelism_effect_sizes(medium_build_duration, "medium build duration")
    print("--------------")

    long_build_duration_projects_stats = scipy.stats.kruskal(
        long_build_duration.loc[long_build_duration["parallelism"] == 2]["speedup"],
        long_build_duration.loc[long_build_duration["parallelism"] == 4]["speedup"],
        long_build_duration.loc[long_build_duration["parallelism"] == 8]["speedup"],
        long_build_duration.loc[long_build_duration["parallelism"] == 16]["speedup"])
    print(f"the p-value of long build duration projects is {long_build_duration_projects_stats}")

    long_build_duration_projects_posthoc = sp.posthoc_dunn(
        [long_build_duration.loc[long_build_duration["parallelism"] == 2]["speedup"],
         long_build_duration.loc[long_build_duration["parallelism"] == 4]["speedup"],
         long_build_duration.loc[long_build_duration["parallelism"] == 8]["speedup"],
         long_build_duration.loc[long_build_duration["parallelism"] == 16]["speedup"]], p_adjust="holm")

    duration_parallelism_effect_sizes(long_build_duration, "long build duration")
    print(f"the posthoc p-values of long build duration projects are {long_build_duration_projects_posthoc}")
    print("--------------")


def parallelism_confidence_levels(data_dir):
    experiments = pd.read_csv(f"{data_dir}/parallelization-experiments.csv")
    experiments = experiments.drop(columns=["target", "critical_path"])
    experiments = experiments.drop(experiments.loc[experiments["subcommand"] == "test"].index)

    for project in experiments["project"].unique():
        for subcommand in ["build", "test"]:
            mean_baseline_time = experiments.loc[(experiments["project"] == project) & (
                    experiments["subcommand"] == subcommand) & (
                                                         experiments["parallelism"] == 1)]["elapsed_time"].mean()

            experiments.loc[(experiments["project"] == project) & (
                    experiments["subcommand"] == subcommand), "mean_baseline_time"] = mean_baseline_time

            for parallelism in experiments["parallelism"].unique():
                median = experiments.loc[(experiments["project"] == project) & (
                        experiments["subcommand"] == subcommand) & (
                                                 experiments["parallelism"] == parallelism)]["elapsed_time"].median()

                experiments.loc[(experiments["project"] == project) & (
                        experiments["subcommand"] == subcommand) & (
                                        experiments["parallelism"] == parallelism), "median_elapsed_time"] = median

    experiments["speedup"] = experiments.apply(
        lambda row: row["mean_baseline_time"] / row["elapsed_time"], axis=1)

    # commits = sorted(
    #     experiments.loc[(experiments['subcommand'] == "build")].filter(["project", "commits"]).drop_duplicates()[
    #         "commits"].tolist())
    # small, medium, _ = np.array_split(commits, 3)
    #
    # experiments["label"] = experiments.apply(
    #     lambda row: "small project" if row["commits"] in small else (
    #         "medium project" if row["commits"] in medium else "large project"), axis=1)
    # experiments.loc[experiments["subcommand"] == "test", "label"] = "test"

    base_build_durations = sorted(
        experiments.loc[(experiments["parallelism"] == 1) & (experiments["subcommand"] == "build")].filter(
            ["project", "median_elapsed_time"]).drop_duplicates()[
            "median_elapsed_time"].unique())

    small_project_durations = base_build_durations[len(base_build_durations) // 3]
    medium_project_durations = base_build_durations[len(base_build_durations) // 3 * 2]

    experiments["label"] = experiments.apply(
        lambda row: "short build duration" if
        experiments.loc[(experiments["project"] == row["project"]) & (experiments["parallelism"] == 1)][
            "median_elapsed_time"].iloc[0] < small_project_durations else (
            "medium build duration" if
            experiments.loc[(experiments["project"] == row["project"]) & (experiments["parallelism"] == 1)][
                "median_elapsed_time"].iloc[0] < medium_project_durations else "long build duration"), axis=1)

    parallelism_confidence_levels = {}

    parallelisms = [2, 4, 8, 16]
    for project in experiments["project"].unique():
        for subcommand in ["build", "test"]:
            for parallelism in parallelisms:
                rows = experiments.loc[(experiments["project"] == project) & (
                        experiments["subcommand"] == subcommand)]["label"]
                if len(rows) == 0:
                    continue
                label = rows.iloc[0]

                if (label, parallelism) not in parallelism_confidence_levels:
                    parallelism_confidence_levels[(label, parallelism)] = {"c1": [], "c2": [], "m": []}

                data = experiments.loc[(experiments["project"] == project) & (
                        experiments["subcommand"] == subcommand) & (
                                               experiments["parallelism"] == parallelism)]["speedup"].values
                m, c1, c2 = mean_confidence_interval(data)
                parallelism_confidence_levels[(label, parallelism)]["c1"].append(c1)
                parallelism_confidence_levels[(label, parallelism)]["c2"].append(c2)
                parallelism_confidence_levels[(label, parallelism)]["m"].append(m)

    overall_confidence_levels = {}
    for key in parallelism_confidence_levels:
        confidence_intervals = parallelism_confidence_levels[key]

        label, parallelism = key
        total = len(confidence_intervals["m"])
        beyond_parallelism = len([c1 for c1 in confidence_intervals["c1"] if c1 > parallelism])
        below_parallelism = len([c2 for c2 in confidence_intervals["c2"] if c2 < parallelism])
        print(f"{label} {parallelism}x: beyond {beyond_parallelism}/{total} ({beyond_parallelism / total})")
        print(f"{label} {parallelism}x: below {below_parallelism}/{total} ({below_parallelism / total})")

        if parallelism not in overall_confidence_levels:
            overall_confidence_levels[parallelism] = {"total": 0, "beyond_parallelism": 0, "below_parallelism": 0}

        overall_confidence_levels[parallelism]["total"] += total
        overall_confidence_levels[parallelism]["beyond_parallelism"] += beyond_parallelism
        overall_confidence_levels[parallelism]["below_parallelism"] += below_parallelism

    for parallelism in overall_confidence_levels:
        print(
            f'{parallelism}x: beyond {overall_confidence_levels[parallelism]["beyond_parallelism"]}/{overall_confidence_levels[parallelism]["total"]} '
            f'({overall_confidence_levels[parallelism]["beyond_parallelism"] / overall_confidence_levels[parallelism]["total"]})')
        print(
            f"{parallelism}x: below {overall_confidence_levels[parallelism]['below_parallelism']}/{overall_confidence_levels[parallelism]['total']} "
            f"({overall_confidence_levels[parallelism]['below_parallelism'] / overall_confidence_levels[parallelism]['total']})")


def duration_parallelism_effect_sizes(durations, label):
    comparisons = [(2, 4), (2, 8), (2, 16), (4, 8), (4, 16), (8, 16)]
    for comparison in comparisons:
        effect_size = cliffs_delta(
            durations.loc[(durations["parallelism"] == comparison[0])]["speedup"],
            durations.loc[(durations["parallelism"] == comparison[1])]["speedup"])
        print(f"{label}: {comparison[0]}x vs {comparison[1]}x: {effect_size}")


def visualize_parallelism_utilization():
    df = pd.DataFrame({"parallelism": [2, 4, 8, 16],
                       "short build duration": [0.2609, 0.7826, 1, 1],
                       "medium build duration": [0.1739, 0.8261, 0.9130, 0.9565],
                       "long build duration": [0.25, 0.4167, 0.5416, 0.833]})

    df = df.melt(id_vars=["parallelism"])
    ax = sns.barplot(x="parallelism", y="value", hue="variable", data=df, palette="Set2")
    ax.set_xlabel("Parallelism")
    ax.set_ylabel("Percentage of projects unable to utilize build parallelism", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_ylim(0, 1)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.0%}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', xytext=(0, 5),
                    textcoords='offset points')

    sns.move_legend(ax, loc="upper left", title="Group", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    savefig("./images/parallelism_utilization_by_duration")
    plt.show()


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


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
    savefig("./images/build_arg_size")
    plt.show()


def process_cache_experiments_data(data_dir: str):
    if fileutils.exists(f"{data_dir}/cache-experiments-processed.csv"):
        return pd.read_csv(f"{data_dir}/cache-experiments-processed.csv")

    experiments = pd.read_csv(f"{data_dir}/cache-experiments.csv")
    experiments = experiments.drop(columns=["target", "critical_path", "subcommand"])
    experiments = experiments.drop(experiments.loc[experiments["status"] == "failed"].index)
    experiments["processes"] = experiments["processes"].astype(int)

    experiments["median_elapsed_time"] = 0
    for project in experiments["project"].unique():
        for cache_type in ["external", "local", "no_cache", "remote"]:
            first = True
            for commit in experiments.loc[(experiments["project"] == project) & (
                    experiments["cache_type"] == cache_type)]["commit"].unique():
                if first and cache_type != "no_cache":
                    experiments.loc[(experiments["project"] == project) & (
                            experiments["cache_type"] == cache_type) & (
                                            experiments["commit"] == commit), "label"] = "delete"
                    first = False
                    continue

                median = experiments.loc[(experiments["project"] == project) & (
                        experiments["cache_type"] == cache_type) & (
                                                 experiments["commit"] == commit)]["elapsed_time"].median()

                experiments.loc[(experiments["project"] == project) & (
                        experiments["cache_type"] == cache_type) & (
                                        experiments["commit"] == commit), "median_elapsed_time"] = median
            if cache_type == "no_cache":
                experiments.loc[(experiments["project"] == project), "median_baseline"] = experiments.loc[
                    (experiments["project"] == project) & (experiments["cache_type"] == "no_cache")]["median_elapsed_time"].median()

    experiments = experiments.drop(columns=["elapsed_time"]).drop_duplicates()
    experiments = experiments.drop(experiments.loc[experiments["label"] == "delete"].index)

    base_build_durations = sorted(experiments["median_baseline"].unique())

    small_project_durations = base_build_durations[len(base_build_durations) // 3]
    medium_project_durations = base_build_durations[len(base_build_durations) // 3 * 2]

    experiments["label"] = experiments.apply(
        lambda row: "short build duration" if
        experiments.loc[(experiments["project"] == row["project"])][
            "median_baseline"].iloc[0] <= small_project_durations else (
            "medium build duration" if
            experiments.loc[(experiments["project"] == row["project"])][
                "median_baseline"].iloc[0] <= medium_project_durations else "long build duration"), axis=1)

    experiments.to_csv(f"{data_dir}/cache-experiments-processed.csv", index=False)
    return experiments


def visualize_cache_experiments_change_size(data_dir):
    experiments = process_cache_experiments_data(data_dir)

    experiments["cache_hit_ratio"] = experiments.apply(
        lambda row: row["cache_hit"] / row["processes"], axis=1)

    experiments["cache_type"] = experiments["cache_type"].replace(
        {"external": "Local-Deps", "local": "Local-Deps-and-Results",
         "remote": "Remote-Deps-and-Results", "no_cache": "No Cache"})

    cache_hit_rates_data = experiments.loc[
        (experiments["cache_type"] == "Local-Deps-and-Results") | (experiments["cache_type"] == "Remote-Deps-and-Results")]
    ax = sns.violinplot(data=cache_hit_rates_data, x="label", y="cache_hit_ratio", cut=0, palette="Set2")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_ylabel("Cache Hit Rate", fontsize=14)
    ax.set_xlabel("")
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    savefig("./images/cache_experiments_cache_hit_ratio")
    plt.show()

    ax = sns.scatterplot(data=cache_hit_rates_data, x="cache_hit_ratio", y="median_elapsed_time", hue="label",
                         palette="Set2")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_yscale("log")
    ax.set_ylabel("Build Duration in Second (Log)", fontsize=12)
    ax.set_xlabel("Cache Hit Rate", fontsize=12)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    savefig("./images/cache_experiments_cache_hit_ratio_vs_build_duration")
    plt.show()

    sd = experiments.loc[experiments["label"] == "short build duration"]
    print(scipy.stats.mannwhitneyu(sd.loc[sd["cache_type"] == "Local-Deps-and-Results"]["cache_hit_ratio"],
                                sd.loc[sd["cache_type"] == "Remote-Deps-and-Results"]["cache_hit_ratio"]))
    print(cliffs_delta(sd.loc[sd["cache_type"] == "Local-Deps-and-Results"]["cache_hit_ratio"],
                      sd.loc[sd["cache_type"] == "Remote-Deps-and-Results"]["cache_hit_ratio"]))
    print(sd.loc[sd["cache_type"] == "Local-Deps-and-Results"]["cache_hit_ratio"].median())
    print(sd.loc[sd["cache_type"] == "Remote-Deps-and-Results"]["cache_hit_ratio"].median())
    sd_cache = sd.loc[(sd["cache_type"] == "Local-Deps-and-Results") | (sd["cache_type"] == "Remote-Deps-and-Results")]

    md = experiments.loc[experiments["label"] == "medium build duration"]
    print(scipy.stats.mannwhitneyu(md.loc[md["cache_type"] == "Local-Deps-and-Results"]["cache_hit_ratio"],
                              md.loc[md["cache_type"] == "Remote-Deps-and-Results"]["cache_hit_ratio"]))
    print(cliffs_delta(md.loc[md["cache_type"] == "Local-Deps-and-Results"]["cache_hit_ratio"],
                        md.loc[md["cache_type"] == "Remote-Deps-and-Results"]["cache_hit_ratio"]))
    print(md.loc[md["cache_type"] == "Local-Deps-and-Results"]["cache_hit_ratio"].median())
    print(md.loc[md["cache_type"] == "Remote-Deps-and-Results"]["cache_hit_ratio"].median())
    md_cache = md.loc[(md["cache_type"] == "Local-Deps-and-Results") | (md["cache_type"] == "Remote-Deps-and-Results")]

    ld = experiments.loc[experiments["label"] == "long build duration"]
    print(scipy.stats.mannwhitneyu(ld.loc[ld["cache_type"] == "Local-Deps-and-Results"]["cache_hit_ratio"],
                                ld.loc[ld["cache_type"] == "Remote-Deps-and-Results"]["cache_hit_ratio"]))
    print(cliffs_delta(ld.loc[ld["cache_type"] == "Local-Deps-and-Results"]["cache_hit_ratio"],
                        ld.loc[ld["cache_type"] == "Remote-Deps-and-Results"]["cache_hit_ratio"]))
    print(ld.loc[ld["cache_type"] == "Local-Deps-and-Results"]["cache_hit_ratio"].median())
    print(ld.loc[ld["cache_type"] == "Remote-Deps-and-Results"]["cache_hit_ratio"].median())

    ld_cache = ld.loc[(ld["cache_type"] == "Local-Deps-and-Results") | (ld["cache_type"] == "Remote-Deps-and-Results")]

    print(
        f"short build duration projects: median cache hit ratio is {sd_cache['cache_hit_ratio'].median()}")
    print("---------------")
    print(
        f"medium build duration projects: median cache hit ratio is {md_cache['cache_hit_ratio'].median()}")
    print("---------------")
    print(
        f"long build duration projects: median cache hit ratio is {ld_cache['cache_hit_ratio'].median()}")
    print("---------------")

    cache_hit_rate_kw = scipy.stats.kruskal(
        sd_cache["cache_hit_ratio"],
        md_cache["cache_hit_ratio"],
        ld_cache["cache_hit_ratio"])

    print(f"the p-value of cache hit rate is {cache_hit_rate_kw}")

    cache_hit_rate_posthoc = sp.posthoc_dunn(
        [sd_cache["cache_hit_ratio"],
         md_cache["cache_hit_ratio"],
         ld_cache["cache_hit_ratio"]],
        p_adjust="holm"
    )
    print(f"the posthoc of cache hit rate is {cache_hit_rate_posthoc}")

    cache_hit_rate_medium_short = cliffs_delta(
        md_cache["cache_hit_ratio"],
        sd_cache["cache_hit_ratio"])
    print(
        f"the cliffs delta of cache hit rate between medium and short build duration is {cache_hit_rate_medium_short}")

    cache_hit_rate_long_short = cliffs_delta(
        ld_cache["cache_hit_ratio"],
        sd_cache["cache_hit_ratio"])
    print(f"the cliffs delta of cache hit rate between long and short build duration is {cache_hit_rate_long_short}")

    cache_hit_rate_long_medium = cliffs_delta(
        ld_cache["cache_hit_ratio"],
        md_cache["cache_hit_ratio"])
    print(f"the cliffs delta of cache hit rate between long and medium build duration is {cache_hit_rate_long_medium}")

    calculate_cache_hit_correlation(sd_cache, "short build duration")
    calculate_cache_hit_correlation(md_cache, "medium build duration")
    calculate_cache_hit_correlation(ld_cache, "long build duration")

def calculate_cache_hit_correlation(cache_data, label):
    for cache_type in cache_data["cache_type"].unique():
        results = {"no": 0, "small": 0, "medium": 0, "large": 0}
        for project in cache_data["project"].unique():
                cor = scipy.stats.kendalltau(
                    cache_data.loc[(cache_data["project"] == project) & (cache_data["cache_type"] == cache_type)]["cache_hit_ratio"],
                    cache_data.loc[(cache_data["project"] == project) & (cache_data["cache_type"] == cache_type)]["median_elapsed_time"])
                if cor.pvalue >= 0.01:
                    results["no"] += 1
                elif cor.statistic >= 0.5:
                    results["large"] += 1
                elif cor.statistic >= 0.3:
                    results["medium"] += 1
                else:
                    results["small"] += 1

        total = sum(results.values())
        for key in results.keys():
            print(f"{label} - {cache_type}: The percentage of {key} correlation is {results[key] / total}")


def calculate_cache_speed_up(experiments):
    for project in experiments["project"].unique():
        baseline_data = experiments.loc[(experiments["project"] == project) & (
                experiments["cache_type"] == "no_cache")]

        baselines = []
        for row in baseline_data.itertuples():
            if row.status == "failed":
                continue
            baselines.append((row.id, row.median_elapsed_time))

        cache_data = experiments.loc[(experiments["project"] == project) & ~(
                experiments["cache_type"] == "no_cache")]
        for row in cache_data.itertuples():
            baseline_id = row.id // 5
            nearest_baseline_build_time = None
            for baseline in baselines:
                nearest_baseline_build_time = baseline[1]
                if baseline[0] > baseline_id:
                    break

            speedup = nearest_baseline_build_time / row.median_elapsed_time
            experiments.loc[row.Index, "speedup"] = speedup

    return experiments


def visualize_cache_speed_up(data_dir):
    experiments = process_cache_experiments_data(data_dir)
    experiments = calculate_cache_speed_up(experiments)

    experiments["cache_type"] = experiments["cache_type"].replace(
        {"external": "Local-Deps", "local": "Local-Deps-and-Results",
         "remote": "Remote-Deps-and-Results", "no_cache": "No Cache"})

    print("----------------")
    baseline_data = experiments.loc[(experiments["cache_type"] == "No Cache")]
    baseline_data = baseline_data.drop(columns=["cache_hit", "commit", "id", "cache_type", "processes", "size", "commits", "median_elapsed_time"]).drop_duplicates()
    for label in ["short build duration", "medium build duration", "long build duration"]:
        print(f"the median baseline build duration of {label} is {baseline_data.loc[baseline_data['label'] == label]['median_baseline'].median()}")
        print(f"the min baseline build duration of {label} is {baseline_data.loc[baseline_data['label'] == label]['median_baseline'].min()}")
        print(f"the max baseline build duration of {label} is {baseline_data.loc[baseline_data['label'] == label]['median_baseline'].max()}")
    ax = sns.boxplot(data=baseline_data, x="label", y="median_baseline", palette="Set2", order=["short build duration", "medium build duration", "long build duration"])
    ax.set_yscale("log")
    ax.set_ylabel("Baseline Build Duration in Seconds (Log)")
    ax.set_xlabel("")
    plt.tight_layout()
    savefig("./images/cache_baseline_build_duration")
    plt.show()


    experiments = experiments.loc[~(experiments["cache_type"] == "No Cache")]
    ax = sns.boxplot(data=experiments, x="label", y="speedup", hue="cache_type", palette="Set2",
                     hue_order=["Local-Deps", "Local-Deps-and-Results",
                                "Remote-Deps-and-Results"])
    ax.set_ylabel("Speedup")
    ax.set_xlabel("Project Build Duration")
    sns.move_legend(ax, loc="best", title="Cache Strategy")

    plt.tight_layout()
    savefig("./images/cache_speed_up")
    plt.show()

    cache_speedup_statistical_analysis_between_strategy(experiments, "short build duration")
    cache_speedup_statistical_analysis_between_strategy(experiments, "medium build duration")
    cache_speedup_statistical_analysis_between_strategy(experiments, "long build duration")

    cache_speedup_statistical_analysis_between_build_duration(experiments, "Local-Deps")
    cache_speedup_statistical_analysis_between_build_duration(experiments, "Local-Deps-and-Results")
    cache_speedup_statistical_analysis_between_build_duration(experiments, "Remote-Deps-and-Results")


def cache_speedup_statistical_analysis_between_strategy(experiments, label):
    group_data = experiments.loc[experiments["label"] == label]

    print("----------------------------------")
    for cache_type in group_data["cache_type"].unique():
        print(
            f"median of {label} speedup with {cache_type} is {group_data.loc[(group_data['cache_type'] == cache_type)]['speedup'].median()}")

    kw_p_value = scipy.stats.kruskal(
        group_data.loc[(group_data["cache_type"] == "Local-Deps")]["speedup"],
        group_data.loc[(group_data["cache_type"] == "Local-Deps-and-Results")]["speedup"],
        group_data.loc[(group_data["cache_type"] == "Remote-Deps-and-Results")]["speedup"])
    print(f"the p-value of {label} speedup is {kw_p_value}")

    posthoc = sp.posthoc_dunn(
        [group_data.loc[(group_data["cache_type"] == "Local-Deps")]["speedup"],
         group_data.loc[(group_data["cache_type"] == "Local-Deps-and-Results")]["speedup"],
         group_data.loc[(group_data["cache_type"] == "Remote-Deps-and-Results")]["speedup"]],
        p_adjust="holm")
    print(f"the posthoc of {label} speedup is {posthoc}")

    for cache_type1, cache_type2 in itertools.combinations(["Local-Deps", "Local-Deps-and-Results", "Remote-Deps-and-Results"], 2):
        cliffs_delta_value = cliffs_delta(
            group_data.loc[(group_data["cache_type"] == cache_type1)]["speedup"],
            group_data.loc[(group_data["cache_type"] == cache_type2)]["speedup"])
        print(f"the cliffs delta of {label} speedup between {cache_type1} and {cache_type2} is {cliffs_delta_value}")


def cache_speedup_statistical_analysis_between_build_duration(experiments, cache_strategy):
    print("----------------------------------")
    sd = experiments.loc[
        (experiments["label"] == "short build duration") & (experiments["cache_type"] == cache_strategy)]
    md = experiments.loc[
        (experiments["label"] == "medium build duration") & (experiments["cache_type"] == cache_strategy)]
    ld = experiments.loc[
        (experiments["label"] == "long build duration") & (experiments["cache_type"] == cache_strategy)]

    kw_p_value = scipy.stats.kruskal(
        sd["speedup"],
        md["speedup"],
        ld["speedup"])
    print(f"the p-value of {cache_strategy} speedup is {kw_p_value}")

    posthoc = sp.posthoc_dunn(
        [sd["speedup"],
         md["speedup"],
         ld["speedup"]],
        p_adjust="holm")
    print(f"the posthoc of {cache_strategy} speedup is {posthoc}")

    for build_duration1, build_duration2 in itertools.combinations(
            ["short build duration", "medium build duration", "long build duration"], 2):
        cliffs_delta_value = cliffs_delta(
            experiments.loc[(experiments["label"] == build_duration1) & (experiments["cache_type"] == cache_strategy)][
                "speedup"],
            experiments.loc[(experiments["label"] == build_duration2) & (experiments["cache_type"] == cache_strategy)][
                "speedup"])
        print(
            f"the cliffs delta of {cache_strategy} speedup between {build_duration1} and {build_duration2} is {cliffs_delta_value}")


def cache_speedup_confidence_levels(data_dir):
    experiments = process_cache_experiments_data(data_dir)
    experiments = calculate_cache_speed_up(experiments)

    cache_confidence_levels = {}
    for project in experiments["project"].unique():
        for cache_type in ["external", "local", "remote"]:
            rows = experiments.loc[(experiments["project"] == project) & (experiments["cache_type"] == cache_type)][
                "label"]
            if len(rows) == 0:
                continue
            label = rows.iloc[0]

            if (label, cache_type) not in cache_confidence_levels:
                cache_confidence_levels[(label, cache_type)] = {"c1": [], "c2": [], "m": []}

            data = experiments.loc[(experiments["project"] == project) & (experiments["cache_type"] == cache_type)][
                "speedup"]
            m, c1, c2 = mean_confidence_interval(data)
            cache_confidence_levels[(label, cache_type)]["c1"].append(c1)
            cache_confidence_levels[(label, cache_type)]["c2"].append(c2)
            cache_confidence_levels[(label, cache_type)]["m"].append(m)

    for key in cache_confidence_levels:
        intervals = cache_confidence_levels[key]
        label, cache_type = key

        total = len(intervals["c1"])
        better_than_baseline = len([ci for ci in intervals["c1"] if ci > 1])
        worse_than_baseline = len([ci for ci in intervals["c2"] if ci <= 1])
        print(
            f"{label} {cache_type}: better than baseline {better_than_baseline}/{total} ({better_than_baseline / total * 100}%)")
        print(
            f"{label} {cache_type}: worse than baseline {worse_than_baseline}/{total} ({worse_than_baseline / total * 100}%)")
        print("------------------")


def savefig(path, fig_types=("pdf", "png")):
    for fig_type in fig_types:
        plt.savefig(f"{path}.{fig_type}")
