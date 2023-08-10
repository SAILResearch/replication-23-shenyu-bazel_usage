import copy
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
    # visualize_ci_tools(data_dir)
    visualize_subcommand_usage(data_dir)
    visualize_subcommand_intersection(data_dir)
    # visualize_parallelization_usage(data_dir)
    # visualize_cache_usage(data_dir)
    # visualize_build_rule_categories(data_dir)
    # visualize_build_system_invoker(data_dir)
    # visualize_arg_size(data_dir)
    # visualize_parallelization_experiments_by_commits(data_dir)
    # visualize_parallelization_experiments_by_build_durations(data_dir)
    # parallelism_confidence_levels(data_dir)
    # visualize_parallelism_utilization()
    # visualize_cache_experiments_change_size(data_dir)
    # visualize_cache_speed_up(data_dir)


def visualize_ci_tools(data_dir: str):
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 10), sharex=True)
    parent_dir_names = {"bazel-projects": "bazel", "maven-large-projects": "maven", "maven-small-projects": "maven"}
    titles = ["Bazel Projects", "Large Maven Projects", "Small Maven Projects"]
    idx = 0
    for parent_dir_name, correspondent_build_tool in parent_dir_names.items():
        df = pd.read_csv(os.path.join(data_dir, f"{parent_dir_name}-build_tools.csv")).drop(
            columns=["subcommands", "skip_tests"]).drop_duplicates()

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
    labels = ["No Cache",  "CI-Local Cache", "Remote Cache"]
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

    experiments = experiments.drop( experiments.loc[experiments["subcommand"] == "test"].index)


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
                                            large_projects.loc[large_projects["parallelism"] == 2]["speedup"]], p_adjust="holm")

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
                                            large_projects.loc[large_projects["parallelism"] == 8]["speedup"]], p_adjust="holm")

    print(parallelism8_posthoc)
    print("--------------")

    parallelism16 = scipy.stats.kruskal(small_projects.loc[small_projects["parallelism"] == 16]["speedup"],
                                        medium_projects.loc[medium_projects["parallelism"] == 16]["speedup"],
                                        large_projects.loc[large_projects["parallelism"] == 16]["speedup"])
    print(f"the p-value of speedup with parallelism 16 is {parallelism16}")

    parallelism16_posthoc = sp.posthoc_dunn([small_projects.loc[small_projects["parallelism"] == 16]["speedup"],
                                             medium_projects.loc[medium_projects["parallelism"] == 16]["speedup"],
                                             large_projects.loc[large_projects["parallelism"] == 16]["speedup"]], p_adjust="holm")

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

    experiments = experiments.drop( experiments.loc[experiments["subcommand"] == "test"].index)

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
                                        experiments["parallelism"] == parallelism)]["critical_path"].median()
                experiments.loc[(experiments["project"] == project) & (
                        experiments["subcommand"] == subcommand) & (
                                        experiments["parallelism"] == parallelism), "critical_path_ratio"] = median_critical_path / median

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
                "median_elapsed_time"].iloc[0] <= medium_project_durations else "long build duration"), axis=1)
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

    parallelism2_short_medium = cliffs_delta(
        short_build_duration.loc[short_build_duration["parallelism"] == 2]["speedup"],
        medium_build_duration.loc[medium_build_duration["parallelism"] == 2]["speedup"])
    print(
        f"the cliffs_delta effect size of short build time project to medium build time project in parallelism 2 is {parallelism2_short_medium}")

    parallelism2_short_large = cliffs_delta(
        short_build_duration.loc[short_build_duration["parallelism"] == 2]["speedup"],
        long_build_duration.loc[long_build_duration["parallelism"] == 2]["speedup"])
    print(
        f"the cliffs_delta effect size of short build time project to long build time project in parallelism 2 is {parallelism2_short_large}")

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
        medium_build_duration.loc[medium_build_duration["parallelism"] == 4]["speedup"])
    print(
        f"the vda effect size of short build time project to medium build time project in parallelism 4 is {parallelism4_short_medium}")
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
        medium_build_duration.loc[medium_build_duration["parallelism"] == 8]["speedup"])
    print(
        f"the vda effect size of short build time project to medium build time project in parallelism 8 is {parallelism8_short_medium}")
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

    parallelism16_medium_long = cliffs_delta(
        medium_build_duration.loc[medium_build_duration["parallelism"] == 16]["speedup"],
        long_build_duration.loc[long_build_duration["parallelism"] == 2]["speedup"])
    print(
        f"the vda effect size of medium build time project to long build time project in parallelism 16 is {parallelism16_medium_long}")

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
    print(f"the posthoc p-values of long build duration projects are {long_build_duration_projects_posthoc}")
    print("--------------")


def parallelism_confidence_levels(data_dir):
    experiments = pd.read_csv(f"{data_dir}/parallelization-experiments.csv")
    experiments = experiments.drop(columns=["target", "critical_path"])

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
    experiments.loc[experiments["subcommand"] == "test", "label"] = "test"

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
        print(f"{label} {parallelism}x: {beyond_parallelism}/{total} ({beyond_parallelism / total})")

        if parallelism not in overall_confidence_levels:
            overall_confidence_levels[parallelism] = {"total": 0, "beyond_parallelism": 0}

        overall_confidence_levels[parallelism]["total"] += total
        overall_confidence_levels[parallelism]["beyond_parallelism"] += beyond_parallelism

    for parallelism in overall_confidence_levels:
        print(
            f'{parallelism}x: {overall_confidence_levels[parallelism]["beyond_parallelism"]}/{overall_confidence_levels[parallelism]["total"]} '
            f'({overall_confidence_levels[parallelism]["beyond_parallelism"] / overall_confidence_levels[parallelism]["total"]})')


def visualize_parallelism_utilization():
    df = pd.DataFrame({"parallelism": [2, 4, 8, 16],
                       "small project": [0.42, 0.42, 0.28, 0.02],
                       "medium project": [0.40, 0.42, 0.36, 0.1],
                       "large project": [0.12, 0.12, 0.16, 0.12],
                       "test": [0.36, 0.36, 0.20, 0.04]})

    df = df.melt(id_vars=["parallelism"])
    ax = sns.barplot(x="parallelism", y="value", hue="variable", data=df, palette="Set2")
    ax.set_xlabel("Parallelism", fontsize=12)
    ax.set_ylabel("Percentage of projects fully utilized the parallelism", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=12)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.0%}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 5),
                    textcoords='offset points')

    sns.move_legend(ax, loc="upper right", title="Group", fontsize=12)

    plt.tight_layout()
    savefig("./images/parallelism_utilization_by_commits")
    plt.show()

    df = pd.DataFrame({"parallelism": [2, 4, 8, 16],
                       "short build duration": [0.60, 0.64, 0.52, 0.00],
                       "medium build duration": [0.26, 0.22, 0.16, 0.10],
                       "long build duration": [0.08, 0.10, 0.12, 0.14],
                       "test": [0.36, 0.36, 0.20, 0.04]})

    df = df.melt(id_vars=["parallelism"])
    ax = sns.barplot(x="parallelism", y="value", hue="variable", data=df, palette="Set2")
    ax.set_xlabel("Parallelism", fontsize=12)
    ax.set_ylabel("Percentage of projects fully utilized the parallelism", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=12)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.0%}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 5),
                    textcoords='offset points')

    sns.move_legend(ax, loc="upper right", title="Group", fontsize=12)

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
            for commit in experiments.loc[(experiments["project"] == project) & (
                    experiments["cache_type"] == cache_type)]["commit"].unique():
                median = experiments.loc[(experiments["project"] == project) & (
                        experiments["cache_type"] == cache_type) & (
                                                 experiments["commit"] == commit)]["elapsed_time"].median()

                experiments.loc[(experiments["project"] == project) & (
                        experiments["cache_type"] == cache_type) & (
                                        experiments["commit"] == commit), "median_elapsed_time"] = median

    experiments = experiments.drop(columns=["elapsed_time"]).drop_duplicates()

    base_build_durations = sorted(
        experiments.loc[(experiments["cache_type"] == "no_cache")].filter(
            ["project", "median_elapsed_time"]).drop_duplicates()[
            "median_elapsed_time"].unique())

    small_project_durations = base_build_durations[len(base_build_durations) // 3]
    medium_project_durations = base_build_durations[len(base_build_durations) // 3 * 2]

    experiments["label"] = experiments.apply(
        lambda row: "short build duration" if
        experiments.loc[(experiments["project"] == row["project"]) & (experiments["cache_type"] == "no_cache")][
            "median_elapsed_time"].iloc[0] < small_project_durations else (
            "medium build duration" if
            experiments.loc[(experiments["project"] == row["project"]) & (experiments["cache_type"] == "no_cache")][
                "median_elapsed_time"].iloc[0] < medium_project_durations else "long build duration"), axis=1)

    experiments.to_csv(f"{data_dir}/cache-experiments-processed.csv", index=False)
    return experiments


def visualize_cache_experiments_change_size(data_dir):
    experiments = process_cache_experiments_data(data_dir)

    experiments.loc[experiments["size"] == 0, "size"] = 1

    experiments["time_size_ratio"] = experiments.apply(
        lambda row: row["median_elapsed_time"] / row["size"], axis=1)
    experiments["cache_hit_ratio"] = experiments.apply(
        lambda row: row["cache_hit"] / row["processes"], axis=1)

    experiments["cache_type"] = experiments["cache_type"].replace(
        {"external": "CI-enabled Repository Cache", "local": "CI-enabled Repository and Action Cache",
         "remote": "Build System-enabled Repository and Action Cache", "no_cache": "No Cache"})

    ax = sns.boxplot(data=experiments, x="label", y="time_size_ratio", hue="cache_type", palette="Set2",
                     hue_order=["No Cache", "CI-enabled Repository Cache", "CI-enabled Repository and Action Cache",
                                "Build System-enabled Repository and Action Cache"])
    ax.set_ylabel("The ratio between the build time and # of changed lines in commit")
    ax.set_yscale("log")
    ax.set_xlabel("Project Build Duration")
    sns.move_legend(ax, loc="best", title="Cache Strategy")

    plt.tight_layout()
    savefig("./images/cache_experiments_change_size")
    plt.show()

    cache_hit_rates_data = experiments.loc[(experiments["cache_type"] == "CI-enabled Repository and Action Cache") | (experiments["cache_type"] == "Build System-enabled Repository and Action Cache")]
    ax = sns.violinplot(data=cache_hit_rates_data, x="label", y="cache_hit_ratio", cut=0, palette="Set2")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_ylabel("Cache Hit Rate", fontsize=12)
    ax.set_xlabel("Project Build Duration", fontsize=12)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    savefig("./images/cache_experiments_cache_hit_ratio")
    plt.show()

    sd = experiments.loc[experiments["label"] == "short build duration"]
    md = experiments.loc[experiments["label"] == "medium build duration"]
    ld = experiments.loc[experiments["label"] == "long build duration"]

    print(f"short build duration projects: median cache hit ratio is {sd.loc[(sd['cache_type'] == 'CI-enabled Repository and Action Cache') | (sd['cache_type'] == 'Build System-enabled Repository and Action Cache')]['cache_hit_ratio'].median()}")
    print("---------------")
    print(f"medium build duration projects: median cache hit ratio is {md.loc[(md['cache_type'] == 'CI-enabled Repository and Action Cache') | (md['cache_type'] == 'Build System-enabled Repository and Action Cache')]['cache_hit_ratio'].median()}")
    print("---------------")
    print(f"long build duration projects: median cache hit ratio is {ld.loc[(ld['cache_type'] == 'CI-enabled Repository and Action Cache') | (ld['cache_type'] == 'Build System-enabled Repository and Action Cache')]['cache_hit_ratio'].median()}")
    print("---------------")

    cache_hit_rate_kw = scipy.stats.kruskal(sd.loc[(sd["cache_type"] == "CI-enabled Repository and Action Cache") | (sd["cache_type"] == "Build System-enabled Repository and Action Cache")]["cache_hit_ratio"],
                                            md.loc[(md["cache_type"] == "CI-enabled Repository and Action Cache") | (md["cache_type"] == "Build System-enabled Repository and Action Cache")]["cache_hit_ratio"],
                                            ld.loc[(ld["cache_type"] == "CI-enabled Repository and Action Cache") | (ld["cache_type"] == "Build System-enabled Repository and Action Cache")]["cache_hit_ratio"])

    print(f"the p-value of cache hit rate is {cache_hit_rate_kw}")

    cache_hit_rate_posthoc = sp.posthoc_dunn(
        [sd.loc[(sd["cache_type"] == "CI-enabled Repository and Action Cache") | (sd["cache_type"] == "Build System-enabled Repository and Action Cache")]["cache_hit_ratio"],
            md.loc[(md["cache_type"] == "CI-enabled Repository and Action Cache") | (md["cache_type"] == "Build System-enabled Repository and Action Cache")]["cache_hit_ratio"],
            ld.loc[(ld["cache_type"] == "CI-enabled Repository and Action Cache") | (ld["cache_type"] == "Build System-enabled Repository and Action Cache")]["cache_hit_ratio"]],
        p_adjust="holm"
    )
    print(f"the posthoc of cache hit rate is {cache_hit_rate_posthoc}")

    cache_hit_rate_medium_short = cliffs_delta(md.loc[(md["cache_type"] == "CI-enabled Repository and Action Cache") | (md["cache_type"] == "Build System-enabled Repository and Action Cache")]["cache_hit_ratio"],
        sd.loc[(sd["cache_type"] == "CI-enabled Repository and Action Cache") | (sd["cache_type"] == "Build System-enabled Repository and Action Cache")]["cache_hit_ratio"])
    print(f"the cliffs delta of cache hit rate between medium and short build duration is {cache_hit_rate_medium_short}")

    cache_hit_rate_long_short = cliffs_delta(ld.loc[(ld["cache_type"] == "CI-enabled Repository and Action Cache") | (ld["cache_type"] == "Build System-enabled Repository and Action Cache")]["cache_hit_ratio"],
        sd.loc[(sd["cache_type"] == "CI-enabled Repository and Action Cache") | (sd["cache_type"] == "Build System-enabled Repository and Action Cache")]["cache_hit_ratio"])
    print(f"the cliffs delta of cache hit rate between long and short build duration is {cache_hit_rate_long_short}")

    cache_hit_rate_long_medium = cliffs_delta(ld.loc[(ld["cache_type"] == "CI-enabled Repository and Action Cache") | (ld["cache_type"] == "Build System-enabled Repository and Action Cache")]["cache_hit_ratio"],
        md.loc[(md["cache_type"] == "CI-enabled Repository and Action Cache") | (md["cache_type"] == "Build System-enabled Repository and Action Cache")]["cache_hit_ratio"])
    print(f"the cliffs delta of cache hit rate between long and medium build duration is {cache_hit_rate_long_medium}")

    print("---------------")

    print (f"short build duration projects: median of time size ratio for No Cache is {sd.loc[(sd['cache_type'] == 'No Cache')]['time_size_ratio'].median()}")
    print (f"short build duration projects: median of time size ratio for CI-enabled Repository Cache is {sd.loc[(sd['cache_type'] == 'CI-enabled Repository Cache')]['time_size_ratio'].median()}")
    print (f"short build duration projects: median of time size ratio for CI-enabled Repository and Action Cache is {sd.loc[(sd['cache_type'] == 'CI-enabled Repository and Action Cache')]['time_size_ratio'].median()}")
    print (f"short build duration projects: median of time size ratio for Build System-enabled Repository and Action Cache is {sd.loc[(sd['cache_type'] == 'Build System-enabled Repository and Action Cache')]['time_size_ratio'].median()}")

    print("---------------")

    print (f"medium build duration projects: median of time size ratio for No Cache is {md.loc[(md['cache_type'] == 'No Cache')]['time_size_ratio'].median()}")
    print (f"medium build duration projects: median of time size ratio for CI-enabled Repository Cache is {md.loc[(md['cache_type'] == 'CI-enabled Repository Cache')]['time_size_ratio'].median()}")
    print (f"medium build duration projects: median of time size ratio for CI-enabled Repository and Action Cache is {md.loc[(md['cache_type'] == 'CI-enabled Repository and Action Cache')]['time_size_ratio'].median()}")
    print (f"medium build duration projects: median of time size ratio for Build System-enabled Repository and Action Cache is {md.loc[(md['cache_type'] == 'Build System-enabled Repository and Action Cache')]['time_size_ratio'].median()}")

    print("---------------")

    print (f"long build duration projects: median of time size ratio for No Cache is {ld.loc[(ld['cache_type'] == 'No Cache')]['time_size_ratio'].median()}")
    print (f"long build duration projects: median of time size ratio for CI-enabled Repository Cache is {ld.loc[(ld['cache_type'] == 'CI-enabled Repository Cache')]['time_size_ratio'].median()}")
    print (f"long build duration projects: median of time size ratio for CI-enabled Repository and Action Cache is {ld.loc[(ld['cache_type'] == 'CI-enabled Repository and Action Cache')]['time_size_ratio'].median()}")
    print (f"long build duration projects: median of time size ratio for Build System-enabled Repository and Action Cache is {ld.loc[(ld['cache_type'] == 'Build System-enabled Repository and Action Cache')]['time_size_ratio'].median()}")

    print("---------------")


    short_build_duration_change_rate_kw = scipy.stats.kruskal(
        sd.loc[(sd["cache_type"] == "No Cache")]["time_size_ratio"],
        sd.loc[(sd["cache_type"] == "CI-enabled Repository Cache")]["time_size_ratio"],
        sd.loc[(sd["cache_type"] == "CI-enabled Repository and Action Cache")]["time_size_ratio"],
        sd.loc[(sd["cache_type"] == "Build System-enabled Repository and Action Cache")]["time_size_ratio"])
    print(f"the p-value of short build duration change rate is {short_build_duration_change_rate_kw}")

    short_build_duration_change_rate_posthoc = sp.posthoc_dunn(
        [sd.loc[(sd["cache_type"] == "No Cache")]["time_size_ratio"],
            sd.loc[(sd["cache_type"] == "CI-enabled Repository Cache")]["time_size_ratio"],
            sd.loc[(sd["cache_type"] == "CI-enabled Repository and Action Cache")]["time_size_ratio"],
            sd.loc[(sd["cache_type"] == "Build System-enabled Repository and Action Cache")]["time_size_ratio"]],
        p_adjust="holm")
    print(f"the posthoc of short build duration change rate is {short_build_duration_change_rate_posthoc}")

    medium_build_duration_change_rate_kw = scipy.stats.kruskal(
        md.loc[(md["cache_type"] == "No Cache")]["time_size_ratio"],
        md.loc[(md["cache_type"] == "CI-enabled Repository Cache")]["time_size_ratio"],
        md.loc[(md["cache_type"] == "CI-enabled Repository and Action Cache")]["time_size_ratio"],
        md.loc[(md["cache_type"] == "Build System-enabled Repository and Action Cache")]["time_size_ratio"])
    print(f"the p-value of medium build duration change rate is {medium_build_duration_change_rate_kw}")

    medium_build_duration_change_rate_posthoc = sp.posthoc_dunn(
        [md.loc[(md["cache_type"] == "No Cache")]["time_size_ratio"],
            md.loc[(md["cache_type"] == "CI-enabled Repository Cache")]["time_size_ratio"],
            md.loc[(md["cache_type"] == "CI-enabled Repository and Action Cache")]["time_size_ratio"],
            md.loc[(md["cache_type"] == "Build System-enabled Repository and Action Cache")]["time_size_ratio"]],
        p_adjust="holm")

    print(f"the posthoc of medium build duration change rate is {medium_build_duration_change_rate_posthoc}")

    long_build_duration_change_rate_kw = scipy.stats.kruskal(
        ld.loc[(ld["cache_type"] == "No Cache")]["time_size_ratio"],
        ld.loc[(ld["cache_type"] == "CI-enabled Repository Cache")]["time_size_ratio"],
        ld.loc[(ld["cache_type"] == "CI-enabled Repository and Action Cache")]["time_size_ratio"],
        ld.loc[(ld["cache_type"] == "Build System-enabled Repository and Action Cache")]["time_size_ratio"])
    print(f"the p-value of long build duration change rate is {long_build_duration_change_rate_kw}")

    long_build_duration_change_rate_posthoc = sp.posthoc_dunn(
        [ld.loc[(ld["cache_type"] == "No Cache")]["time_size_ratio"],
            ld.loc[(ld["cache_type"] == "CI-enabled Repository Cache")]["time_size_ratio"],
            ld.loc[(ld["cache_type"] == "CI-enabled Repository and Action Cache")]["time_size_ratio"],
            ld.loc[(ld["cache_type"] == "Build System-enabled Repository and Action Cache")]["time_size_ratio"]],
        p_adjust="holm")
    print(f"the posthoc of long build duration change rate is {long_build_duration_change_rate_posthoc}")

def visualize_cache_speed_up(data_dir):
    experiments = process_cache_experiments_data(data_dir)

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

    experiments["cache_type"] = experiments["cache_type"].replace(
        {"external": "CI-enabled Repository Cache", "local": "CI-enabled Repository and Action Cache",
         "remote": "Build System-enabled Repository and Action Cache", "no_cache": "No Cache"})

    experiments = experiments.loc[~(experiments["cache_type"] == "No Cache")]
    ax = sns.boxplot(data=experiments, x="label", y="speedup", hue="cache_type", palette="Set2",
                     hue_order=["CI-enabled Repository Cache", "CI-enabled Repository and Action Cache",
                                "Build System-enabled Repository and Action Cache"])
    ax.set_ylabel("Speedup")
    ax.set_xlabel("Project Build Duration")
    sns.move_legend(ax, loc="best", title="Cache Strategy")

    plt.tight_layout()
    savefig("./images/cache_speed_up")
    plt.show()

    sd = experiments.loc[experiments["label"] == "short build duration"]
    md = experiments.loc[experiments["label"] == "medium build duration"]
    ld = experiments.loc[experiments["label"] == "long build duration"]



    print(f"short build duration projects: median speedup for CI-enabled Repository Cache is {sd.loc[(sd['cache_type'] == 'CI-enabled Repository Cache')]['speedup'].median()}")
    print(f"short build duration projects: median speedup for CI-enabled Repository and Action Cache is {sd.loc[(sd['cache_type'] == 'CI-enabled Repository and Action Cache')]['speedup'].median()}")
    print(f"short build duration projects: median speedup for Build System-enabled Repository and Action Cache is {sd.loc[(sd['cache_type'] == 'Build System-enabled Repository and Action Cache')]['speedup'].median()}")

    print("----------")

    print(f"medium build duration projects: median speedup for CI-enabled Repository Cache is {md.loc[(md['cache_type'] == 'CI-enabled Repository Cache')]['speedup'].median()}")
    print(f"medium build duration projects: median speedup for CI-enabled Repository and Action Cache is {md.loc[(md['cache_type'] == 'CI-enabled Repository and Action Cache')]['speedup'].median()}")
    print(f"medium build duration projects: median speedup for Build System-enabled Repository and Action Cache is {md.loc[(md['cache_type'] == 'Build System-enabled Repository and Action Cache')]['speedup'].median()}")

    print("----------")

    print(f"long build duration projects: median speedup for CI-enabled Repository Cache is {ld.loc[(ld['cache_type'] == 'CI-enabled Repository Cache')]['speedup'].median()}")
    print(f"long build duration projects: median speedup for CI-enabled Repository and Action Cache is {ld.loc[(ld['cache_type'] == 'CI-enabled Repository and Action Cache')]['speedup'].median()}")
    print(f"long build duration projects: median speedup for Build System-enabled Repository and Action Cache is {ld.loc[(ld['cache_type'] == 'Build System-enabled Repository and Action Cache')]['speedup'].median()}")

    print("----------")

    short_build_duration_speedup_kw = scipy.stats.kruskal(
        sd.loc[(sd["cache_type"] == "CI-enabled Repository Cache")]["speedup"],
        sd.loc[(sd["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"],
        sd.loc[(sd["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"])
    print(f"the p-value of short build duration speedup is {short_build_duration_speedup_kw}")

    short_build_duration_speedup_posthoc = sp.posthoc_dunn(
        [sd.loc[(sd["cache_type"] == "CI-enabled Repository Cache")]["speedup"],
            sd.loc[(sd["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"],
            sd.loc[(sd["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"]],
        p_adjust="holm")
    print(f"the posthoc of short build duration speedup is {short_build_duration_speedup_posthoc}")

    medium_build_duration_speedup_kw = scipy.stats.kruskal(
        md.loc[(md["cache_type"] == "CI-enabled Repository Cache")]["speedup"],
        md.loc[(md["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"],
        md.loc[(md["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"])
    print(f"the p-value of medium build duration speedup is {medium_build_duration_speedup_kw}")

    medium_build_duration_speedup_posthoc = sp.posthoc_dunn(
        [md.loc[(md["cache_type"] == "CI-enabled Repository Cache")]["speedup"],
            md.loc[(md["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"],
            md.loc[(md["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"]],
        p_adjust="holm")
    print(f"the posthoc of medium build duration speedup is {medium_build_duration_speedup_posthoc}")

    long_build_duration_speedup_kw = scipy.stats.kruskal(
        ld.loc[(ld["cache_type"] == "CI-enabled Repository Cache")]["speedup"],
        ld.loc[(ld["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"],
        ld.loc[(ld["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"])
    print(f"the p-value of long build duration speedup is {long_build_duration_speedup_kw}")

    long_build_duration_speedup_posthoc = sp.posthoc_dunn(
        [ld.loc[(ld["cache_type"] == "CI-enabled Repository Cache")]["speedup"],
            ld.loc[(ld["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"],
            ld.loc[(ld["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"]],
        p_adjust="holm")
    print(f"the posthoc of long build duration speedup is {long_build_duration_speedup_posthoc}")

    ci_enabled_repository_and_action_cache_kw = scipy.stats.kruskal(
        sd.loc[(sd["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"],
        md.loc[(md["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"],
        ld.loc[(ld["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"])
    print(f"the p-value of CI-enabled Repository and Action Cache speedup is {ci_enabled_repository_and_action_cache_kw}")

    ci_enabled_repository_and_action_cache_posthoc = sp.posthoc_dunn(
        [sd.loc[(sd["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"],
            md.loc[(md["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"],
            ld.loc[(ld["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"]],
        p_adjust="holm")
    print(f"the posthoc of CI-enabled Repository and Action Cache speedup is {ci_enabled_repository_and_action_cache_posthoc}")

    ci_long_medium = cliffs_delta(
        ld.loc[(ld["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"],
        md.loc[(md["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"])
    print(f"the cliffs delta of CI-enabled Repository and Action Cache speedup between long and medium build duration is {ci_long_medium}")

    ci_long_short = cliffs_delta(
        ld.loc[(ld["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"],
        sd.loc[(sd["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"])
    print(f"the cliffs delta of CI-enabled Repository and Action Cache speedup between long and short build duration is {ci_long_short}")

    ci_medium_short = cliffs_delta(
        md.loc[(md["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"],
        sd.loc[(sd["cache_type"] == "CI-enabled Repository and Action Cache")]["speedup"])
    print(f"the cliffs delta of CI-enabled Repository and Action Cache speedup between medium and short build duration is {ci_medium_short}")


    build_system_enabled_repository_and_action_cache_kw = scipy.stats.kruskal(
        sd.loc[(sd["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"],
        md.loc[(md["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"],
        ld.loc[(ld["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"])
    print(f"the p-value of Build System-enabled Repository and Action Cache speedup is {build_system_enabled_repository_and_action_cache_kw}")

    build_system_enabled_repository_and_action_cache_posthoc = sp.posthoc_dunn(
        [sd.loc[(sd["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"],
            md.loc[(md["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"],
            ld.loc[(ld["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"]],
        p_adjust="holm")

    print(f"the posthoc of Build System-enabled Repository and Action Cache speedup is {build_system_enabled_repository_and_action_cache_posthoc}")

    build_long_medium = cliffs_delta(
        ld.loc[(ld["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"],
        md.loc[(md["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"])
    print(f"the cliffs delta of Build System-enabled Repository and Action Cache speedup between long and medium build duration is {build_long_medium}")

    build_long_short = cliffs_delta(
        ld.loc[(ld["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"],
        sd.loc[(sd["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"])
    print(f"the cliffs delta of Build System-enabled Repository and Action Cache speedup between long and short build duration is {build_long_short}")

    build_medium_short = cliffs_delta(
        md.loc[(md["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"],
        sd.loc[(sd["cache_type"] == "Build System-enabled Repository and Action Cache")]["speedup"])

    print(f"the cliffs delta of Build System-enabled Repository and Action Cache speedup between medium and short build duration is {build_medium_short}")



def savefig(path, fig_types=("pdf", "png")):
    for fig_type in fig_types:
        plt.savefig(f"{path}.{fig_type}")
