import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick


def visualize_data():
    visualize_project_tools()


def visualize_build_rule_categories():
    project_df = pd.read_csv("data/projects.csv")
    project_df["size"] = project_df["size"].div(1024).round(2)

    df = pd.read_csv("data/build_targets.csv")
    fig, axs = plt.subplots(figsize=(15, 10), ncols=2, tight_layout=True)
    df.drop(["name"], axis=1, inplace=True)

    ax = sns.countplot(data=df, x="category", order=df["category"].value_counts().index, ax=axs[0])
    ax.set_title('Build Rule Categories')
    ax.bar_label(ax.containers[0])
    ax.set_ylabel("Number of Build Rules")
    ax.set_xlabel("Category")
    ax.set_xticklabels(["External Build Rule", "Custom Build Rule", "Native Build Rule"])

    build_rules_per_project = df.drop(["category"], axis=1).groupby("project").value_counts().reset_index(name="count")
    build_rules_per_project = pd.merge(build_rules_per_project,
                                       project_df.drop(["stars", "language", "commits"], axis=1),
                                       left_on="project", right_on="project")

    print(f"median value of build rules per project = {build_rules_per_project.median()}")
    print(f"mean value of build rules per project = {build_rules_per_project.mean()}")

    ax = sns.scatterplot(data=build_rules_per_project, x="size", y="count", ax=axs[1])
    ax.set_title('Build Rules and Size per Project')
    ax.set_ylabel("Number of Build Rules")
    ax.set_xlabel("Project Size (MB)")

    plt.savefig("./images/build_rule_categories")
    plt.show()

    build_rule_category_percentages = df.groupby(["project"]).value_counts().reset_index(name="count")

    print(
        f"median value of custom build rules per project = {build_rule_category_percentages[build_rule_category_percentages['category'] == 'custom'].median()}")
    print(
        f"mean value of custom build rules per project = {build_rule_category_percentages[build_rule_category_percentages['category'] == 'custom'].mean()}")
    print(
        f"median value of external build rules per project = {build_rule_category_percentages[build_rule_category_percentages['category'] == 'external'].median()}")
    print(
        f"mean value of external build rules per project = {build_rule_category_percentages[build_rule_category_percentages['category'] == 'external'].mean()}")
    print(
        f"median value of native build rules per project = {build_rule_category_percentages[build_rule_category_percentages['category'] == 'native'].median()}")
    print(
        f"mean value of native build rules per project = {build_rule_category_percentages[build_rule_category_percentages['category'] == 'native'].mean()}")

    fig = plt.figure(figsize=(8, 10), tight_layout=True)

    for p in build_rule_category_percentages["project"].unique():
        total = build_rule_category_percentages[build_rule_category_percentages["project"] == p]["count"].sum()
        build_rule_category_percentages.loc[build_rule_category_percentages["project"] == p, "percentage"] = \
            build_rule_category_percentages[build_rule_category_percentages["project"] == p]["count"] / total

    ax = sns.boxplot(data=build_rule_category_percentages, x="category", y="percentage")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticklabels(["External Build Rule", "Custom Build Rule", "Native Build Rule"])
    ax.set_xlabel("Build Rule Category")
    ax.set_ylabel("Percentage of Build Rules")
    ax.set_title("Percentage of Build Rules per Category in Projects")

    medians = build_rule_category_percentages.groupby(['category'])['percentage'].median().round(3)
    medians[0], medians[1] = medians[1], medians[0]
    vertical_offset = build_rule_category_percentages['percentage'].median() * 0.03  # offset from median for display
    for xtick in ax.get_xticks():
        ax.text(xtick, medians[xtick] + vertical_offset, f"{medians[xtick] * 100}%",
                horizontalalignment='center', size='x-small', color='w', weight='semibold')

    plt.savefig("./images/build_rule_categories_percentage")
    plt.show()


def visualize_project_tools():
    df = pd.read_csv("data/project_tools.csv")
    df.drop('use_test', axis=1, inplace=True)

    bazel_usage_df = pd.DataFrame({"Local": [0, 0, 0, 0, 0, 0, 0], "CI": [0, 0, 0, 0, 0, 0, 0],
                                   "CI/CD Services": ["github-actions", "circleci", "buildkite",
                                                      "buildkite+github-actions", "buildkite+circleci",
                                                      "circleci+github-actions", "buildkite+circleci+github-actions"]})
    for project in df["project"].unique():
        tools = sorted(df[df["project"] == project]["tool"].unique())
        all_true = True
        some_true = False
        for tool in tools:
            used_in_tool = df.query(f"project == '{project}' and tool == '{tool}'")["use_bazel"].iloc[0]
            all_true = all_true and used_in_tool
            some_true = some_true or used_in_tool

        if len(tools) > 1:
            bazel_usage_df.loc[bazel_usage_df["CI/CD Services"] == "+".join(tools), "CI" if all_true else "Local"] += 1
        else:
            bazel_usage_df.loc[bazel_usage_df["CI/CD Services"] == tools[0], "CI" if some_true else "Local"] += 1

    bazel_usage_df = bazel_usage_df.melt(id_vars="CI/CD Services")

    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    ax = sns.histplot(data=bazel_usage_df, x='CI/CD Services', hue='variable', weights='value', discrete=True,
                      multiple='stack')
    for c in ax.containers:
        # remove the labels parameter if it's not needed for customized labels
        ax.bar_label(c, label_type='center')

    ax.set_title('Bazel usage in CI/CD Services')
    ax.set_xlabel("CI/CD Service")
    ax.set_ylabel("Number of Projects")

    fig.autofmt_xdate()

    plt.savefig("./images/bazel_ci_tools")
    plt.show()


1


def draw_test_parallelization_usage():
    data = {"Parallelization": ["Serial", "Test Suite Parallelization", "Test Sharding"],
            "Project number": [0, 164, 59]}
    df = pd.DataFrame(data)
    ax = sns.barplot(data=df, x="Parallelization", y="Project number")

    ax.set_title('Usage of Test Parallelization')
    ax.set_xlabel("Parallelization Strategy")
    ax.set_ylabel("Number of Projects")
    for c in ax.containers:
        ax.bar_label(c, label_type='center')
    plt.savefig("./images/test_parallelization_usage")
    plt.tight_layout()
    plt.show()


def visualize_project_tools_use_test():
    df = pd.read_csv("data/project_tools.csv")

    # new_rows = []
    # for p in df["project"].values:
    #     tools = sorted((df[df["project"] == p])["tool"].values)
    #     if len(tools) > 1:
    #         df.drop(df[df["project"] == p].index, inplace=True)
    #         new_rows.append({"project": p, "tool": "+".join(tools)})
    # df = df.append(new_rows, ignore_index=True)

    unique_projects = df["project"].unique()
    print(f"Number of projects = {len(unique_projects)}")

    fig = plt.figure(figsize=(8, 6))
    ax = sns.histplot(data=df, x="tool", hue="use_test", multiple="stack")
    # ax = sns.countplot(data=df, x="tool", hue="use_test")
    # ax.set_title('CI/CD Tools')
    # ax.bar_label(ax.containers[0])
    # ax.set_ylabel("Number of Projects")
    # ax.set_xlabel("CI/CD Tools")
    # set bar value on top of each bar

    ax.set_ylabel("Number of Projects")
    ax.set_xlabel("CI/CD Tools")
    for c in ax.containers:
        ax.bar_label(c, label_type='center')

    sns.move_legend(ax, loc="center right", title="Whether execute tests")

    fig.autofmt_xdate()
    plt.title('Usages of Bazel in CI/CD Service')
    plt.tight_layout()

    plt.savefig("./images/usage_of_bazel_in_cicd")
    plt.show()


def visualize_test_suite_types():
    df = pd.read_csv("data/test_targets.csv")
    df.loc[df["name"].str.contains("integration"), 'size'] = "medium"
    df.loc[df["name"].str.contains("e2e"), 'size'] = "large"

    df["size"].fillna("Unknown", inplace=True)
    df.drop(df[df["size"] == "Unknown"].index, inplace=True)

    df.loc[df['size'] == "small", 'test_type'] = 'Unit Test'
    df.loc[df['size'] == "medium", 'test_type'] = 'Integration Test'
    df.loc[df['size'] == "large", 'test_type'] = 'End to End Test'
    df.loc[df['size'] == "enormous", 'test_type'] = 'End to End Test'

    df.drop('size', axis=1, inplace=True)

    fig = plt.figure(figsize=(10, 5))
    ax = sns.countplot(data=df, x="test_type", order=df["test_type"].value_counts().index)
    ax.set_title('Test Suite Types')
    ax.set_xlabel("Test Suite Type")
    ax.set_ylabel("Number of Test Targets")
    for c in ax.containers:
        ax.bar_label(c, label_type='center')
    fig.autofmt_xdate()
    plt.savefig("./images/total_test_suite_types")
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(6, 12), tight_layout=True)
    df.drop(["name", "category", "shard_count", "timeout", "flaky"], axis=1, inplace=True)
    test_type_percentages = df.groupby(["project", "test_type"]).value_counts().reset_index(name="count")
    for project in test_type_percentages["project"].unique():
        total = test_type_percentages[test_type_percentages["project"] == project]["count"].sum()
        test_type_percentages.loc[test_type_percentages["project"] == project, "percentage"] = \
            test_type_percentages[test_type_percentages["project"] == project]["count"] / total

    unit_test_percentages = test_type_percentages[test_type_percentages["test_type"] == "Unit Test"].drop(
        ["project", "test_type", "count"], axis=1)
    ax = sns.boxplot(data=unit_test_percentages, ax=axs[0][0])
    ax.set_title("Unit Test Suite Percentage")
    ax.set_ylabel("Percentage of Unit Test Suite")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    unit_test_count = test_type_percentages[test_type_percentages["test_type"] == "Unit Test"].drop(
        ["project", "test_type", "percentage"], axis=1)
    ax = sns.boxplot(data=unit_test_count, ax=axs[0][1])
    ax.set_title("Unit Test Suite Count")
    ax.set_ylabel("Number of Unit Test Suite")
    ax.set_yscale("log")

    integration_test_percentages = test_type_percentages[test_type_percentages["test_type"] == "Integration Test"].drop(
        ["project", "test_type", "count"], axis=1)
    ax = sns.boxplot(data=integration_test_percentages, ax=axs[1][0])
    ax.set_title("Integration Test Suite Percentage")
    ax.set_ylabel("Percentage of Integration Test Suite")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    integration_test_count = test_type_percentages[test_type_percentages["test_type"] == "Integration Test"].drop(
        ["project", "test_type", "percentage"], axis=1)
    ax = sns.boxplot(data=integration_test_count, ax=axs[1][1])
    ax.set_title("Integration Test Suite Count")
    ax.set_ylabel("Number of Integration Test Suite")
    ax.set_yscale("log")

    e2e_test_percentages = test_type_percentages[test_type_percentages["test_type"] == "End to End Test"].drop(
        ["project", "test_type", "count"], axis=1)
    ax = sns.boxplot(data=e2e_test_percentages, ax=axs[2][0])
    ax.set_title("E2E Test Suite Percentage")
    ax.set_ylabel("Percentage of E2E Test Suite")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    e2e_test_count = test_type_percentages[test_type_percentages["test_type"] == "End to End Test"].drop(
        ["project", "test_type", "percentage"], axis=1)
    ax = sns.boxplot(data=e2e_test_count, ax=axs[2][1])
    ax.set_title("E2E Test Suite Count")
    ax.set_ylabel("Number of E2E Test Suite")
    ax.set_yscale("log")

    fig.autofmt_xdate()
    plt.savefig("./images/test_suite_types_distribution")
    plt.tight_layout()
    plt.show()


def draw_bazel_cache():
    # build kite 13, bazelci 42
    df = pd.DataFrame({"Disk Cache": [9, 4, 2], "Remote Cache": [10, 0, 30], "No Cache": [105, 11, 8],
                       "CI/CD Services": ["GitHub Action", "CircleCI", "Buildkite"]})

    df = df.melt(id_vars="CI/CD Services")
    fig = plt.figure(figsize=(10, 10))
    ax = sns.histplot(data=df, x='CI/CD Services', hue='variable', weights='value', discrete=True, multiple='stack')
    ax.set_title('Usage of Bazel Cache')
    for c in ax.containers:
        # Optional: if the segment is small or 0, customize the labels
        labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in c]

        # remove the labels parameter if it's not needed for customized labels
        ax.bar_label(c, labels=labels, label_type='center')
    plt.savefig("./images/usage-of-cache-in-ci-cd")
    plt.show()


def draw_flaky_tests():
    df = pd.read_csv("./data/test_targets.csv")
    df.loc[df["name"].str.contains("integration"), 'size'] = "medium"
    df.loc[df["name"].str.contains("e2e"), 'size'] = "large"

    flaky_projects = df[df["flaky"] == "True"]["project"].unique().size
    non_flaky_projects = df["project"].unique().size - flaky_projects
    flaky_projects_df = pd.DataFrame({"Flaky": [flaky_projects], "Non Flaky": [non_flaky_projects],
                                      "Don't Use Bazel to Test": 456 - flaky_projects - non_flaky_projects})

    fig, axs = plt.subplots(nrows=3, figsize=(8, 20), tight_layout=True)
    ax = sns.barplot(data=flaky_projects_df, ax=axs[0])
    ax.set_title("Flaky Test Projects")
    ax.set_ylabel("Number of Projects")
    for c in ax.containers:
        # remove the labels parameter if it's not needed for customized labels
        ax.bar_label(c, label_type='center')

    df["size"].fillna("Unknown", inplace=True)
    df["flaky"].fillna("False", inplace=True)
    df.drop(df[(df["flaky"] != "True") & (df["flaky"] != "False")].index, inplace=True)

    df.loc[df['size'] == "small", 'test_type'] = 'Unit Test'
    df.loc[df['size'] == "medium", 'test_type'] = 'Integration Test'
    df.loc[df['size'] == "large", 'test_type'] = 'End to End Test'
    df.loc[df['size'] == "enormous", 'test_type'] = 'End to End Test'

    df.drop(["name", "category", "shard_count", "timeout"], axis=1, inplace=True)
    ax = sns.histplot(data=df, x="test_type", hue="flaky", ax=axs[1], multiple="stack", hue_order=["True", "False"])
    ax.set_title("Number of Tests")
    ax.set_ylabel("Number of Tests")
    for c in ax.containers:
        ax.bar_label(c, label_type='center')

    df.drop(df[(df["flaky"] == "False")].index, inplace=True)
    ax = sns.histplot(data=df, x="test_type")

    ax.set_ylabel("Number of Flaky Tests")
    for c in ax.containers:
        ax.bar_label(c, label_type='center')

    # fig.autofmt_xdate()
    plt.savefig("./images/flaky_tests")
    plt.show()

# def draw_parallelism():
#     df = pd.DataFrame({"Project": ["CodeIntelligenceTesting_jazzer", "AcademySoftwareFoundation_openexr",
#                                    "adobe_rules_gitops", "go-resty_resty", "nanopb_nanopb",
#                                    "martian-lang_martian", "google_go-jsonnet", "googleapis_gapic-generator-go",
#                                    "ewish_asciiflow", "Neargye_magic_enum"],
#                        "Test Time": [527.123, 32.016, 0.964, 236.987, 8.385, 70.151, 76.019, 0.524, 8.663, 35.585],
#                        "Critical Path Time": [260.06, 6.11, 0.50, 173.64, 7.93, 23.62, 23.42, 0.16, 6.79, 10.73],
#                        "Theoretical Parallelism": [2.03, 5.24, 1.928, 1.364, 1.06, 2.97, 3.26, 3.28, 1.23, 3.32],
#                        "Actual Parallelism": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]})
