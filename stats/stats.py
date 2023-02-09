import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats


def analyze_build_targets():
    df = pd.read_csv("data/targets_analysis.csv")

    fig, axs = plt.subplots(ncols=5, figsize=(30, 15), tight_layout=True)
    ax = sns.boxplot(x="critical_path", y="in_degree", data=df, ax=axs[0], showfliers=False)
    ax.set_title("In-degree")
    ax.set_ylabel("# of In-degree")
    ax.set_xticklabels(["Targets in non-critical path", "Targets in critical path"])

    medians = df.groupby(['critical_path'])['in_degree'].median().round(3)
    vertical_offset = df['in_degree'].median() * 0.03  # offset from median for display
    for xtick in ax.get_xticks():
        ax.text(xtick, medians[xtick] + vertical_offset, medians[xtick],
                horizontalalignment='center', size='large', color='w', weight='semibold')

    ax = sns.boxplot(x="critical_path", y="out_degree", data=df, ax=axs[1], showfliers=False)
    ax.set_title("Out-degree")
    ax.set_ylabel("# of Out-degree")
    ax.set_xticklabels(["Targets in non-critical path", "Targets in critical path"])
    medians = df.groupby(['critical_path'])['out_degree'].median().round(3)
    vertical_offset = df['out_degree'].median() * -0.05  # offset from median for display
    for xtick in ax.get_xticks():
        ax.text(xtick, medians[xtick] + vertical_offset, medians[xtick],
                horizontalalignment='center', size='large', color='w', weight='semibold')

    ax = sns.boxplot(x="critical_path", y="number_of_dependencies", data=df, ax=axs[2], showfliers=False)
    ax.set_title("Number of Transitive Dependencies")
    ax.set_ylabel("# of Dependencies")
    ax.set_xticklabels(["Targets in non-critical path", "Targets in critical path"])
    medians = df.groupby(['critical_path'])['number_of_dependencies'].median().round(3)
    vertical_offset = df['number_of_dependencies'].median() * 0.03  # offset from median for display
    for xtick in ax.get_xticks():
        ax.text(xtick, medians[xtick] + vertical_offset, medians[xtick],
                horizontalalignment='center', size='large', color='w', weight='semibold')

    ax = sns.boxplot(x="critical_path", y="number_of_dependents", data=df, ax=axs[3], showfliers=False)
    ax.set_title("Number of Transitive Dependents")
    ax.set_ylabel("# of Dependents")
    ax.set_xticklabels(["Targets in non-critical path", "Targets in critical path"])
    medians = df.groupby(['critical_path'])['number_of_dependents'].median().round(3)
    vertical_offset = df['number_of_dependents'].median() * 0.03  # offset from median for display
    for xtick in ax.get_xticks():
        ax.text(xtick, medians[xtick] + vertical_offset, medians[xtick],
                horizontalalignment='center', size='large', color='w', weight='semibold')

    ax = sns.boxplot(x="critical_path", y="num_of_source_files", data=df, ax=axs[4], showfliers=False)
    ax.set_title("Number of Source Files")
    ax.set_ylabel("# of Source Files")
    ax.set_xticklabels(["Targets in non-critical path", "Targets in critical path"])
    medians = df.groupby(['critical_path'])['num_of_source_files'].median().round(3)
    vertical_offset = df['num_of_source_files'].median() * 0.01  # offset from median for display
    for xtick in ax.get_xticks():
        ax.text(xtick, medians[xtick] + vertical_offset, medians[xtick],
                horizontalalignment='center', size='large', color='w', weight='semibold')

    fig.autofmt_xdate()
    plt.savefig("images/targets_analysis")
    plt.show()
