import os
import re

from collections.abc import Iterable

from ciconfig.models import CIConfigStats, BazelCommand
from utils import utils


class CIConfigParser:
    def __init__(self):
        self.github_action_pattern = re.compile(re.compile(r'.*\.github/workflows/.*\.ya?ml$'))
        self.bazelrc_pattern = re.compile(r"^\.bazelrc$")
        self.circleci_pattern = re.compile(r".*\.circleci/.*\.ya?ml$")
        self.buildkite_pattern = re.compile(r".*\.(buildkite|bazelci)/.*\.ya?ml$")
        # self.bazel_command_pattern = re.compile(r"bazel\s+(.*)")
        self.bazel_command_pattern = re.compile(r".*[Bb]azel\s*(.*)")

    def __scan_github_action_files(self, directory: str) -> Iterable[os.DirEntry]:
        yield from utils.scan_tree(directory, self.github_action_pattern)

    def __scan_circleci_files(self, directory: str) -> Iterable[os.DirEntry]:
        yield from utils.scan_tree(directory, self.circleci_pattern)

    def __scan_buildkite_files(self, directory: str) -> Iterable[os.DirEntry]:
        yield from utils.scan_tree(directory, self.buildkite_pattern)

    def __scan_bazelrc_files(self, directory: str) -> Iterable[os.DirEntry]:
        yield from utils.scan_tree(directory, self.bazelrc_pattlern)

    def __parse_bazel_commands(self, directory, file) -> [str]:
        if file.path.startswith(
                (os.path.join(directory, "bazel-"), os.path.join(directory, "dist"),
                 os.path.join(directory, "vendor"))):
            return []

        with open(file.path, "r") as f:
            content = f.read()
            if ".bazelci" in file.path:
                return [content]

            commands = []

            for match in self.bazel_command_pattern.finditer(content):
                command = match.group(1)
                commands.append(command)

        return commands

    def parse(self, directory: str) -> CIConfigStats:
        stats = CIConfigStats()

        for file in self.__scan_github_action_files(directory):
            stats.tools.add("github-actions")
            for command in self.__parse_bazel_commands(directory, file):
                stats.commands.append(BazelCommand("github-actions", command))

        for file in self.__scan_circleci_files(directory):
            stats.tools.add("circleci")
            for command in self.__parse_bazel_commands(directory, file):
                stats.commands.append(BazelCommand("circleci", command))

        for file in self.__scan_buildkite_files(directory):
            stats.tools.add("buildkite")
            for command in self.__parse_bazel_commands(directory, file):
                stats.commands.append(BazelCommand("buildkite", command))

        # scan bazelrc files
        return stats
