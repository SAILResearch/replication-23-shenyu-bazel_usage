import os
import re

from collections.abc import Iterable

from buildfile.models import BuildFileStats, BuildRule
from utils import utils

native_rules = ["extra_action", "action_listener", "filegroup", "genquery", "test_suite", "alias", "config_setting",
               "genrule", "constraint_setting", "constraint_value", "platform", "toolchain", "toolchain_type", "bind",
               "local_repository", "new_local_repository", "xcode_config", "xcode_version"]


class BuildFileParser:
    def __init__(self, build_file_pattern: str = "^BUILD(\\.bazel)?$"):
        self.build_file_pattern = re.compile(build_file_pattern)
        self.custom_rule_file_pattern = re.compile(r"^.*\.bzl$")
        self.build_rule_pattern = re.compile(r"^([a-z][a-z0-9_]+)\($")
        self.custom_build_rule_pattern = re.compile(r"def (.*)\(")
        # We don't parse multi-line attribute as it makes the parsing process too complicated and the current
        # research project doesn't need it.
        self.attr_pattern = re.compile(r"\s+([a-z0-9_]+)\s*=\s*\"?([^\"]*)\"?,")

    def __scan_build_files(self, directory: str) -> Iterable[os.DirEntry]:
        yield from utils.scan_tree(directory, self.build_file_pattern)

    def __scan_custom_rule_files(self, directory) -> Iterable[os.DirEntry]:
        yield from utils.scan_tree(directory, self.custom_rule_file_pattern)

    def __scan_custom_rules(self, directory) -> set:
        possible_custom_rules = set()

        for entry in self.__scan_custom_rule_files(directory):
            with open(entry.path) as custom_rule_file:
                while line := custom_rule_file.readline():
                    line = line.rstrip()
                    if not line:
                        continue

                    if matches := self.custom_build_rule_pattern.match(line):
                        possible_custom_rules.add(matches.group(1))
        return possible_custom_rules

    def parse(self, directory: str) -> BuildFileStats:
        possible_custom_rules = self.__scan_custom_rules(directory)

        build_file_stats = BuildFileStats()

        build_file_count = 0
        for entry in self.__scan_build_files(directory):
            if entry.path.startswith(
                    (os.path.join(directory, "."), os.path.join(directory, "bazel-"), os.path.join(directory, "dist"), os.path.join(directory, "vendor"))):
                continue

            build_file_count += 1

            rule = None
            with open(entry.path) as build_file:
                while line := build_file.readline():
                    line = line.rstrip()
                    if not line:
                        continue

                    matches = self.build_rule_pattern.match(line)
                    # if there is no match of rule name, and it is not in the rule block, skip
                    if not matches and not rule:
                        continue

                    if matches:
                        rule_name = matches.group(1)
                        if rule_name == "load":
                            continue

                        # TODO improve the way to determine if a rule is a external rule
                        # In Bazel, it will use load statement to declare the external and custom rules used in
                        # the build file. We can use this information to determine if a rule is a external rule.
                        rule_type = "external"
                        if rule_name in native_rules:
                            rule_type = "native"
                        elif rule_name in possible_custom_rules:
                            rule_type = "custom"

                        if rule:
                            build_file_stats.rules.append(rule)

                        rule = BuildRule(name=rule_name, category=rule_type)
                    else:
                        matches = self.attr_pattern.match(line)
                        if matches:
                            rule.attrs[matches.group(1)] = matches.group(2)

                if rule:
                    build_file_stats.rules.append(rule)

        build_file_stats.build_file_count = build_file_count
        return build_file_stats
