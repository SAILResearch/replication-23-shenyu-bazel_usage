import logging
import os
import re

from abc import abstractmethod
from pathlib import Path
from typing import Any

from lark import Lark, Transformer, Token
from lark.exceptions import UnexpectedToken
from lark.indenter import PythonIndenter

from lxml import etree
from lxml.etree import XMLSyntaxError

from utils import fileutils


class BuildRule:
    def __init__(self, name: str, category: str, args: list[Any]):
        self.name = name
        self.category = category
        self.args = args


class BuildFileConfig:
    def __init__(self, build_tool: str):
        self.build_tool = build_tool
        self.rules: [BuildRule] = []


class BuildFileParser:
    def __init__(self, project_dir: str):
        self.build_file_pattern = re.compile(self._build_file_pattern())
        self.project_dir = project_dir

    def _scan_build_files(self) -> [os.DirEntry]:
        possible_build_files = []
        for f in fileutils.scan_tree(self.project_dir, self.build_file_pattern, match_path=False):
            rel_path = str(Path(f.path).relative_to(self.project_dir))
            # Skipping the parsing for a build file if it satisfied the following criteria:
            if rel_path.startswith((".", "bazel-", "vendor", "dist", "node_modules", "target")):
                continue
            possible_build_files.append(f)
        return possible_build_files

    @abstractmethod
    def _build_file_pattern(self) -> str:
        pass

    def parse(self) -> [BuildFileConfig]:
        build_configs = []
        for build_file in self._scan_build_files():
            with open(build_file.path, "r") as f:
                try:
                    build_file_str = f.read()
                except UnicodeDecodeError as e:
                    logging.warning(f"skipping {build_file.path} due to UnicodeDecodeError, reason {e}")
                    continue
            print(f"parsing file {build_file.path}")
            build_config = self.parse_build_file(build_file_str)
            if not build_config:
                continue
            build_configs.append(build_config)
        return build_configs

    @abstractmethod
    def parse_build_file(self, build_file: str) -> BuildFileConfig:
        pass

    @abstractmethod
    def build_tool_type(self) -> str:
        pass


class StarlarkTransformer(Transformer):
    list = list

    def string(self, s: list[Any]) -> str:
        (s,) = s
        return s[1:-1]

    def getattr(self, attr) -> str:
        prefix = ".".join(attr[0]) if type(attr[0]) is list else attr[0]
        return f"{prefix}.{attr[1]}"

    def name(self, n: list[Any]) -> str:
        return str(n[0])

    def arguments(self, args: list[Any]) -> list[Any]:
        return args

    def argvalue(self, arg: list[Any]) -> dict[Token, Any]:
        # TODO recheck this part as the current implementation may throw outofindex exception
        name = arg[0][0][0] if type(arg[0][0]) is list and len(arg[0][0]) > 0 else arg[0][0]
        val = arg[1]
        return {name: val}

    def number(self, n: list[Any]) -> float:
        (n,) = n
        if type(n) is Token:
            if n.type in ["DEC_NUMBER", "FLOAT_NUMBER"]:
                return float(n.value)
            if n.type == "HEX_NUMBER":
                return int(n.value, 16)
            if n.type == "OCT_NUMBER":
                return int(n.value, 8)
            if n.type == "BIN_NUMBER":
                return int(n.value, 2)
            return float(n.value)

        return float(n[1:-1]) if type(n) is not str else float(n)

    def var(self, var: Token) -> Token:
        return var

    def funccall(self, vals: list[Any]) -> dict[str, Any]:
        name = vals[0][0] if type(vals[0]) is list and len(vals[0]) > 0 else vals[0]
        return {"name": name, "args": vals[1], "type": "funccall"}


def starlark_parsing_on_errors(e: UnexpectedToken):
    return False


class BazelBuildFileParser(BuildFileParser):
    def __init__(self, project_dir: str):
        super().__init__(project_dir)
        kwargs = dict(postlex=PythonIndenter(), start='file_input')
        # starlark is syntactically a strict subset of python,
        # so we should be able to use python parser to parse the starlark codes.
        self.starlark_parser = Lark.open_from_package("lark", "python.lark", ["grammars"], parser="lalr",
                                                      transformer=StarlarkTransformer(), **kwargs)

    def _build_file_pattern(self) -> str:
        return r"^BUILD(\.bazel)$"

    # For a Bazel build file, its AST structure typically likes this.
    # So, we will only parse the expr_stmt node that has the node funccall and under the root node.
    # file_input
    #   expr_stmt
    #     funccall
    #       var
    #         name    load
    #       arguments
    #         string  "@rules_python//python:defs.bzl"
    #   expr_stmt
    #     funccall
    #       var
    #         name    py_binary
    #       arguments
    #         argvalue
    #           var
    #             name        name
    #           string        "spec_md_gen"
    #         argvalue
    #           var
    #             name        srcs
    #           list
    #             string      "spec_md_gen.py"
    #             string      "spec_md_gen_lib.py"
    def parse_build_file(self, build_file_str: str) -> BuildFileConfig:
        # Lark parser throws error if the text doesn't end with a new line, so we manually add one if it doesn't exist.
        # See: https://github.com/lark-parser/lark/issues/139 and https://github.com/lark-parser/lark/issues/237
        if not build_file_str.endswith("\n"):
            build_file_str += "\n"

        build_file_str = build_file_str.replace("\\\n", "")
        # TODO fix it in the future
        build_file_str = build_file_str.replace('if target != "1.13"', "")

        ast_tree = self.starlark_parser.parse(build_file_str, on_error=starlark_parsing_on_errors)

        build_rules = []
        third_party_rule_names = []
        custom_rule_names = []

        for c in ast_tree.children:
            if c.data != "expr_stmt" or len(c.children) == 0:
                continue

            funccall = c.children[0]
            if type(funccall) is not dict or funccall["type"] != "funccall":
                continue

            name = funccall["name"]
            args = funccall["args"]
            if name == "load":
                loaded_symbols = []
                for i in range(1, len(args)):
                    if type(args[i]) is str:
                        loaded_symbols.append(args[i])
                    elif type(args[i]) is dict:
                        loaded_symbols.extend(args[i].keys())

                load_path = args[0]
                # If the load path starts with // (absolute path) or : (relative path),
                # it looks up .bzl files in the same project. So, these rules are custom rules.
                if load_path.startswith(("//", ":")):
                    custom_rule_names.extend(loaded_symbols)
                elif load_path.startswith("@"):
                    # TODO we need to check if the external rules under the same group as the current project.
                    third_party_rule_names.extend(loaded_symbols)
                else:
                    logging.warning(f"unknown load type: {load_path}, {loaded_symbols}")

            category = "native"
            if name in third_party_rule_names:
                category = "external"
            elif name in custom_rule_names:
                category = "custom"
            build_rules.append(BuildRule(name, category, args))

        bf_config = BuildFileConfig(self.build_tool_type())
        bf_config.rules.extend(build_rules)
        return bf_config

    def build_tool_type(self) -> str:
        return "bazel"


class MavenBuildFileParser(BuildFileParser):

    def __init__(self, project_dir: str):
        super().__init__(project_dir)

    def _build_file_pattern(self) -> str:
        return r"^pom.xml$"

    # TODO process parent pom?
    def parse_build_file(self, build_file_str: str) -> BuildFileConfig:
        try:
            root = etree.fromstring(bytes(build_file_str, encoding='utf-8'))
        except XMLSyntaxError as e:
            logging.warning(f"error when parsing the xml file, reason {e}")
            return None
        except UnicodeDecodeError as e:
            logging.warning(f"error when using utf-8 to parse the xml file, reason {e}")
            return None

        ns = {"pom": "http://maven.apache.org/POM/4.0.0"}

        project_group_id_res = root.xpath("/pom:project/pom:groupId/text()", namespaces=ns)
        if not project_group_id_res:
            parent_group_id_res = root.xpath("/pom:project/pom:parent/pom:groupId/text()", namespaces=ns)
            if parent_group_id_res:
                project_group_id = parent_group_id_res[0]
            else:
                logging.error(
                    f"{self.project_dir} is not a valid Maven projects, reason: cannot find groupId in the project.")
                return None
        else:
            project_group_id = project_group_id_res[0]

        build_rules = []

        plugins = root.xpath("/pom:project/pom:build/pom:plugins/pom:plugin", namespaces=ns)
        for plugin in plugins:
            plugin_group_id_res = plugin.xpath("./pom:groupId/text()", namespaces=ns)
            plugin_artifact_id_res = plugin.xpath("./pom:artifactId/text()", namespaces=ns)
            if not plugin_group_id_res or not plugin_artifact_id_res:
                continue

            plugin_group_id = plugin_group_id_res[0]
            plugin_artifact_id = plugin_artifact_id_res[0]

            category = "external"
            if plugin_group_id == "org.apache.maven.plugins":
                category = "native"
            elif plugin_group_id == project_group_id:
                category = "custom"

            # TODO parse the arguments
            build_rules.append(BuildRule(f"{plugin_group_id}:{plugin_artifact_id}", category, []))

        bf_config = BuildFileConfig(self.build_tool_type())
        bf_config.rules.extend(build_rules)
        return bf_config

    def build_tool_type(self) -> str:
        return "maven"
