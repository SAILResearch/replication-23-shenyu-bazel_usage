import json
import logging
import os
import subprocess

from git import Repo


class Project:
    def __init__(self, num_files: int, num_lines: int):
        self.num_files = num_files
        self.num_lines = num_lines


class ProjectAnalyzer:
    def __init__(self):
        self.cloc_cmd = 'cloc --exclude-dir=".,vendor,dist,node_modules,target" --exclude-lang="XML,Text,JSON,YAML,Starlark,Bazel,CSV,Markdown,SVG,make,CMake,Bourne Shell,PowerShell,DOS Batch" '

    def analyze_project(self, project_dir) -> Project:
        proc = subprocess.run([f"{self.cloc_cmd} {project_dir} --json"], capture_output=True, text=True, shell=True)
        if proc.returncode != 0:
            raise Exception(
                f"error when examining the source files for project {project_dir}, reason {proc.stderr}")
        results = json.loads(proc.stdout)
        if "SUM" not in results:
            raise Exception(
                f"error when parse the results returned by cloc, reason no key 'SUM' exists in\n{results}"
            )

        num_files = results["SUM"]["nFiles"]
        num_lines = results["SUM"]["code"]

        return Project(num_files, num_lines)

    def analyze_projects(self, project_base_dir: str, output_dir: str):
        with open(f"{output_dir}/projects.csv", "w") as project_file:
            project_file.write("project#num_files#num_lines\n")

            for entry in os.scandir(project_base_dir):
                if not entry.is_dir():
                    continue

                logging.info(f"starting to analyze source files for project {entry.name}")
                project = self.analyze_project(entry.path)

                project_file.write(f"{entry.name}#{project.num_files}#{project.num_lines}\n")
