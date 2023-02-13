import os
import re
from collections.abc import Iterable
from pathlib import Path


def scan_tree(directory: str, file_pattern: re.Pattern, match_path: bool = True) -> Iterable[os.DirEntry]:
    """
    :param directory:  Directory to scan
    :param file_pattern: Regex pattern used to match file. By default, it will match file by its name and path.
    :param match_path: Whether to also use the file_pattern to match file by its path, if isn't set to True, file_pattern is used to only match file name.
    :return: List of files matched by the file pattern under the directory.
    """
    for entry in os.scandir(directory):
        if entry.is_dir():
            yield from scan_tree(entry.path, file_pattern)
        elif entry.is_file() and (file_pattern.match(entry.name) or (match_path and file_pattern.match(entry.path))):
            yield entry


def exists(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        # If it already doesn't exist(),
        # we can skip iterdir().
        return False
    return p in p.parent.iterdir()