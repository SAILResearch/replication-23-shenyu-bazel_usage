import os
import re
from collections.abc import Iterable


def scan_tree(directory: str, file_pattern: re.Pattern) -> Iterable[os.DirEntry]:

    for entry in os.scandir(directory):
        if entry.is_dir():
            yield from scan_tree(entry.path, file_pattern)
        elif entry.is_file() and (file_pattern.match(entry.name) or file_pattern.match(entry.path)):
            yield entry
