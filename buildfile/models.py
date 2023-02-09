class BuildFileStats:
    def __init__(self):
        self.build_file_count = 0
        self.rules = []


class BuildRule:
    def __init__(self, name=None, category=None):
        self.name = name
        self.category = category
        self.attrs = {}

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
