class CIConfigStats:
    def __init__(self):
        self.commands = []
        self.tools = set()


class BazelCommand:
    def __init__(self, tool, command):
        self.tool = tool
        self.command = command
