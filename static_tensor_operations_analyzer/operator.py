class Operator:
    def __init__(self, name: str, input: str, output: str, description: str = None):
        self.name = name
        self.input = input 
        self.output = output
        self.description = description

    def __str__(self):
        return f"{self.name}: {self.input} -> {self.output}"

