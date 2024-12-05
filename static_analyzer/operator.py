class Operator:
    def __init__(self, name: str, input: list[str], output: str, description: str = None):
        self.name = name
        self.input = input
        self.output = output
        self.description = description

    def __str__(self):
        inputs_str = ", ".join(self.input)
        return f"{self.name} | [{inputs_str}] → {self.output}"


if __name__ == "__main__":
    op = Operator(
        name="tensor",
        input=["data", "dtype=None", "device=None", "requires_grad=False", "pin_memory=False"],
        output="Tensor"
    )
    print(op)
