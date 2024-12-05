import ast
from collections import defaultdict
from operations.torch import PYTORCH_OPERATIONS

class StaticAnalyzerTensorOperation(ast.NodeVisitor):
    def __init__(self):
        pass  # No need for operations tracking anymore

    def generic_visit(self, node):
        print(f"Node type: {type(node).__name__}")
        print(f"Node fields: {ast.dump(node)}")
        print("-" * 50)
        super().generic_visit(node)
