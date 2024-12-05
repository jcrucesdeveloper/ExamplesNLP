import ast
from collections import defaultdict
from pytorch.torch import PYTORCH_OPERATIONS

class StaticAnalyzerTensorOperation(ast.NodeVisitor):
    def __init__(self):
        self.operations = defaultdict(int)

    def visit_Call(self, node):

        if isinstance(node.func, ast.Attribute):
            op_name = node.func.attr
            print(op_name)

