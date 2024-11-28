import os
import ast
import nbformat
import re
from collections import defaultdict

class TensorOperationAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.operations = defaultdict(int)
        
    def visit_Call(self, node):
        # Check for tensor/array operations
        if isinstance(node.func, ast.Attribute):
            op_name = node.func.attr
            # PyTorch operations
            if op_name in ['matmul', 'mm', 'bmm', 'dot', 'mul', 'add', 'sub', 'div',
                          'transpose', 'permute', 'reshape', 'view', 'squeeze', 'unsqueeze',
                          'cat', 'stack', 'concat', 'mean', 'sum', 'max', 'min']:
                self.operations[f'torch_{op_name}'] += 1
            
            # NumPy operations
            elif op_name in ['dot', 'matmul', 'multiply', 'add', 'subtract', 'divide',
                           'transpose', 'reshape', 'squeeze', 'concatenate', 'stack',
                           'mean', 'sum', 'max', 'min']:
                self.operations[f'numpy_{op_name}'] += 1
                
        # Neural network operations
        if isinstance(node.func, ast.Name):
            if node.func.id in ['Conv1d', 'Conv2d', 'Conv3d', 'Linear', 'LSTM', 'GRU',
                              'RNN', 'Transformer', 'MultiheadAttention']:
                self.operations[f'nn_{node.func.id}'] += 1
                
        self.generic_visit(node)

def analyze_python_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
            analyzer = TensorOperationAnalyzer()
            analyzer.visit(tree)
            return analyzer.operations
    except:
        return defaultdict(int)

def analyze_notebook(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            operations = defaultdict(int)
            
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    try:
                        tree = ast.parse(cell.source)
                        analyzer = TensorOperationAnalyzer()
                        analyzer.visit(tree)
                        for op, count in analyzer.operations.items():
                            operations[op] += count
                    except:
                        continue
            return operations
    except:
        return defaultdict(int)

def analyze_project(root_dir):
    total_operations = defaultdict(int)
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py') or file.endswith('.ipynb'):
                file_path = os.path.join(root, file)
                print(f"Analyzing: {file_path}")
                
                if file.endswith('.py'):
                    ops = analyze_python_file(file_path)
                else:
                    ops = analyze_notebook(file_path)
                
                for op, count in ops.items():
                    total_operations[op] += count
    
    return total_operations

def print_results(operations):
    print("\nTensor Operations Analysis Results:")
    print("-" * 50)
    
    categories = {
        'Matrix Operations': ['matmul', 'mm', 'bmm', 'dot'],
        'Element-wise Operations': ['mul', 'multiply', 'add', 'subtract', 'sub', 'divide', 'div'],
        'Shape Operations': ['transpose', 'permute', 'reshape', 'view', 'squeeze', 'unsqueeze'],
        'Combining Operations': ['cat', 'stack', 'concat', 'concatenate'],
        'Reduction Operations': ['mean', 'sum', 'max', 'min'],
        'Neural Network Layers': ['Conv1d', 'Conv2d', 'Conv3d', 'Linear', 'LSTM', 'GRU', 'RNN', 'Transformer', 'MultiheadAttention']
    }
    
    for category, ops in categories.items():
        print(f"\n{category}:")
        category_total = 0
        for op_type in ops:
            torch_count = operations.get(f'torch_{op_type}', 0)
            numpy_count = operations.get(f'numpy_{op_type}', 0)
            nn_count = operations.get(f'nn_{op_type}', 0)
            total = torch_count + numpy_count + nn_count
            if total > 0:
                print(f"  {op_type}: {total} (PyTorch: {torch_count}, NumPy: {numpy_count}, NN: {nn_count})")
                category_total += total
        print(f"  Total {category}: {category_total}")

if __name__ == "__main__":
    project_root = "fcfm_course_nlp"  # Replace with your project root
    operations = analyze_project(project_root)
    print_results(operations)
