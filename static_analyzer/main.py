import nbformat
import os

def read_notebook(name_file):
    notebook = nbformat.read(name_file, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            print(cell.source)
    
def read_python_code(name_file):
    with open(name_file, 'r') as file:
        print(file.read())


if __name__ == "__main__":
    read_python_code("example_code.py")
    read_notebook("example_notebook.ipynb")
