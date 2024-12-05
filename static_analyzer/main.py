import nbformat

def read_notebook(nombre_archivo):
    notebook = nbformat.read(nombre_archivo, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            print(cell.source)
    
def read_python_code(nombre_archivo):
    with open(nombre_archivo, 'r') as file:
        print(file.read())


if __name__ == "__main__":
    read_notebook("examples/example_tarea.ipynb")
    read_python_code("examples/example_code.py") 
