import nbformat


def read_notebook(nombre_archivo):
    notebook = nbformat.read(nombre_archivo, as_version=4)
    for cell in notebook.cells:
        # Cada celda tiene un tipo ('code', 'markdown', etc.)
    if cell.cell_type == 'code':
        # .source contiene el contenido de la celda
        print(cell.source)
    
def read_python_code(nombre_archivo):
    with open(nombre_archivo, 'r') as file:
        return file.read()


if __name__ == "__main__":


