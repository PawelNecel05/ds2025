import nbformat
from nbconvert import MarkdownExporter

notebooks = ["Exercise 4.ipynb"]
markdown_files = ["Exercise 4.md"]

for notebook, markdown_file in zip(notebooks, markdown_files):
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
    exporter = MarkdownExporter()
    body, _ = exporter.from_notebook_node(nb)
    with open(markdown_file, 'w') as f:
        f.write(body)

print("Notebooks converted to Markdown files.")