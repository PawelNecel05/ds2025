import nbformat
from nbconvert import MarkdownExporter
import os
from pathlib import Path
import shutil

# Input and output paths
notebook_file = "Exercise 4.ipynb"
markdown_file = "Exercise_4.md"
output_dir = "Exercise_4_images"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read the notebook
with open(notebook_file, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Process cells to save plots
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "outputs" in cell:
        for output in cell["outputs"]:
            if output.get("data", {}).get("image/png"):
                # Save the image
                image_data = output["data"]["image/png"]
                image_filename = f"{output_dir}/plot_{cell['execution_count']}.png"
                with open(image_filename, "wb") as img_file:
                    img_file.write(bytes(image_data, encoding="utf-8"))

                # Replace the PNG link in the Markdown
                cell["source"] += f'\n\n![Plot](./{image_filename})\n'

# Convert the notebook to Markdown
exporter = MarkdownExporter()
body, _ = exporter.from_notebook_node(nb)

# Save the Markdown file
with open(markdown_file, "w", encoding="utf-8") as f:
    f.write(body)

print(f"Markdown file saved as {markdown_file}.")
print(f"Images saved in {output_dir}.")