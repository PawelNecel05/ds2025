import os

# List of Markdown files to combine
markdown_files = ["E5.md", "E6.md"]

# Output file
output_file = "Exercises5and6.md"

with open(output_file, 'w') as outfile:
    for fname in markdown_files:
        with open(fname) as infile:
            outfile.write(infile.read())
            outfile.write("\n\n")  # Add a newline between files

print(f"Combined Markdown file created: {output_file}")