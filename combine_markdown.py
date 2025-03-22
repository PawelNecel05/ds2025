import os

# List of Markdown files to combine
markdown_files = ["Exercise 1.md", "Exercise 2.md", "Exercise 3.md"]

# Output file
output_file = "Combined_Exercises.md"

with open(output_file, 'w') as outfile:
    for fname in markdown_files:
        with open(fname) as infile:
            outfile.write(infile.read())
            outfile.write("\n\n")  # Add a newline between files

print(f"Combined Markdown file created: {output_file}")