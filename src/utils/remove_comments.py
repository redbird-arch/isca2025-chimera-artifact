import os
import re
import sys

def remove_comments_and_trailing_spaces_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    code = ''.join(lines)
    code = re.sub(r'(?s)\'\'\'(.*?)\'\'\'', '', code)
    code = re.sub(r'(?s)\"\"\"(.*?)\"\"\"', '', code)
    
    code = re.sub(r'#.*', '', code)

    lines = code.splitlines()
    lines = [line.rstrip() for line in lines]

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write("\n".join(lines))

def remove_comments_and_trailing_spaces_in_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".py"):  
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                remove_comments_and_trailing_spaces_from_file(file_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python remove_comments.py PATH")
        sys.exit(1)

    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        sys.exit(1)

    remove_comments_and_trailing_spaces_in_directory(directory_path)
