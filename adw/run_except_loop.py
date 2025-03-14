#!/usr/bin/env python3
"""
Run-Except Loop

This script executes a command and catches exceptions. When an exception occurs,
it uses an LLM to fix the exception by modifying the relevant files.
The script continues this loop until the command executes successfully.

Usage:
    python run_except_loop.py <execution_command>

Example:
    python run_except_loop.py "python scripts/train_audio2eda.py"
"""

import argparse
import os
import subprocess
import sys
import traceback
import re
import git
from datetime import datetime
import openai
import tempfile
from pathlib import Path

# Configure OpenAI client
client = openai.OpenAI()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a command and fix exceptions using an LLM.")
    parser.add_argument("command", help="The command to execute")
    return parser.parse_args()

def execute_command(command):
    """
    Execute a command and return the result.
    
    Args:
        command (str): The command to execute
        
    Returns:
        tuple: (success, output, error)
    """
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return True, result.stdout, None
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def get_exception_info(error_output):
    """
    Extract exception type and traceback from error output.
    
    Args:
        error_output (str): The stderr output
        
    Returns:
        tuple: (exception_type, traceback)
    """
    # Extract the exception type and message
    exception_match = re.search(r'([A-Za-z0-9_.]+Error|Exception): (.+?)(\n|$)', error_output)
    exception_type = exception_match.group(1) if exception_match else "Unknown Error"
    exception_message = exception_match.group(2) if exception_match else "Unknown error message"
    
    # Return the full error output as the traceback
    return f"{exception_type}: {exception_message}", error_output

def get_relevant_files(traceback_text, repo_path="."):
    """
    Extract relevant files from the traceback.
    
    Args:
        traceback_text (str): The traceback text
        repo_path (str): Path to the repository
        
    Returns:
        list: List of relevant file paths
    """
    # Extract file paths from traceback
    file_matches = re.findall(r'File "([^"]+)", line \d+', traceback_text)
    
    # Filter to only include files in the repository
    repo_files = []
    repo = git.Repo(repo_path)
    repo_root = repo.git.rev_parse("--show-toplevel")
    
    for file_path in file_matches:
        # Convert to absolute path if it's relative
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        
        # Check if the file is in the repository
        if file_path.startswith(repo_root) and os.path.exists(file_path):
            # Convert to relative path
            rel_path = os.path.relpath(file_path, repo_root)
            repo_files.append(rel_path)
    
    return repo_files

def get_file_content(file_path):
    """
    Get the content of a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Content of the file
    """
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def find_dependent_files(file_path, repo_path="."):
    """
    Find files that import or use the given file.
    
    Args:
        file_path (str): Path to the file
        repo_path (str): Path to the repository
        
    Returns:
        list: List of dependent file paths
    """
    # Get the module name from the file path
    module_path = os.path.splitext(file_path)[0].replace("/", ".")
    file_name = os.path.basename(file_path)
    module_name = os.path.splitext(file_name)[0]
    
    # Search for imports of this module in the repository
    dependent_files = []
    repo = git.Repo(repo_path)
    repo_root = repo.git.rev_parse("--show-toplevel")
    
    # Use git grep to find files that import the module
    try:
        # Search for different import patterns
        patterns = [
            f"import {module_name}",
            f"from {module_name} import",
            f"import {module_path}",
            f"from {module_path} import"
        ]
        
        for pattern in patterns:
            try:
                grep_result = repo.git.grep("-l", pattern)
                if grep_result:
                    dependent_files.extend(grep_result.splitlines())
            except git.exc.GitCommandError:
                # No matches found for this pattern
                pass
    except git.exc.GitCommandError:
        # No matches found
        pass
    
    return list(set(dependent_files))

def apply_llm_fix(exception_info, traceback_text, relevant_files):
    """
    Use an LLM to generate fixes for the exception.
    
    Args:
        exception_info (str): Exception type and message
        traceback_text (str): Full traceback
        relevant_files (list): List of relevant file paths and their content
        
    Returns:
        dict: Dictionary mapping file paths to their updated content
    """
    # Prepare the prompt for the LLM
    file_contents = []
    for file_path in relevant_files:
        content = get_file_content(file_path)
        file_contents.append(f"File: {file_path}\n```\n{content}\n```")
    
    files_text = "\n\n".join(file_contents)
    
    prompt = f"""
You are an expert Python developer tasked with fixing an exception in a codebase.

EXCEPTION:
{exception_info}

TRACEBACK:
{traceback_text}

RELEVANT FILES:
{files_text}

Your task is to fix the exception by modifying the relevant files. Follow these guidelines:
1. Change as little as possible to fix the exception.
2. Don't change the fundamental logic of the code.
3. Focus only on fixing the immediate exception.
4. Provide your solution as a series of file edits in the format:

FILE: <file_path>
```
<entire new content of the file>
```

Explain your changes briefly before providing the file edits.
"""

    # Call the LLM API
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an expert Python developer who fixes exceptions with minimal changes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=4000
    )
    
    # Parse the response to extract file edits
    response_text = response.choices[0].message.content
    file_edits = {}
    
    # Extract file edits using regex
    file_edit_pattern = r'FILE: ([^\n]+)\n```(?:\w+)?\n(.*?)\n```'
    matches = re.finditer(file_edit_pattern, response_text, re.DOTALL)
    
    for match in matches:
        file_path = match.group(1).strip()
        new_content = match.group(2)
        file_edits[file_path] = new_content
    
    return file_edits

def apply_fixes(file_edits):
    """
    Apply the fixes to the files.
    
    Args:
        file_edits (dict): Dictionary mapping file paths to their updated content
        
    Returns:
        bool: True if changes were made, False otherwise
    """
    changes_made = False
    
    for file_path, new_content in file_edits.items():
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the new content to the file
        with open(file_path, 'w') as f:
            f.write(new_content)
            changes_made = True
    
    return changes_made

def create_branch(script_name):
    """
    Create a new git branch for the fixes.
    
    Args:
        script_name (str): Name of the script being executed
        
    Returns:
        str: Name of the created branch
    """
    # Convert script name to camel case
    script_base = os.path.basename(script_name)
    script_camel = ''.join(x.capitalize() for x in script_base.split('_'))
    
    # Create branch name
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    branch_name = f"gen-diff-fix-exceptions-{script_camel}-{timestamp}"
    
    # Create the branch
    repo = git.Repo(".")
    repo.git.checkout("-b", branch_name)
    
    return branch_name

def commit_changes(message):
    """
    Commit changes to the git repository.
    
    Args:
        message (str): Commit message
    """
    repo = git.Repo(".")
    repo.git.add(".")
    repo.git.commit("-m", message)

def generate_diff_summary(branch_name):
    """
    Generate a summary of the changes made in the branch.
    
    Args:
        branch_name (str): Name of the branch
        
    Returns:
        str: Summary of the changes
    """
    repo = git.Repo(".")
    diff = repo.git.diff("main", branch_name)
    
    # Use LLM to summarize the diff
    prompt = f"""
Please summarize the following git diff that contains fixes for exceptions:

```
{diff}
```

Provide a concise summary of:
1. What exceptions were fixed
2. What changes were made to fix them
3. Any potential side effects of these changes
"""

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at summarizing code changes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def main():
    """Main function to run the except loop."""
    args = parse_args()
    command = args.command
    
    # Extract script name from command for branch naming
    script_name = command.split()[-1] if len(command.split()) > 0 else "unknown"
    
    # Create a new branch
    branch_name = create_branch(script_name)
    print(f"Created branch: {branch_name}")
    
    iteration = 0
    while True:
        iteration += 1
        print(f"\n=== Iteration {iteration} ===")
        print(f"Executing: {command}")
        
        # Execute the command
        success, output, error = execute_command(command)
        
        if success:
            print("Command executed successfully!")
            break
        
        print("Exception occurred. Analyzing...")
        
        # Get exception information
        exception_info, traceback_text = get_exception_info(error)
        print(f"Exception: {exception_info}")
        
        # Get relevant files
        relevant_files = get_relevant_files(traceback_text)
        print(f"Relevant files: {relevant_files}")
        
        # Find dependent files
        all_files = set(relevant_files)
        for file_path in relevant_files:
            dependent_files = find_dependent_files(file_path)
            all_files.update(dependent_files)
        
        all_files = list(all_files)
        print(f"All relevant files (including dependencies): {all_files}")
        
        # Use LLM to generate fixes
        print("Generating fixes using LLM...")
        file_edits = apply_llm_fix(exception_info, traceback_text, all_files)
        
        # Apply the fixes
        changes_made = apply_fixes(file_edits)
        
        if changes_made:
            # Commit the changes
            commit_message = f"Fix exception: {exception_info} (iteration {iteration})"
            commit_changes(commit_message)
            print(f"Committed changes: {commit_message}")
        else:
            print("No changes were made. Exiting loop.")
            break
    
    # Generate summary of changes
    if iteration > 1:  # Only if fixes were applied
        print("\n=== Summary of Changes ===")
        summary = generate_diff_summary(branch_name)
        print(summary)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
