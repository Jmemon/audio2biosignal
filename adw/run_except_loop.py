#!/usr/bin/env python3
"""
Run-Except Loop

This script runs a Python script in a loop, catches exceptions, and uses an LLM to fix the code
until no exceptions are thrown. It's designed for minor errors like PyTorch tensor dimension
mismatches, not for fundamental refactors of a repository.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import git
import anthropic
from aider.coders import Coder
from aider.io import InputOutput

def create_branch(script_name: Union[str, Path]) -> str:
    """
    Create a new git branch named "automated-exceptions-fix__<script_name_camel_case>" and switch to it.
    
    Args:
        script_name: Name of the script being fixed
        
    Returns:
        str: Name of the created branch
    """
    try:
        # Convert script name to camel case for branch name
        script_name = Path(script_name).stem
        script_name_camel = script_name.replace('_', '-')
        branch_name = f"automated-exceptions-fix__{script_name_camel}"
        
        # Initialize git repo object for current directory
        repo = git.Repo(os.getcwd())
        
        # Check if branch already exists
        if branch_name in [b.name for b in repo.branches]:
            print(f"Branch {branch_name} already exists, switching to it")
            repo.git.checkout(branch_name)
        else:
            print(f"Creating and switching to new branch: {branch_name}")
            repo.git.checkout('HEAD', b=branch_name)
            
        return branch_name
    except git.GitCommandError as e:
        print(f"Git error: {e}")
        sys.exit(1)

def run_script(script_path: Path, script_args: List[str] = None) -> Tuple[str, bool]:
    """
    Run the given Python script with optional arguments and capture stdout/stderr.
    
    Args:
        script_path: Path to the Python script to run
        script_args: Optional list of arguments to pass to the script
        
    Returns:
        Tuple[str, bool]: (output or error message, error flag)
    """
    cmd = [sys.executable, str(script_path)]
    if script_args:
        cmd.extend(script_args)
        
    try:
        result = subprocess.run(
            cmd, 
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout, False  # No error
    except subprocess.CalledProcessError as e:
        # Return error message + stacktrace
        error_output = f"Script failed with exit code {e.returncode}\n"
        if e.stdout:
            error_output += f"STDOUT:\n{e.stdout}\n"
        if e.stderr:
            error_output += f"STDERR:\n{e.stderr}\n"
        return error_output, True  # Error occurred

def get_editable_files(stacktrace: str, repo_path: Path) -> List[Path]:
    """
    Parse the stacktrace to get any paths that reference files in the repo.
    
    Args:
        stacktrace: The error stacktrace
        repo_path: Path to the repository
        
    Returns:
        List[Path]: List of editable file paths (deduplicated)
    """
    # Convert repo_path to absolute path
    repo_path = repo_path.resolve()
    
    # Regular expression to find file paths in stacktrace
    # Looks for patterns like "File "/path/to/file.py", line 123"
    file_pattern = r'File "([^"]+)", line \d+'
    file_matches = re.findall(file_pattern, stacktrace)
    
    editable_files = []
    for file_path in file_matches:
        path = Path(file_path).resolve()
        # Check if the file is within the repo
        if str(path).startswith(str(repo_path)) and path.exists() and path.is_file():
            editable_files.append(path)
    
    # Deduplicate the list while preserving order
    return list(dict.fromkeys(editable_files))

def get_readonly_files(stacktrace: str, repo_path: Path) -> List[Path]:
    """
    Get the file that contains the lowest-level function in stacktrace above the script,
    and all files that depend on that function.
    
    Args:
        stacktrace: The error stacktrace
        repo_path: Path to the repository
        
    Returns:
        List[Path]: List of read-only file paths (deduplicated)
    """
    # Convert repo_path to absolute path
    repo_path = repo_path.resolve()
    
    # Regular expression to find file paths in stacktrace
    file_pattern = r'File "([^"]+)", line \d+'
    file_matches = re.findall(file_pattern, stacktrace)
    
    # Find the last repo-contained path in the stacktrace
    problem_root = None
    for file_path in reversed(file_matches):
        path = Path(file_path).resolve()
        if str(path).startswith(str(repo_path)) and path.exists() and path.is_file():
            problem_root = path
            break
    
    if not problem_root:
        return []
    
    # Find all Python files in the repo
    all_py_files = list(repo_path.glob('**/*.py'))
    
    # Find files that import from problem_root
    dependent_files = []
    problem_module = problem_root.stem
    
    for py_file in all_py_files:
        if py_file == problem_root:
            continue
            
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                
            # Check for imports of the problem module
            # This is a simple check and might miss some complex import patterns
            if re.search(rf'(from|import)\s+.*{problem_module}', content):
                dependent_files.append(py_file)
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    
    # Add the problem_root to the list
    if problem_root not in dependent_files:
        dependent_files.append(problem_root)
    
    # Deduplicate the list while preserving order
    return list(dict.fromkeys(dependent_files))

def main():
    """
    Main function to run the script in a loop, catch exceptions, and fix the code.
    """
    # Ensure we're in the root of a git repository
    try:
        repo = git.Repo(os.getcwd())
    except git.InvalidGitRepositoryError:
        print("Error: Current directory is not a git repository. Please run from the root of your project.")
        sys.exit(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a script in a loop and fix exceptions")
    parser.add_argument("script_path", help="Path to the Python script to run")
    parser.add_argument("src_dir", help="The path to the source directory we are fixing")
    parser.add_argument("--max-iterations", type=int, default=10, 
                        help="Maximum number of iterations to attempt (default: 10)")
    parser.add_argument("--script-args", nargs=argparse.REMAINDER, 
                        help="Arguments to pass to the script being run")
    args = parser.parse_args()
    
    # Convert paths to Path objects
    script_path = Path(args.script_path).resolve()
    src_dir = Path(args.src_dir).resolve()
    
    # Check that script_path exists
    if not script_path.exists() or not script_path.is_file():
        print(f"Error: Script {script_path} does not exist or is not a file.")
        sys.exit(1)
    
    # Check that src_dir exists
    if not src_dir.exists() or not src_dir.is_dir():
        print(f"Error: Source directory {src_dir} does not exist or is not a directory.")
        sys.exit(1)
    
    # Check that script_path is not in src_dir
    if str(script_path).startswith(str(src_dir)):
        print(f"Error: Script {script_path} is inside the source directory {src_dir}.")
        print("The script should be outside the source directory to avoid modifying it.")
        sys.exit(1)
    
    # Store the current branch before switching
    original_branch = repo.active_branch.name
    branch_name = None
    
    try:
        # Create a new branch for the fixes
        branch_name = create_branch(script_path.name)
        
        # Get the initial commit hash for later comparison
        initial_commit = repo.head.commit
        
        # Run the script in a loop until no exceptions are thrown
        loop_flag = True
        iteration = 1
        
        while loop_flag and iteration <= args.max_iterations:
            print(f"\n=== Iteration {iteration} ===")
            print(f"Running script: {script_path}")
            if args.script_args:
                print(f"With arguments: {' '.join(args.script_args)}")
            
            result, loop_flag = run_script(script_path, args.script_args)
            
            if not loop_flag:
                print("Script ran successfully! No exceptions thrown.")
                break
            
            print("Exception caught:")
            print(result)
            print("Analyzing stacktrace...")
            
            # Get editable and read-only files from the stacktrace
            editable_files = get_editable_files(result, src_dir)
            readonly_files = get_readonly_files(result, src_dir)
            
            # Make sure the lists are disjoint, with editable_files taking precedence
            readonly_files = [f for f in readonly_files if f not in editable_files]
            
            print(f"Editable files: {[str(f) for f in editable_files]}")
            print(f"Read-only files: {[str(f) for f in readonly_files]}")
            
            if not editable_files:
                print("No editable files found in the stacktrace. Cannot proceed.")
                break
            
            # Convert Path objects to strings for aider
            editable_files_str = [str(f) for f in editable_files]
            readonly_files_str = [str(f) for f in readonly_files]
            
            # Create aider Coder instance using the create method
            coder = Coder.create(
                model="claude-3-7-sonnet-latest",
                fnames=editable_files_str,
                read_only_fnames=readonly_files_str,
                auto_commit=False,
                suggest_shell_commands=False,
                yes=True
            )
            
            # Create prompt for the LLM
            prompt = f"""
    The following exception occurred when running '{script_path}':

    {result}

    Please fix the code so it runs without exceptions. Focus only on addressing the specific error shown in the stacktrace.

    Important guidelines:
    1. Do not change the fundamental logic or architecture of the code
    2. Make minimal, targeted changes to fix the specific exception
    3. Focus on issues like incorrect tensor dimensions, type mismatches, or parameter errors
    4. Explain your reasoning for each change

    Only modify the files I've provided as editable. The read-only files are for context only.
    """
            
            # Run aider with the prompt
            coder.run(prompt)
            
            iteration += 1
    
        if iteration > args.max_iterations:
            print(f"Reached maximum number of iterations ({args.max_iterations}). Exiting loop.")
        
        # Generate diff from the start commit of the branch to the current working state
        diff = repo.git.diff(initial_commit.hexsha, '--', src_dir)
        
        if not diff:
            print("No changes were made to fix exceptions.")
            return
        
        print("\n=== Changes made to fix exceptions ===")
        print(diff)
        
        # Use Anthropic API directly to summarize the changes
        client = anthropic.Anthropic()
        
        summary_prompt = f"""
    Please provide a concise summary of the following git diff that fixed exceptions in the code:

    {diff}

    Focus on:
    1. What was the root cause of the exception(s)
    2. What changes were made to fix it
    3. Any potential side effects of these changes

    Keep the summary technical but easy to understand.
    """
        
        try:
            response = client.messages.create(
                model="claude-3-7-sonnet-latest",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": summary_prompt}
                ]
            )
            summary = response.content[0].text
            print("\n=== Summary of changes ===")
            print(summary)
        except Exception as e:
            print(f"Error generating summary: {e}")
            summary = "Failed to generate summary."
        
            # Ask user if they want to merge changes
            merge_response = input(f"\nWould you like to merge these changes into the '{original_branch}' branch? (y/n): ").strip().lower()
            if merge_response in ('y', 'yes'):
                try:
                    # Switch to original branch and merge
                    repo.git.checkout(original_branch)
                    repo.git.merge(branch_name)
                    print(f"Successfully merged changes from {branch_name} into {original_branch}.")
                except git.GitCommandError as e:
                    print(f"Error merging changes: {e}")
                    print("Please resolve conflicts manually and merge the branch.")
            else:
                print(f"Changes remain in branch '{branch_name}'. You can merge them later if needed.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        # Clean up: switch back to original branch and delete the new branch if it was created
        if branch_name:
            try:
                print(f"Switching back to original branch '{original_branch}'")
                repo.git.checkout(original_branch)
                
                print(f"Deleting branch '{branch_name}'")
                repo.git.branch('-D', branch_name)
                print(f"Branch '{branch_name}' deleted successfully")
            except git.GitCommandError as git_err:
                print(f"Error during cleanup: {git_err}")
        sys.exit(1)

if __name__ == "__main__":
    main()
