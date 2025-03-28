#!/usr/bin/env python3
"""
Script to create unit tests for specified code targets using AI assistance.
"""

import argparse
import inspect
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
import anthropic

try:
    import aider
    from aider.coders import Coder
    from aider.models import Model
    from aider.io import InputOutput
except ImportError:
    print("Error: aider package not found. Please install it with 'pip install aider'")
    sys.exit(1)


def main():
    """
    Main function to create unit tests for specified code targets.
    """
    # Load environment variables
    load_dotenv()
    
    # Verify we're in the project root
    if not verify_project_root():
        print("Error: Must run from project root directory (audio2biosignal)")
        sys.exit(1)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Extract target and filepath from description using Claude
    extraction_result = extract_target_from_description(args.description)
    filepath_str = extraction_result["target_file"]
    target_spec = extraction_result["target_name"]
    
    # Normalize the filepath (handle both relative and absolute paths)
    filepath = normalize_filepath(filepath_str)
    
    # Verify that the filepath exists
    if not filepath.exists():
        print(f"Error: File {filepath} does not exist")
        sys.exit(1)
    
    # Extract the code text from the target
    code_txt = extract_code_text(filepath, target_spec)
    if not code_txt:
        print(f"Error: Could not extract code for '{target_spec}' from {filepath}")
        sys.exit(1)
    
    # Find dependent files
    dependents = find_dependent_files(filepath, target_spec)
    
    # Set up the AI model
    model = Model(args.model)
    print(f"\nInitialized model: {model}")
    print(f"\nTarget file: {filepath}")
    print(f"Target specification: {target_spec}")
    print(f"\nExtracted code:\n{code_txt}\n")
    
    # Load the prompt template from the specification file
    spec_path = Path("specs/test_writing/generate-tests.md")
    with open(spec_path, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    # Create the test file
    test_file_path = create_test_file(filepath)
    
    # Read the existing test file content
    test_code_txt = ""
    if test_file_path.exists() and test_file_path.stat().st_size > 0:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_code_txt = f.read()
    
    # Helper function to read dependent files
    def read(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return ""
    
    # Replace placeholders in the prompt
    prompt = prompt.replace("<TARGET_CODE>", code_txt)
    prompt = prompt.replace("<DEPENDENT_CODE>", ",".join(str(read(d)) for d in dependents))
    prompt = prompt.replace("<EXISTING_TESTS>", test_code_txt)
    prompt = prompt.replace("<TEST_FILE_PATH>", str(test_file_path))
    
    # Store the base prompt for iterations
    base_prompt = prompt
    
    # Set up the coder with the model and dependents as read-only context
    coder = Coder.create(
        main_model=model,
        fnames=[test_file_path],
        read_only_fnames=dependents + [filepath],
        suggest_shell_commands=False,
        auto_commits=False,
        io=InputOutput(yes=True)
    )
    
    # Generate the unit tests with multiple iterations
    for iteration in range(2):
        print(f"\nIteration {iteration + 1}/2 for generating test cases...")
        
        # Generate the unit tests
        generate_test_cases(coder, filepath, target_spec, code_txt, prompt)
        
        # Read the updated test file content for the next iteration
        if test_file_path.exists() and test_file_path.stat().st_size > 0:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                current_test_code = f.read()
                
            # Update the prompt for the next iteration
            prompt = current_test_code + "\n\nGIVEN WE ARE WRITING TESTS ONLY FOR " + target_spec + "\n\nCAN YOU THINK OF ANY MORE TESTS NEEDED? ONLY ADD MORE IF THEY ARE CONSTRUCTIVE ADDITIONS, SIMPLE COMPONENTS MIGHT NOT NEED ANYTHING ELSE\n\n" + base_prompt


def verify_project_root() -> bool:
    """
    Verify that we are in the project root directory.
    
    Returns:
        bool: True if we are in the project root, False otherwise.
    """
    cwd = Path.cwd().name
    return cwd == "audio2biosignal"


def normalize_filepath(filepath_str: str) -> Path:
    """
    Normalize a filepath string to an absolute Path.
    
    Args:
        filepath_str: A string representing a file path, which can be:
                     - An absolute path
                     - A path relative to the project root (audio2biosignal)
    
    Returns:
        Path: An absolute Path object
    """
    filepath = Path(filepath_str)
    
    # If it's already an absolute path, return it
    if filepath.is_absolute():
        return filepath
    
    # Otherwise, treat it as relative to the project root
    return Path.cwd() / filepath


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Create unit tests for specified code targets")
    parser.add_argument(
        "description",
        type=str,
        help="Description of what class/method/function to test and the filepath it's located in"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MIRROR add_architecture",
        help="Model to use for test generation"
    )
    return parser.parse_args()


def extract_target_from_description(description: str) -> Dict[str, str]:
    """
    Use Claude to extract the target function/class/method and file from the description.
    
    Args:
        description: User's description of what to test
        
    Returns:
        Dictionary with keys 'target_name' and 'target_file'
    """
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    prompt = f"""
    Extract the target function/class/method name and the file path from this description:
    
    "{description}"
    
    Return only a JSON object with these keys:
    - target_name: The name of the function, class, or method (e.g., "ClassName.method_name" or "function_name")
    - target_file: The file path where the code is located (e.g., "src/module/file.py")
    
    If you can't determine one of these values, use null for that field.
    """
    
    message = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1000,
        temperature=0,
        system="You are a helpful assistant that extracts structured information from text.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract the JSON from the response
    try:
        response_text = message.content[0].text
        # Find JSON in the response (it might be wrapped in markdown code blocks)
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_text = response_text.strip()
            
        result = json.loads(json_text)
        
        # Validate the result
        if "target_name" not in result or "target_file" not in result:
            print("Error: Claude's response is missing required fields")
            sys.exit(1)
            
        return result
    except Exception as e:
        print(f"Error parsing Claude's response: {e}")
        print("Using fallback parsing method...")
        # Implement a simple fallback method
        return {
            "target_name": input("Please enter the target name (e.g., ClassName.method_name): "),
            "target_file": input("Please enter the file path (e.g., src/module/file.py): ")
        }


def parse_target_spec(target_spec: str) -> Tuple[Path, str]:
    """
    Parse the target specification into filepath and target.
    
    Args:
        target_spec: String in format <filepath>:class.method or <filepath>:function
        
    Returns:
        Tuple containing the filepath and the target specification.
    """
    if ":" not in target_spec:
        print("Error: Target specification must be in format <filepath>:class.method or <filepath>:function")
        sys.exit(1)
        
    filepath_str, target = target_spec.split(":", 1)
    filepath = Path(filepath_str)
    
    if not filepath.exists():
        print(f"Error: File {filepath} does not exist")
        sys.exit(1)
        
    return filepath, target


def extract_code_text(filepath: Path, target_spec: str) -> Optional[str]:
    """
    Extract the code text for the specified target using Claude.
    
    Args:
        filepath: Path to the file containing the code.
        target_spec: Target specification (class.method or function).
        
    Returns:
        The extracted code text, or None if extraction failed.
    """
    try:
        # Read the file content
        with open(filepath, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        # Create Anthropic client
        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        
        # Create the prompt for Claude
        prompt = f"""
        I have a Python file with the following content:
        
        ```python
        {file_content}
        ```
        
        Please extract the complete source code for the target: "{target_spec}"
        
        The target could be:
        1. A function (e.g., "function_name")
        2. A class (e.g., "ClassName")
        3. A class method (e.g., "ClassName.method_name")
        
        Return ONLY the exact source code of the target, including docstrings, comments, and all code within the target's scope.
        Include any decorators that are applied to the target.
        For classes, include all methods and attributes.
        """
        
        # Call Claude API
        message = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=8192,
            temperature=0,
            system="You are a helpful assistant that extracts code from Python files with perfect accuracy.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the code from the response
        response_text = message.content[0].text
        
        # Find code in the response (it might be wrapped in markdown code blocks)
        if "```python" in response_text:
            code_text = response_text.split("```python")[1].split("```")[0].strip()
        elif "```" in response_text:
            code_text = response_text.split("```")[1].split("```")[0].strip()
        else:
            code_text = response_text.strip()
        
        if not code_text:
            print(f"Error: Could not extract code for '{target_spec}' from {filepath}")
            return None
            
        return code_text
        
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return None
    except Exception as e:
        print(f"Error extracting code: {e}")
        return None


def create_test_file(filepath: Path) -> Path:
    """
    Create an empty test file in tests/unit directory that mirrors the structure of the source file.
    
    Args:
        filepath: Path to the source file.
        
    Returns:
        Path: Path to the created test file.
    """
    # Get the relative path from the src directory
    try:
        rel_path = filepath.relative_to(Path.cwd() / 'src')
    except ValueError:
        # If the file is not in src, use the filename only
        rel_path = Path(filepath.name)
    
    # Create the test file path in tests/unit
    test_dir = Path.cwd() / 'tests' / 'unit' / rel_path.parent
    test_file = test_dir / f"test_{filepath.name}"
    
    # Create the directory if it doesn't exist
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the test file if it doesn't exist
    if not test_file.exists():
        with open(test_file, 'w', encoding='utf-8') as f:
            pass  # Create an empty file
    
    return test_file


def find_dependent_files(filepath: Path, target_spec: str) -> List[Path]:
    """
    Find files that import or use the specified target.
    
    Args:
        filepath: Path to the file containing the target.
        target_spec: Target specification (class.method or function).
        
    Returns:
        List of paths to dependent files.
    """
    # Get the module name from the filepath
    rel_path = filepath.relative_to(Path.cwd())
    module_path = str(rel_path).replace('/', '.').replace('\\', '.').replace('.py', '')
    
    # Get the target name (last part of the target_spec)
    target_name = target_spec.split('.')[-1]
    
    # Find all Python files in src and scripts directories
    src_files = list(Path.cwd().joinpath('src').glob('**/*.py'))
    script_files = list(Path.cwd().joinpath('scripts').glob('**/*.py'))
    python_files = src_files + script_files
    
    # Exclude the target file itself
    python_files = [f for f in python_files if f != filepath]
    
    dependent_files = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Check for imports of the module
                module_import_patterns = [
                    f"import {module_path}",
                    f"from {module_path} import",
                ]
                
                # Check for direct usage of the target
                target_usage_patterns = [
                    f"{target_name}(",  # Function call
                    f"{target_spec}(",  # Method call with class prefix
                    f" {target_name} ",  # Variable usage
                    f":{target_name}",  # Type hint
                ]
                
                # If any pattern is found, add the file to dependents
                if any(pattern in content for pattern in module_import_patterns + target_usage_patterns):
                    dependent_files.append(file_path)
        except Exception as e:
            print(f"Warning: Error scanning {file_path}: {e}")
            continue
    
    return dependent_files


def generate_test_cases(coder: Coder, filepath: Path, target_spec: str, code_txt: str, prompt: str):
    """
    Generate unit tests for the specified code target.
    
    Args:
        coder: The Aider coder instance.
        filepath: Path to the file containing the code.
        target_spec: Target specification (class.method or function).
        code_txt: The code text to generate tests for.
        prompt: The prompt to send to the AI model.
    """
    # Use the coder to generate the tests
    print(f"Generating test cases for {target_spec} in {filepath}")
    
    # Generate test cases using the AI model
    coder.run(prompt)




if __name__ == "__main__":
    main()
