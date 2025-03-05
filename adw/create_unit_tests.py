#!/usr/bin/env python3
"""
Script to create unit tests for specified code targets using AI assistance.
"""

import argparse
import inspect
import importlib
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from dotenv import load_dotenv

try:
    import aider
    from aider.coders import Coder
    from aider.models import Model
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
    
    # Parse the target specification
    filepath, target_spec = parse_target_spec(args.target)
    
    # Extract the code text from the target
    code_txt = extract_code_text(filepath, target_spec)
    if not code_txt:
        print(f"Error: Could not extract code from {args.target}")
        sys.exit(1)
    
    # Find dependent files
    dependents = find_dependent_files(filepath)
    
    # Set up the AI model
    model = aider.models.Model("claude-3-7-sonnet-latest")
    
    # Create an empty prompt
    prompt = ""
    
    # Set up the coder with the model and dependents as read-only context
    coder = aider.coders.Coder(
        main_model=model,
        fnames=[],
        read_only_fnames=dependents,
        suggest_shell_commands=False,
        autocommit=False
    )
    
    # Generate the unit tests
    generate_unit_tests(coder, filepath, target_spec, code_txt)


def verify_project_root() -> bool:
    """
    Verify that we are in the project root directory.
    
    Returns:
        bool: True if we are in the project root, False otherwise.
    """
    cwd = Path.cwd().name
    return cwd == "audio2biosignal"


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
    return parser.parse_args()


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
    Extract the code text for the specified target using the inspect module.
    
    Args:
        filepath: Path to the file containing the code.
        target_spec: Target specification (class.method or function).
        
    Returns:
        The extracted code text, or None if extraction failed.
    """
    try:
        # Convert file path to module path
        rel_path = filepath.relative_to(Path.cwd())
        module_path = str(rel_path).replace('/', '.').replace('\\', '.').replace('.py', '')
        
        # Import the module
        spec = importlib.util.spec_from_file_location(module_path, filepath)
        if not spec or not spec.loader:
            print(f"Error: Could not load module from {filepath}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Parse the target specification
        parts = target_spec.split('.')
        
        # Handle function case
        if len(parts) == 1:
            if not hasattr(module, parts[0]):
                print(f"Error: Function {parts[0]} not found in {filepath}")
                return None
            target_obj = getattr(module, parts[0])
            return inspect.getsource(target_obj)
        
        # Handle class.method case
        elif len(parts) == 2:
            class_name, method_name = parts
            
            if not hasattr(module, class_name):
                print(f"Error: Class {class_name} not found in {filepath}")
                return None
                
            class_obj = getattr(module, class_name)
            
            if not hasattr(class_obj, method_name):
                print(f"Error: Method {method_name} not found in class {class_name}")
                return None
                
            method_obj = getattr(class_obj, method_name)
            return inspect.getsource(method_obj)
        
        # Handle class case (if target_spec is just a class name)
        elif len(parts) == 1 and inspect.isclass(getattr(module, parts[0], None)):
            class_obj = getattr(module, parts[0])
            return inspect.getsource(class_obj)
            
        else:
            print(f"Error: Invalid target specification format: {target_spec}")
            return None
            
    except ImportError as e:
        print(f"Error importing module: {e}")
        return None
    except Exception as e:
        print(f"Error extracting code: {e}")
        return None


def find_dependent_files(filepath: Path) -> List[Path]:
    """
    Find files that import or use the specified file.
    
    Args:
        filepath: Path to the file to find dependents for.
        
    Returns:
        List of paths to dependent files.
    """
    # This is a placeholder implementation
    # In a real implementation, you would need to scan the codebase
    # to find files that import the target file
    
    # For now, just return an empty list
    return []


def generate_unit_tests(coder: Coder, filepath: Path, target_spec: str, code_txt: str):
    """
    Generate unit tests for the specified code target.
    
    Args:
        coder: The Aider coder instance.
        filepath: Path to the file containing the code.
        target_spec: Target specification (class.method or function).
        code_txt: The code text to generate tests for.
    """
    # Create the test file path
    test_filepath = create_test_filepath(filepath)
    
    # Create the prompt for the AI
    prompt = f"""
    Please create unit tests for the following code:
    
    File: {filepath}
    Target: {target_spec}
    
    Code:
    ```python
    {code_txt}
    ```
    
    The tests should be comprehensive and cover all edge cases.
    Please use pytest for the tests.
    """
    
    # TODO: Use the coder to generate the tests
    # This would involve interacting with the AI model
    print(f"Would generate tests for {target_spec} in {filepath}")
    print(f"Test file would be created at: {test_filepath}")


def create_test_filepath(filepath: Path) -> Path:
    """
    Create the path for the test file based on the source file path.
    
    Args:
        filepath: Path to the source file.
        
    Returns:
        Path to the test file.
    """
    # Convert src/module/file.py to tests/module/test_file.py
    parts = list(filepath.parts)
    
    if "src" in parts:
        src_index = parts.index("src")
        parts[src_index] = "tests"
    else:
        # If not in src, just prepend tests/
        parts = ["tests"] + parts[1:]
    
    # Add test_ prefix to the filename
    filename = parts[-1]
    parts[-1] = f"test_{filename}"
    
    return Path(*parts)


if __name__ == "__main__":
    main()
