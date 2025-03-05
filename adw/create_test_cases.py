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
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import anthropic

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
    
    # Extract target and filepath from description using Claude
    extraction_result = extract_target_from_description(args.description)
    filepath = Path(extraction_result["target_file"])
    target_spec = extraction_result["target_name"]
    
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
    model = aider.models.Model("claude-3-7-sonnet-latest")
    
    # Load the prompt template from the specification file
    spec_path = Path("specs/test_writing/generate-test-cases.md")
    with open(spec_path, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    # Replace placeholders in the prompt
    output_path = f"{target_spec.lower().replace('.', '_')}_test_cases.json"
    prompt = prompt.replace("<TARGET_FILE>", str(filepath))
    prompt = prompt.replace("<TARGET_CODE>", code_txt)
    prompt = prompt.replace("<OUTPUT_PATH>", output_path)
    
    # Set up the coder with the model and dependents as read-only context
    coder = aider.coders.Coder(
        main_model=model,
        fnames=[],
        read_only_fnames=dependents,
        suggest_shell_commands=False,
        autocommit=False
    )
    
    # Generate the unit tests
    generate_unit_tests(coder, filepath, target_spec, code_txt, prompt)


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
            max_tokens=20000,
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


def generate_unit_tests(coder: Coder, filepath: Path, target_spec: str, code_txt: str, prompt: str):
    """
    Generate unit tests for the specified code target.
    
    Args:
        coder: The Aider coder instance.
        filepath: Path to the file containing the code.
        target_spec: Target specification (class.method or function).
        code_txt: The code text to generate tests for.
        prompt: The prompt to send to the AI model.
    """
    # Create the test file path
    test_filepath = create_test_filepath(filepath)
    
    # Use the coder to generate the tests
    print(f"Generating test cases for {target_spec} in {filepath}")
    print(f"Test file will be created at: {test_filepath}")
    
    # Generate test cases using the AI model
    response = coder.main_model.complete(prompt)
    
    # Extract JSON from the response
    try:
        # Look for JSON content in the response
        if "```json" in response:
            json_text = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_text = response.split("```")[1].split("```")[0].strip()
        else:
            json_text = response.strip()
            
        # Parse the JSON
        test_cases = json.loads(json_text)
        
        # Save the test cases to a file
        output_path = f"{target_spec.lower().replace('.', '_')}_test_cases.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=2)
            
        print(f"Test cases saved to {output_path}")
        
        # TODO: Convert the test cases to actual pytest code
        # This would involve generating a pytest file from the test cases
        
    except Exception as e:
        print(f"Error processing AI response: {e}")
        print("Saving raw response for debugging")
        with open(f"{target_spec.lower().replace('.', '_')}_raw_response.txt", 'w', encoding='utf-8') as f:
            f.write(response)


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
