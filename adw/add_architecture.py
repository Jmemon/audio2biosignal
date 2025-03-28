#!/usr/bin/env python3
"""
Script to add a new neural network architecture to the audio2biosignal project.

This script:
1. Takes a markdown file describing a neural network architecture
2. Validates the description with Claude
3. Creates the architecture implementation file
4. Updates the configs.py file to support the new architecture
5. Creates comprehensive tests
6. Generates example configurations
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List
import subprocess
from dotenv import load_dotenv
import anthropic
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

def main() -> None:
    """
    Main function to add a new neural network architecture to the project.
    
    Parses command line arguments, validates the architecture description,
    creates implementation files, updates configs, and generates tests and
    example configurations.
    """
    # Load environment variables
    load_dotenv()
    
    # Verify we're in the project root directory
    ensure_project_root()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Add a new neural network architecture to the audio2biosignal project"
    )
    parser.add_argument(
        "markdown_file",
        type=str,
        help="Path to a markdown file describing the architecture"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the architecture (used for filenames)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-7-sonnet-default",
        help="model to use with aider"
    )
    
    args = parser.parse_args()
    model_choice = args.model
    
    # Normalize the architecture name (lowercase, no spaces)
    arch_name = args.name.lower().replace(" ", "_")
    
    # Validate the markdown file exists
    markdown_path = Path(args.markdown_file)
    if not markdown_path.exists():
        print(f"Error: Markdown file {markdown_path} does not exist")
        sys.exit(1)
    
    # Read the markdown file
    with open(markdown_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Validate the architecture description with Claude
    if not validate_architecture_description(markdown_content):
        print("Error: The markdown file does not contain a thorough description of an architecture and its configuration")
        sys.exit(1)
    
    print(f"✅ Architecture description validated for {arch_name}")
    
    # Create the architecture implementation file
    create_architecture_file(arch_name, markdown_content, markdown_path, model_choice)
    
    # Update the configs.py file
    update_configs_file(arch_name, markdown_content, markdown_path, model_choice)
    
    # Create tests for the new architecture
    create_tests(arch_name, model_choice)
    
    # Create example configurations
    create_example_configs(arch_name, model_choice)
    
    print(f"\n✅ Successfully added {arch_name} architecture to the project!")

def validate_architecture_description(markdown_content: str) -> bool:
    """
    Validate that the markdown file contains a thorough description of an architecture and its configuration.
    
    Args:
        markdown_content: The content of the markdown file
        
    Returns:
        True if the description is valid, False otherwise
    """
    # Get API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment")
        sys.exit(1)
    
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Create prompt for Claude
    prompt = f"""
    I have a markdown file that should describe a neural network architecture and its configuration.
    Please analyze this file and determine if it contains:
    
    1. A thorough description of a neural network architecture
    2. A clear specification of the configuration parameters needed for this architecture
    
    Here is the content of the file:
    
    {markdown_content}
    
    Please respond with ONLY "YES" if the file contains both a thorough architecture description and configuration specification, or "NO" if it does not.
    """
    
    # Call Claude API
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=100,
            temperature=0,
            system="You are a helpful assistant that validates neural network architecture descriptions.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract the response
        result = response.content[0].text.strip().upper()
        
        # Check if the response is YES
        return result == "YES"
    
    except Exception as e:
        print(f"Error validating architecture description: {e}")
        return False

def create_architecture_file(arch_name: str, markdown_content: str, markdown_path: Path, model_choice: str) -> None:
    """
    Create the architecture implementation file using Claude.
    
    Args:
        arch_name: Name of the architecture
        markdown_content: Content of the markdown file
        markdown_path: Path to the markdown file
    """
    # Create the architecture file path
    arch_file_path = Path(f"src/architectures/{arch_name}.py")
    
    # Convert architecture name to CamelCase for class name
    class_name = ''.join(word.capitalize() for word in arch_name.split('_'))
    arch_file_path.write_text("", encoding='utf-8')
    print(f"Created empty architecture file: {arch_file_path}")
    
    # Create the prompt for Claude
    prompt = f"""
    Based on the architecture description in the markdown file, please ONLY ADD the architecture code to the Python implementation file, and NOT the config. The architecture should:

    1. Follow PyTorch best practices
    2. Include comprehensive docstrings
    3. Be consistent with the existing codebase style
    4. Include all necessary imports
    5. Implement the architecture as described in the markdown file
    6. Name the main architecture class "{class_name}" (in CamelCase)
    """
    
    # Initialize aider Coder
    model = Model(model_choice)
    coder = Coder.create(
        main_model=model,
        fnames=[arch_file_path],
        edit_format="diff",
        read_only_fnames=[markdown_path],
        suggest_shell_commands=False,
        auto_commits=False,
        io=InputOutput(yes=True)
    )
    
    # Run the coder with the prompt
    coder.run(prompt)
    print(f"Architecture implemented in: {arch_file_path}")

def update_configs_file(arch_name: str, markdown_content: str, markdown_path: Path, model_choice: str) -> None:
    """
    Update the configs.py file to support the new architecture.
    
    Args:
        arch_name: Name of the architecture
        markdown_content: Content of the markdown file
        markdown_path: Path to the markdown file
    """
    # Create the prompt for Claude
    prompt = f"""
    Please ONLY ADD the config from the markdown for the new {arch_name} architecture to src/architectures/configs.py,
    and also add a type in whichever enum corresponds to this architecture.
    Do not add or modify any other architecture or config in this file.
    """
    
    # Initialize aider Coder
    model = Model(model_choice)
    coder = Coder.create(
        main_model=model,
        edit_format="diff",
        fnames=["src/architectures/configs.py"],
        read_only_fnames=[markdown_path],
        suggest_shell_commands=False,
        auto_commits=False,
        io=InputOutput(yes=True)
    )
    
    print(f"\nUpdating configs.py to support {arch_name} architecture")
    
    # Run the coder with the prompt
    coder.run(prompt)
    
    print(f"✅ Updated configs.py to support {arch_name} architecture")

def create_tests(arch_name: str, model_choice: str) -> None:
    """
    Create comprehensive tests for the new architecture.
    
    Args:
        arch_name: Name of the architecture
    """
    print(f"\nCreating tests for {arch_name} architecture")
    
    # Run the create_tests.py script
    try:
        subprocess.run(
            ["python", "adw/create_tests.py", f"src/architectures/{arch_name}.py:{arch_name}", "--model", model_choice],
            check=True
        )
        print(f"✅ Created tests for {arch_name} architecture")
    except subprocess.CalledProcessError as e:
        print(f"Error creating tests: {e}")

def create_example_configs(arch_name: str, model_choice: str) -> None:
    """
    Create example configurations for the new architecture.
    
    Args:
        arch_name: Name of the architecture
    """
    print("\nCreating example configurations")
    
    # Create a small version of the architecture
    try:
        subprocess.run(
            ["python", "adw/create_config.py", f"A small version of the {arch_name} architecture with minimal parameters", "--name", f"{arch_name}_small", "--model", model_choice],
            check=True
        )
        print(f"✅ Created small {arch_name} configuration")
    except subprocess.CalledProcessError as e:
        print(f"Error creating small configuration: {e}")
    
    # Create a big version of the architecture
    try:
        subprocess.run(
            ["python", "adw/create_config.py", f"A large version of the {arch_name} architecture with maximal parameters for best performance", "--name", f"{arch_name}_large", "--model", model_choice],
            check=True
        )
        print(f"✅ Created large {arch_name} configuration")
    except subprocess.CalledProcessError as e:
        print(f"Error creating large configuration: {e}")
    
    # Create a configuration that will fail instantiation
    try:
        subprocess.run(
            ["python", "adw/create_config.py", f"An invalid configuration for the {arch_name} architecture", "--name", f"{arch_name}_invalid", "--model", model_choice],
            check=True
        )
        print(f"✅ Created invalid {arch_name} configuration")
    except subprocess.CalledProcessError as e:
        print(f"Error creating invalid configuration: {e}")

def ensure_project_root() -> None:
    """
    Ensure that we're running from the project root directory.
    """
    # Check if we're in the project root
    if not (Path.cwd() / "src" / "models").exists():
        print("Error: Not in the project root directory")
        print("Please run this script from the project root directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
