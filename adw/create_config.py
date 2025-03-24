#!/usr/bin/env python3
import argparse
import glob
import random
import datetime
import os
import sys
from typing import List
from pathlib import Path
from dotenv import load_dotenv
from aider.coders import Coder
from aider.models import Model
import anthropic

def main():
    """
    Main function to parse command line arguments for experiment configuration.
    """
    # Load environment variables
    load_dotenv()
    
    # Verify we're in the project root directory
    ensure_project_root()
    
    parser = argparse.ArgumentParser(
        description="Generate a configuration file for audio-to-EDA experiments"
    )
    parser.add_argument(
        "description", 
        help="Description of the experiment to generate a configuration for"
    )
    parser.add_argument(
        "--name", "-n", 
        help="Name for the experiment (used for the config filename)"
    )
    
    args = parser.parse_args()
    
    # Generate a unique experiment name if not provided
    experiment_name = args.name
    if not experiment_name:
        # Generate a brief name using Haiku to summarize the description
        experiment_name = generate_experiment_name(args.description)
        
        # Add timestamp for uniqueness
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{experiment_name}_{current_time}"
    
    # Create list of read-only files
    read_only_files = get_read_only_files()
    
    # Create aider Coder instance
    model = Model("sonnet")
    coder = Coder.create(
        main_model=model,
        edit_format="diff",
        read_only_fnames=read_only_files,
        suggest_shell_commands=False,
        auto_commits=False
    )
    
    # Create prompt string with experiment description
    prompt = f"""CREATE configs/{experiment_name}.yml ensuring that it satisfies ```{args.description}``` and making intelligent choices for any unspecified fields keeping in mind the overall purpose of this repo: to train model that translate audio as MFCCs to EDA signals."""
    
    # Run the coder with the prompt
    print(f"Generating configuration for experiment: {experiment_name}")
    print(f"Description: {args.description}")
    print(f"Using {len(read_only_files)} reference files for context")
    
    response = coder.run(prompt)
    print(f"Configuration generated for {experiment_name}")

def generate_experiment_name(description: str) -> str:
    """
    Generate a brief experiment name by using Anthropic's Haiku API to summarize the description.
    
    Args:
        description: The experiment description to summarize
        
    Returns:
        A brief, slug-friendly name for the experiment
    """
    # Get API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: ANTHROPIC_API_KEY not found in environment. Using fallback naming.")
        return "audio2eda"
    
    try:
        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Create prompt for Haiku
        prompt = f"""
        Summarize this experiment description into a very brief, memorable name (3-4 words max) 
        that captures its essence. Use only lowercase letters, numbers, and hyphens (no spaces).
        
        Description: {description}
        
        Name:
        """
        
        # Call Haiku API
        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=50,
            temperature=0.7,
            system="You create brief, memorable experiment names.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract and clean the name
        name = response.content[0].text.strip().lower()
        
        # Remove any non-alphanumeric characters except hyphens
        import re
        name = re.sub(r'[^a-z0-9\-]', '-', name)
        
        # Ensure no consecutive hyphens
        name = re.sub(r'-+', '-', name)
        
        # Trim hyphens from start and end
        name = name.strip('-')
        
        return name if name else "audio2eda"
    
    except Exception as e:
        print(f"Error generating experiment name: {e}")
        return "audio2eda"

def ensure_project_root():
    """
    Ensure that we're running from the project root directory (audio2biosignal).
    Check if we're in the project root or if the current directory is named audio2biosignal.
    Otherwise, raise an error.
    """
    current_dir = Path.cwd()
    current_dir_name = current_dir.name
    
    # Check if we're already in the project root
    if (current_dir_name == "audio2biosignal" and 
        (current_dir / "src").exists() and 
        (current_dir / "configs").exists()):
        return
    
    # Not in the project directory
    print("Error: Not in the audio2biosignal project root directory.")
    print("Please run this script from the project root directory.")
    sys.exit(1)

def get_read_only_files() -> List[str]:
    """
    Get a list of read-only files including up to 5 configs/*.yml files and configuration files
    """
    read_only_files = []
    
    # Add up to 5 YAML files from configs directory
    config_files = glob.glob("configs/*.yml")
    read_only_files.extend(config_files[:5])
    
    # Add configuration files
    read_only_files.append("src/configs.py")
    read_only_files.append("src/models/configs.py")
    read_only_files.append("src/data/configs.py")
    
    return read_only_files

if __name__ == "__main__":
    main()
