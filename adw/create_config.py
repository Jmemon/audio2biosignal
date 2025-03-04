#!/usr/bin/env python3
import argparse
import glob
import random
import datetime
import os
import sys
from typing import List
from pathlib import Path
from aider.coders import Coder
from aider.models import get_model_from_name

def main():
    """
    Main function to parse command line arguments for experiment configuration.
    """
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
        # Use current datetime for unique name
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{current_time}"
    
    # Create list of read-only files
    read_only_files = get_read_only_files()
    
    # Create aider Coder instance
    model = get_model_from_name("claude-3-5-sonnet-latest")
    coder = Coder(model=model)
    
    # Create prompt string with experiment description
    prompt = f"""CREATE configs/{experiment_name}.yml ensuring that it satisfies ```{args.description}``` and making intelligent choices for any unspecified fields keeping in mind the overall purpose of this repo: to train model that translate audio as MFCCs to EDA signals."""
    
    # TODO: Implement the rest of the functionality

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
    Get a list of read-only files including up to 5 configs/*.yml files and src/configs.py
    """
    read_only_files = []
    
    # Add up to 5 YAML files from configs directory
    config_files = glob.glob("configs/*.yml")
    read_only_files.extend(config_files[:5])
    
    # Add src/configs.py
    read_only_files.append("src/configs.py")
    
    return read_only_files

if __name__ == "__main__":
    main()
