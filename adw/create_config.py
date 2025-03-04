#!/usr/bin/env python3
import argparse
import glob
import random
from typing import List
from pathlib import Path
from aider.coders import Coder
from aider.models import get_model_from_name

def main():
    """
    Main function to parse command line arguments for experiment configuration.
    """
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
        import random
        configs_dir = Path("configs")
        while True:
            rand_int = random.randint(1, 10000)
            experiment_name = f"example_{rand_int}"
            if not (configs_dir / f"{experiment_name}.yml").exists():
                break
    
    # Create list of read-only files
    read_only_files = get_read_only_files()
    
    # Create aider Coder instance
    model = get_model_from_name("claude-3-5-sonnet-latest")
    coder = Coder(model=model)
    
    # Create empty prompt string
    prompt = ""
    
    # TODO: Implement the rest of the functionality

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
