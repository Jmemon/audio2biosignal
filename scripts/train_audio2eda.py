import argparse
import yaml
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.configs import RunConfig
from src.train import train

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Train Audio2EDA Model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create RunConfig from the loaded config
    run_config = RunConfig(**config_dict)
    
    # Ensure checkpoint directory exists
    os.makedirs(run_config.checkpoint.checkpoint_dir, exist_ok=True)
    
    # Call the train function with the config
    best_checkpoint_path = train(run_config)
    
    print(f"Training completed. Best checkpoint saved at: {best_checkpoint_path}")

if __name__ == "__main__":
    main()
