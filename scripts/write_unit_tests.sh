#!/bin/bash

# Script to automatically generate unit tests for all components in src directory

# Ensure we're in the project root
if [[ $(basename "$PWD") != "audio2biosignal" ]]; then
    echo "Error: Must run from project root directory (audio2biosignal)"
    exit 1
fi

# Create a temporary file to store component names
TEMP_FILE=$(mktemp)

# Function to extract top-level components from a Python file
extract_components() {
    local file=$1
    # Use grep to find class and function definitions at the beginning of lines
    # This regex looks for lines that start with "class " or "def " (with possible indentation)
    grep -E "^[[:space:]]*(class|def)[[:space:]]+" "$file" | 
    # Extract the component name (class name or function name)
    sed -E 's/^[[:space:]]*(class|def)[[:space:]]+([a-zA-Z0-9_]+).*/\2/' > "$TEMP_FILE"
    
    # Read the components from the temp file
    while read -r component; do
        echo "$component"
    done < "$TEMP_FILE"
}

# Find all Python files in src directory
find src -name "*.py" | while read -r file; do
    echo "Processing file: $file"
    
    # Skip __init__.py files as they typically don't contain testable components
    if [[ $(basename "$file") == "__init__.py" ]]; then
        echo "Skipping __init__.py file"
        continue
    fi
    
    # Extract components from the file
    components=$(extract_components "$file")
    
    # If no components found, continue to next file
    if [[ -z "$components" ]]; then
        echo "No components found in $file"
        continue
    fi
    
    # Process each component
    echo "$components" | while read -r component; do
        if [[ -n "$component" ]]; then
            echo "Generating tests for component: $component in $file"
            python adw/create_tests.py "Test $component from $file"
            
            # Sleep briefly to avoid overwhelming the API
            sleep 2
        fi
    done
done

# Clean up
rm -f "$TEMP_FILE"

echo "Test generation complete!"
