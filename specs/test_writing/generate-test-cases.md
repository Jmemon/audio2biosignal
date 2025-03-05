# Test Case Generator
<!-- 
USAGE INSTRUCTIONS:
1. Load this specification into a string variable
2. Replace the following tags with actual values:
   - <TARGET_FILE>: Path to the file containing the code to test
   - <TARGET_CODE>: The actual code to generate tests for (full content or reference)
   - <TARGET_DEPENDENTS>: Comma-separated list of files that use the code being tested
   - <OUTPUT_PATH>: Path where the generated JSON file should be saved
3. Feed the processed specification to the code generation model
4. Parse and validate the generated test cases
-->

## High-Level Objective
Generate comprehensive, well-structured test cases for the target software component based on signature, implementation details, and usage examples:
```python
<TARGET_CODE>
```
DO NOT WRITE ANY TEST CODE, ONLY THE JSON OUTPUT AS OUTLINED IN LOW-LEVEL TASKS.

## Mid-Level Objectives
<!-- High-level processing flow -->
1. Analyze the target component to understand its purpose, inputs, outputs, dependencies, and usage contexts.
2. Define appropriate test categories based on the component type and functionality.
3. Generate specific test cases covering happy paths, edge cases, error conditions, and corner cases.
4. Prioritize test cases based on criticality and importance.
5. Structure the output in a clear, organized format that can be easily translated into executable tests.

## Implementation Notes

### Test Design Techniques
- **Equivalence Partitioning**: Group inputs into classes that should behave similarly:
  - Valid inputs that should work
  - Invalid inputs that should fail in specific ways
- **Boundary Value Analysis**: Test values at the boundaries of valid/invalid ranges:
  - Minimum valid values
  - Maximum valid values
  - Just below/above valid ranges
- **State Transition Testing**: For stateful components, test transitions between states:
  - State A → State B
  - Invalid state transitions

### Test Categories By Component Type
- **For Functions**:
  - Input Validation Tests: Verify handling of different input types and values
  - Output Verification Tests: Verify correct return values
  - Exception Handling Tests: Verify appropriate exceptions are raised
  - Side Effect Tests: Verify any expected side effects
  - Performance Tests: Verify function performs within acceptable parameters (if applicable)
- **For Methods**:
  - State Change Tests: Verify object state changes correctly
  - Input Validation Tests: Verify method handles inputs correctly
  - Return Value Tests: Verify correct return values
  - Exception Tests: Verify appropriate exceptions are raised
  - Interaction Tests: Verify interactions with other methods/objects
- **For Classes**:
  - Initialization Tests: Verify constructor behavior
  - Method Interaction Tests: Verify methods work together correctly
  - State Management Tests: Verify object maintains consistent state
  - Interface Tests: Verify all public methods work as expected
  - Lifecycle Tests: Verify creation/modification/destruction processes

### Test Case Construction
- Each test case should verify one specific behavior or scenario
- Cover four key scenario types:
  - **Happy Paths**: Normal, expected usage scenarios
  - **Edge Cases**: Boundary conditions and unusual but valid inputs
  - **Error Cases**: Invalid inputs and error conditions
  - **Corner Cases**: Rare or extreme scenarios
- Include necessary mocking information for dependencies:
  - External services
  - File systems, databases
  - Randomness or time-dependent behavior
- Use consistent naming conventions for test cases
- Provide clear descriptions explaining what each test verifies
- Specify exact input values and expected outputs

### Prioritization and Tagging
- Tag test cases with priority levels:
  - **Critical**: Tests that verify core functionality
  - **High**: Tests for important but not critical features
  - **Medium**: Tests for edge cases and less common scenarios
  - **Low**: Tests for very rare scenarios
- Add additional tags to facilitate filtering:
  - Test level (unit/integration)
  - Test type (happy_path/edge_case/error_case)

### Output Structure
- Organize test cases by category
- Use consistent formatting across all test cases
- Ensure all test cases collectively achieve high code coverage
- Structure output for easy translation into executable tests

## Target Component
```python
<TARGET_CODE>
```

## Low-Level Tasks
<!-- These tasks map directly to JSON output fields -->

1. **Component Analysis** (Populates the `component_analysis` object in JSON)
   - Extract component name → `component_analysis.name`
   - Identify component type (function, method, or class) → `component_analysis.type`
   - Document component purpose → `component_analysis.purpose`
   - List parameter names, types, and default values → `component_analysis.inputs`
   - Determine return type(s) and possible exceptions → `component_analysis.outputs`
   - Identify external dependencies → `component_analysis.dependencies`
   - Document typical usage patterns → `component_analysis.usage_contexts`

2. **Test Category Definition** (Populates the `test_categories` array in JSON)
   - For each test category:
     - Define category name → `test_categories[i].name`
     - Write category description → `test_categories[i].description`
   - Create appropriate categories based on component type:
     - For functions: input validation, output verification, exception handling, side effects, performance
     - For methods: state changes, input validation, return values, exceptions, interactions
     - For classes: initialization, method interaction, state management, interface, lifecycle

3. **Test Case Generation** (Populates individual test case objects in the `test_cases` array)
   - For each test case:
     - Generate unique ID → `test_cases[i].id`
     - Create descriptive name → `test_cases[i].name`
     - Assign to appropriate category → `test_cases[i].category`
     - Write description of what is being verified → `test_cases[i].description`
     - Define specific input values → `test_cases[i].inputs`
     - Specify expected output or behavior → `test_cases[i].expected_output`
     - Identify mocking requirements:
       - Dependency name → `test_cases[i].mocks[j].dependency`
       - Method to mock → `test_cases[i].mocks[j].method`
       - Return value → `test_cases[i].mocks[j].return_value`

4. **Test Case Prioritization** (Populates priority and tags fields for each test case)
   - Assign priority level → `test_cases[i].priority`
     - "critical" for core functionality
     - "high" for important features
     - "medium" for edge cases
     - "low" for rare scenarios
   - Apply appropriate tags → `test_cases[i].tags`
     - Test level (unit/integration)
     - Test type (happy_path/edge_case/error_case)

5. **Output Formatting**
   - Generate output as a structured JSON file at the path specified by <OUTPUT_PATH>
   - Format the JSON as an array of test case objects with the following schema:
```aider
CREATE <OUTPUT_PATH>:
     {
       "metadata": {
         "source_file": "<TARGET_FILE>",
         "dependents": "<TARGET_DEPENDENTS>",
         "generated_at": "YYYY-MM-DD HH:MM:SS",
         "version": "1.0"
       },
       "component_analysis": {
         "name": "string",                  // Component name
         "type": "string",                  // "function", "method", or "class"
         "purpose": "string",               // Brief description
         "inputs": ["string"],              // List of inputs
         "outputs": ["string"],             // List of outputs
         "dependencies": ["string"],        // List of dependencies
         "usage_contexts": ["string"]       // List of usage contexts
       },
       "test_categories": [
         {
           "name": "string",                // Category name
           "description": "string"          // Category description
         }
       ],
       "test_cases": [
         {
           "id": "string",                  // Unique identifier (e.g., "1.1", "2.3")
           "name": "string",                // Descriptive name
           "category": "string",            // Test category name
           "description": "string",         // What this test verifies
           "inputs": {                      // Key-value pairs of input parameters
             "param1": "value1",
             "param2": "value2"
           },
           "expected_output": "any",        // Expected return value or behavior
           "mocks": [                       // Array of dependencies to mock
             {
               "dependency": "string",      // Name of dependency
               "method": "string",          // Method to mock
               "return_value": "any"        // Value to return
             }
           ],
           "tags": ["string"],              // Array of tags
           "priority": "string"             // "critical", "high", "medium", or "low"
         }
       ]
     }
```
   - Ensure all test cases have unique IDs following the pattern "CategoryNumber.TestNumber"
   - Use descriptive names for all test cases that clearly indicate what is being tested
   - Include all relevant input parameters with their exact values
   - Specify precise expected outputs or behaviors
   - Document all mocking requirements with dependency names, methods, and return values
   - Tag each test case appropriately for filtering and prioritization