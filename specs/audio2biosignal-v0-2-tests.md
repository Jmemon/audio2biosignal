# Audio2Biosignal v0 Tests

## High-Level Objective
Write comprehensive testing for all components of the Audio2Biosignal v0 system, as well as the entire system as a whole.

## Mid-Level Objectives
- Write tests for every function
- Create a range of config files that give a full picture of the system's behavior, where the config is used as input and the state of the Trainer is used as output.

## Implementation Notes
Use pytest to write tests.
Make sure all components (every method, class, function, etc) are tested.
When developing test cases, make sure to test for ALL edge cases and error conditions, and representatively test normal cases. Think through the full space of possible inputs and outputs.
Give tests descriptive names.
Tests should NOT reflect how the function is implemented, but how the function is ideally supposed to behave.

## Context
### Beginning Context
src/**/*.py
tests/

### Ending Context
src/**/*.py
tests/**/*.py

## Low-Level Tasks
