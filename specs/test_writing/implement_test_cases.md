# Guide: Implementing Test Suites with pytest

This guide explains how to translate structured test cases into working pytest implementations. It covers the pytest framework, test structure, assertion patterns, mocking techniques, and best practices.

## Input Understanding

You'll receive:
1. A **structured list of test cases** for a component
2. Possibly the **component implementation** or description
3. **Context information** about usage and dependencies

## Output Format

You'll produce pytest-compatible test files organized as:

```python
"""Test module for [component name]."""

import pytest
# Import the component under test
from module.path import ComponentName
# Import dependencies for mocking
from module.path import Dependency

# Optional fixtures
@pytest.fixture
def fixture_name():
    """Description of fixture purpose."""
    # Set up code
    yield setup_object
    # Optional teardown code

# Test functions grouped by category
class TestCategory1:
    """Tests for [category description]."""
    
    def test_case_1_1_descriptive_name(self, fixture_name):
        """Test case 1.1: [Brief description]."""
        # Arrange
        # ...
        
        # Act
        # ...
        
        # Assert
        # ...
    
    def test_case_1_2_descriptive_name(self):
        """Test case 1.2: [Brief description]."""
        # ...

class TestCategory2:
    """Tests for [category description]."""
    # ...
```

## Standard Test Directory Structure

When implementing pytest tests, the directory structure should be organized in a way that mirrors your source code and makes it easy to locate and manage tests. Here's the standard pytest test directory structure, with a focus on unit tests:

### Basic Unit Test Directory Structure

```
project_root/
├── src/                          # Source code directory
│   ├── package_name/             # Main package
│   │   ├── __init__.py
│   │   ├── module1.py
│   │   ├── module2.py
│   │   └── subpackage/
│   │       ├── __init__.py
│   │       └── module3.py
├── tests/                        # Test root directory
│   ├── __init__.py               # Makes tests importable
│   ├── conftest.py               # Shared fixtures for all tests
│   ├── unit/                     # Unit tests directory
│   │   ├── __init__.py
│   │   ├── conftest.py           # Fixtures specific to unit tests
│   │   ├── test_module1.py       # Tests for module1.py
│   │   ├── test_module2.py       # Tests for module2.py
│   │   └── subpackage/           # Mirror the src structure
│   │       ├── __init__.py
│   │       ├── conftest.py       # Subpackage-specific fixtures
│   │       └── test_module3.py   # Tests for module3.py
│   ├── integration/              # Integration tests directory
│   └── functional/               # Functional tests directory
├── setup.py                      # Project setup file
└── pytest.ini                    # pytest configuration
```

### Key Structural Components

1. **Mirror the Source Structure**: The directory structure under `tests/unit/` typically mirrors your source code's structure, making it easy to locate tests for specific components.

2. **Naming Conventions**:
   - Test files are prefixed with `test_` (e.g., `test_module1.py`)
   - Test functions and methods are also prefixed with `test_`
   - Test classes are prefixed with `Test` (e.g., `TestUserAccount`)

3. **Test Isolation by Type**:
   - `unit/`: Tests for individual components in isolation
   - `integration/`: Tests for component interactions
   - `functional/`: End-to-end tests of application features

4. **Fixture Hierarchy**:
   - Root `conftest.py`: Global fixtures available to all tests
   - Directory-level `conftest.py`: Fixtures available to tests in that directory and subdirectories

### Alternative Structures

For simpler projects, a flatter structure might be used:

```
project_root/
├── my_package/
│   ├── __init__.py
│   ├── module1.py
│   └── module2.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_module1.py
    └── test_module2.py
```

For larger projects, tests might be further categorized:

```
tests/
├── unit/
│   ├── models/              # Tests for data models
│   │   └── test_user.py
│   ├── views/               # Tests for view functions
│   │   └── test_user_views.py
│   └── utils/               # Tests for utility functions
│       └── test_formatters.py
├── integration/
│   └── ...
└── conftest.py
```

### Common Test Directory Files

- `__init__.py`: Makes the directory a Python package; can be empty
- `conftest.py`: Contains shared fixtures and hooks
- `pytest.ini`: Configuration file for pytest options
- `test_*.py`: Test modules containing test functions/classes

## Step-by-Step Implementation Approach

### 1. Set up the Test Environment

1. **Create the test file structure**:
   - Name the file `test_[component_name].py`
   - Add docstring explaining what's being tested
   - Import the component and dependencies

2. **Define necessary fixtures**:
   - Common test objects
   - Mock setups
   - Environment configurations

3. **Group tests by category**:
   - Use classes for test categories
   - Add descriptive docstrings

### 2. Implement Individual Test Cases

For each test case:

1. **Create test function**:
   - Name: `test_[descriptive_name]`
   - Docstring: Brief description of what's being tested

2. **Follow AAA pattern**:
   - **Arrange**: Set up test conditions
   - **Act**: Execute the code under test
   - **Assert**: Verify expected outcomes

3. **Include test metadata**:
   - Add pytest marks for categorization if needed
   - Use clear comments for complex logic

### 3. Implement Assertions

Use appropriate assertion patterns:

1. **Basic assertions**:
   - `assert value == expected`
   - `assert condition is True/False`

2. **Collection assertions**:
   - `assert item in collection`
   - `assert len(collection) == expected_length`

3. **Exception assertions**:
   - ```python
     with pytest.raises(ExceptionType) as excinfo:
         # code that should raise exception
     assert "expected message" in str(excinfo.value)
     ```

4. **Approximate assertions**:
   - `assert value == pytest.approx(expected, rel=1e-6)`

5. **Custom assertions**:
   - Create helper functions for complex verifications

### 4. Implement Mocking

Use pytest-mock (fixture `mocker`) for dependency isolation:

1. **Function mocks**:
   ```python
   def test_with_mock(mocker):
       mock_func = mocker.patch('module.function')
       mock_func.return_value = 'mocked result'
       # Test code that calls module.function
       assert mock_func.called
   ```

2. **Class mocks**:
   ```python
   def test_with_class_mock(mocker):
       MockClass = mocker.patch('module.ClassName')
       MockClass.return_value.method.return_value = 'result'
       # Test code that instantiates and uses ClassName
       assert MockClass.return_value.method.called
   ```

3. **Object attribute mocks**:
   ```python
   def test_with_attribute_mock(mocker):
       obj = ClassName()
       mocker.patch.object(obj, 'attribute', 'mocked value')
       # Test code that uses obj.attribute
   ```

4. **Context mocking**:
   ```python
   def test_with_context_mock(mocker):
       mock_context = mocker.patch('module.context_manager')
       mock_context.return_value.__enter__.return_value = 'mocked resource'
       # Test code that uses with module.context_manager() as resource:
   ```

### 5. Implement Parameterized Tests

For test cases with multiple input variations:

```python
@pytest.mark.parametrize(
    "input_value,expected_result",
    [
        (5, 25),
        (0, 0),
        (-5, 25),
    ]
)
def test_square(input_value, expected_result):
    assert square(input_value) == expected_result
```

### 6. Add Test Markers

Use markers to categorize tests:

```python
@pytest.mark.critical
def test_critical_functionality():
    # ...

@pytest.mark.slow
def test_performance_intensive():
    # ...
```

## Example Implementations

### Example 1: Function Test Implementation

```python
"""Tests for the calculate_discount function."""

import pytest
from shop.pricing import calculate_discount

class TestInputValidation:
    """Tests that verify the function correctly validates inputs."""
    
    def test_valid_inputs(self):
        """Test case 1.1: Function works with valid price and percentage."""
        # Arrange
        price = 100.0
        percentage = 20.0
        
        # Act
        result = calculate_discount(price, percentage)
        
        # Assert
        assert result == 80.0
    
    def test_negative_price(self):
        """Test case 1.2: Function raises ValueError for negative price."""
        # Arrange
        price = -100.0
        percentage = 20.0
        
        # Act & Assert
        with pytest.raises(ValueError) as excinfo:
            calculate_discount(price, percentage)
        
        assert "Price must be non-negative" in str(excinfo.value)
    
    def test_percentage_out_of_range(self):
        """Test case 1.3: Function raises ValueError for percentage outside 0-100."""
        # Arrange
        price = 100.0
        percentage = 120.0
        
        # Act & Assert
        with pytest.raises(ValueError) as excinfo:
            calculate_discount(price, percentage)
        
        assert "Percentage must be between 0 and 100" in str(excinfo.value)

class TestDiscountCalculation:
    """Tests that verify correct discount calculations."""
    
    def test_zero_discount(self):
        """Test case 2.1: Function returns original price when discount is 0%."""
        assert calculate_discount(100.0, 0.0) == 100.0
    
    def test_full_discount(self):
        """Test case 2.2: Function returns zero when discount is 100%."""
        assert calculate_discount(100.0, 100.0) == 0.0
    
    def test_regular_discount(self):
        """Test case 2.3: Function correctly calculates a typical discount."""
        assert calculate_discount(100.0, 25.0) == 75.0

class TestMaximumDiscountHandling:
    """Tests that verify the max_discount parameter works correctly."""
    
    def test_max_discount_not_reached(self):
        """Test case 3.1: Discount applies normally when below max_discount."""
        result = calculate_discount(100.0, 20.0, max_discount=30.0)
        assert result == 80.0
    
    def test_max_discount_reached(self):
        """Test case 3.2: Discount is capped at max_discount."""
        result = calculate_discount(100.0, 50.0, max_discount=30.0)
        assert result == 70.0
    
    def test_none_max_discount(self):
        """Test case 3.3: Function applies full discount when max_discount is None."""
        result = calculate_discount(100.0, 40.0, max_discount=None)
        assert result == 60.0

class TestEdgeCases:
    """Tests that verify behavior at boundary conditions."""
    
    def test_zero_price(self):
        """Test case 4.1: Function correctly handles zero price."""
        assert calculate_discount(0.0, 20.0) == 0.0
    
    def test_very_small_price(self):
        """Test case 4.2: Function handles very small prices without rounding errors."""
        result = calculate_discount(0.01, 10.0)
        assert result == pytest.approx(0.009)
    
    def test_very_large_price(self):
        """Test case 4.3: Function handles very large prices without overflow."""
        result = calculate_discount(1000000000.0, 10.0)
        assert result == 900000000.0
```

### Example 2: Class Test Implementation

```python
"""Tests for the UserAccount class."""

import pytest
from users.account import UserAccount
from users.security import PasswordHasher
from users.validation import EmailValidator, ValidationError

@pytest.fixture
def valid_user_data():
    """Returns valid test data for user creation."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "P@ssw0rd"
    }

@pytest.fixture
def user_account(valid_user_data, mocker):
    """Creates a UserAccount with mocked dependencies."""
    # Mock the password hasher
    mocker.patch.object(PasswordHasher, 'hash', return_value='hashed_password')
    mocker.patch.object(PasswordHasher, 'verify', return_value=True)
    
    # Mock the email validator
    mocker.patch.object(EmailValidator, 'validate')
    
    # Create and return user account
    return UserAccount(**valid_user_data)

class TestInitialization:
    """Tests for object creation and parameter validation."""
    
    def test_valid_initialization(self, valid_user_data, mocker):
        """Test case 1.1: UserAccount initializes with valid parameters."""
        # Mock dependencies
        mock_hash = mocker.patch.object(PasswordHasher, 'hash', return_value='hashed_password')
        mock_validate = mocker.patch.object(EmailValidator, 'validate')
        
        # Act
        account = UserAccount(**valid_user_data)
        
        # Assert
        assert account.username == valid_user_data["username"]
        assert account.email == valid_user_data["email"]
        assert account.password == 'hashed_password'  # Password should be hashed
        mock_hash.assert_called_once_with(valid_user_data["password"])
        mock_validate.assert_called_once_with(valid_user_data["email"])
    
    def test_invalid_email(self, valid_user_data, mocker):
        """Test case 1.2: UserAccount raises ValueError for invalid email."""
        # Mock dependencies
        mocker.patch.object(PasswordHasher, 'hash', return_value='hashed_password')
        mock_validate = mocker.patch.object(
            EmailValidator, 'validate', 
            side_effect=ValidationError("Invalid email format")
        )
        
        # Act & Assert
        with pytest.raises(ValueError) as excinfo:
            UserAccount(**valid_user_data)
        
        assert "Invalid email format" in str(excinfo.value)
        mock_validate.assert_called_once_with(valid_user_data["email"])

class TestAuthentication:
    """Tests for the authenticate method."""
    
    def test_correct_password(self, user_account, mocker):
        """Test case 2.1: authenticate returns True for correct password."""
        # Mock dependencies
        mock_verify = mocker.patch.object(PasswordHasher, 'verify', return_value=True)
        
        # Act
        result = user_account.authenticate("P@ssw0rd")
        
        # Assert
        assert result is True
        mock_verify.assert_called_once_with("P@ssw0rd", 'hashed_password')
    
    def test_incorrect_password(self, user_account, mocker):
        """Test case 2.2: authenticate returns False for incorrect password."""
        # Mock dependencies
        mock_verify = mocker.patch.object(PasswordHasher, 'verify', return_value=False)
        
        # Act
        result = user_account.authenticate("WrongP@ss")
        
        # Assert
        assert result is False
        mock_verify.assert_called_once_with("WrongP@ss", 'hashed_password')

class TestPasswordManagement:
    """Tests for the change_password method."""
    
    def test_successful_password_change(self, user_account, mocker):
        """Test case 3.1: change_password updates password when old password is correct."""
        # Mock dependencies
        mocker.patch.object(PasswordHasher, 'verify', return_value=True)
        mock_hash = mocker.patch.object(PasswordHasher, 'hash', return_value='new_hashed_password')
        
        # Act
        result = user_account.change_password("P@ssw0rd", "NewP@ss123")
        
        # Assert
        assert result is True
        assert user_account.password == 'new_hashed_password'
        mock_hash.assert_called_once_with("NewP@ss123")
    
    def test_failed_password_change(self, user_account, mocker):
        """Test case 3.2: change_password fails when old password is incorrect."""
        # Mock dependencies
        mocker.patch.object(PasswordHasher, 'verify', return_value=False)
        mock_hash = mocker.patch.object(PasswordHasher, 'hash')
        
        # Act
        result = user_account.change_password("WrongP@ss", "NewP@ss123")
        
        # Assert
        assert result is False
        assert user_account.password == 'hashed_password'  # Password unchanged
        mock_hash.assert_not_called()

class TestEmailManagement:
    """Tests for the update_email method."""
    
    def test_successful_email_update(self, user_account, mocker):
        """Test case 4.1: update_email changes email when password is correct."""
        # Mock dependencies
        mocker.patch.object(PasswordHasher, 'verify', return_value=True)
        mock_validate = mocker.patch.object(EmailValidator, 'validate')
        
        # Act
        result = user_account.update_email("P@ssw0rd", "new@example.com")
        
        # Assert
        assert result is True
        assert user_account.email == "new@example.com"
        mock_validate.assert_called_once_with("new@example.com")
    
    def test_failed_email_update_with_incorrect_password(self, user_account, mocker):
        """Test case 4.2: update_email fails when password is incorrect."""
        # Mock dependencies
        mocker.patch.object(PasswordHasher, 'verify', return_value=False)
        mock_validate = mocker.patch.object(EmailValidator, 'validate')
        
        # Original email
        original_email = user_account.email
        
        # Act
        result = user_account.update_email("WrongP@ss", "new@example.com")
        
        # Assert
        assert result is False
        assert user_account.email == original_email  # Email unchanged
        mock_validate.assert_not_called()

class TestIntegration:
    """Tests for interactions with dependencies."""
    
    def test_password_hashing_during_initialization(self, valid_user_data, mocker):
        """Test case 5.1: Password is properly hashed during account creation."""
        # Mock with spy to track calls
        mock_hash = mocker.patch.object(PasswordHasher, 'hash', return_value='hashed_password')
        mocker.patch.object(EmailValidator, 'validate')
        
        # Act
        account = UserAccount(**valid_user_data)
        
        # Assert
        mock_hash.assert_called_once_with(valid_user_data["password"])
        assert account.password == 'hashed_password'
    
    def test_email_validation_during_initialization(self, valid_user_data, mocker):
        """Test case 5.2: Email is validated during account creation."""
        # Mock with spy to track calls
        mocker.patch.object(PasswordHasher, 'hash', return_value='hashed_password')
        mock_validate = mocker.patch.object(EmailValidator, 'validate')
        
        # Act
        account = UserAccount(**valid_user_data)
        
        # Assert
        mock_validate.assert_called_once_with(valid_user_data["email"])
```

## Advanced pytest Features

### 1. Fixtures with Scope

Control fixture lifetime:

```python
@pytest.fixture(scope="module")
def expensive_setup():
    # Setup that runs once per module
    resource = setup_expensive_resource()
    yield resource
    # Teardown that runs once after all tests in module
    resource.cleanup()
```

Available scopes: `function` (default), `class`, `module`, `package`, `session`

### 2. Fixture Parametrization

Test with multiple fixture configurations:

```python
@pytest.fixture(params=["small", "medium", "large"])
def data_size(request):
    sizes = {
        "small": 10,
        "medium": 100,
        "large": 1000
    }
    return sizes[request.param]

def test_with_different_sizes(data_size):
    # This test runs three times with different data sizes
    assert process_data([0] * data_size) is not None

### 3. Test Skipping

Skip tests conditionally:

```python
@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires Python 3.9+")
def test_new_feature():
    # Test that uses Python 3.9+ features
    ...

@pytest.mark.skip(reason="Feature not implemented yet")
def test_future_feature():
    # Test for planned feature
    ...
```

### 4. Expected Failures

Mark tests that are expected to fail:

```python
@pytest.mark.xfail(reason="Known bug #123")
def test_with_known_issue():
    # This test is expected to fail
    ...
```

### 5. Setup and Teardown

Add class-level setup/teardown:

```python
class TestWithSetup:
    @classmethod
    def setup_class(cls):
        # Setup that happens once for the class
        cls.resource = setup_shared_resource()
    
    @classmethod
    def teardown_class(cls):
        # Teardown that happens once after all tests in the class
        cls.resource.cleanup()
    
    def setup_method(self):
        # Setup that happens before each test method
        self.reset_data()
    
    def teardown_method(self):
        # Teardown that happens after each test method
        ...
```

## Working with Test Data

### 1. Test Data Files

Use external files for test data:

```python
def get_test_data_path(filename):
    """Return the full path to a test data file."""
    return os.path.join(os.path.dirname(__file__), "test_data", filename)

def test_with_data_file():
    data_path = get_test_data_path("sample.json")
    with open(data_path, "r") as f:
        data = json.load(f)
    # Test using the loaded data
    ...
```

### 2. Temporary Files and Directories

Use pytest's tmp_path fixture:

```python
def test_file_operations(tmp_path):
    # Create a temporary file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    
    # Test function that reads/writes files
    result = process_file(test_file)
    assert result == expected_result
```

## Testing for Different Types of Components

### 1. Testing Async Functions

Use pytest-asyncio:

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected_value
```

### 2. Testing Web Applications

Use pytest-flask, pytest-django, or similar:

```python
def test_homepage(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Welcome' in response.data
```

### 3. Testing Database Operations

Use database fixtures:

```python
@pytest.fixture
def db_connection():
    # Set up test database
    conn = create_test_db_connection()
    yield conn
    # Clean up test database
    conn.rollback()
    conn.close()

def test_db_query(db_connection):
    result = execute_query(db_connection, "SELECT * FROM test_table")
    assert len(result) == expected_count
```

## Best Practices for pytest Implementation

1. **Use descriptive test names**: Make test function names clear about what they test
2. **Keep tests isolated**: Each test should run independently
3. **Use appropriate assertions**: Choose the right assertion for each verification
4. **Mock external dependencies**: Isolate the code under test
5. **Organize with classes**: Group related tests in classes
6. **Use fixtures efficiently**: Share setup code through fixtures
7. **Add docstrings**: Document what each test is verifying
8. **Follow AAA pattern**: Arrange, Act, Assert in each test
9. **Use parameterization**: Test multiple inputs without duplicating code
10. **Keep tests simple**: Complex test logic can hide bugs in the tests themselves

By following these guidelines, you can create structured, maintainable test suites that effectively verify your code's behavior.

## Using pytest Effectively

### Running Tests

```bash
# Run all tests
pytest

# Run tests in specific file
pytest test_module.py

# Run specific test
pytest test_module.py::TestClass::test_function

# Run tests matching pattern
pytest -k "pattern"

# Run tests with specific marker
pytest -m critical
```

### Useful Command Line Options

```bash
# Show verbose output
pytest -v

# Show extra verbose output
pytest -vv

# Exit on first failure
pytest -x

# Show local variables in tracebacks
pytest --showlocals

# Generate test coverage report
pytest --cov=module_name

# Output test results in JUnit XML format
pytest --junitxml=results.xml
```

### conftest.py Configuration

Create a `conftest.py` file for shared fixtures and configuration:

```python
# conftest.py
import pytest

# Define fixtures available to all test files
@pytest.fixture(scope="session")
def app_config():
    return {
        "api_key": "test_key",
        "environment": "testing"
    }

# Register custom markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "critical: mark test as critical"
    )
```

This comprehensive approach to implementing pytest test suites ensures that your tests are well-structured, maintainable, and effective at validating your code's behavior.