# BULLETPROOF TEST SUITE GENERATOR

## CORE MISSION
Generate industrial-strength, production-hardened test suites that expose defects, protect against regressions, and document expected behavior with surgical precision.

## COMPONENT SUBMISSION
```python
# TARGET COMPONENT (REQUIRED)
<TARGET_CODE>

# DEPENDENT CODE (OPTIONAL)
<DEPENDENT_CODE>

# EXISTING TESTS (OPTIONAL)
<EXISTING_TESTS>
```

## RULES
- Add tests to the test file `<TEST_FILE_PATH>`, do not create a new file.
- Follow the test directory structure described below.
- Organize tests into classes reflecting what component you are testing and potentially an overarching category of tasks.

## TEST DIRECTORY STRUCTURE

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

## OUTPUT: INDUSTRIAL-GRADE TEST SUITE

You will receive a comprehensive test file with:

1. Strategic test organization with failure isolation
2. Complete coverage of code paths and edge cases
3. Properly mocked external dependencies
4. Advanced pytest features when appropriate
5. Defensive test patterns that anticipate future changes

## COMPONENT DISSECTION

### 1. Interface Analysis
Systematically extract:
- Function signature(s) / Class API
- Parameter types, validation rules, and default values
- Return types and structures
- Side effects (file I/O, network, database)
- Exception patterns

### 2. Dependency Mapping
Identify all:
- Internal imports
- External services
- System resources (filesystem, network, etc.)
- Global state dependencies
- Hidden coupling points

### 3. Behavior Identification
Document all:
- Core functionality
- Edge cases and boundary conditions
- Error handling pathways
- Performance characteristics
- Thread safety concerns

## TEST ARCHITECTURE

Structure tests with military precision:

```python
"""
Test suite for [component name].

This module provides comprehensive testing for:
1. [Core functionality 1]
2. [Core functionality 2]
3. [Edge cases]
4. [Error handling]
"""

import pytest
from unittest.mock import patch, MagicMock
# Import target component and dependencies

# === FIXTURES ===

@pytest.fixture
def standard_fixture():
    """
    Provides a standardized test environment for [purpose].
    
    Returns:
        [type]: [description]
    """
    # Setup
    yield setup_object
    # Teardown (if needed)

@pytest.fixture(scope="module")
def expensive_fixture():
    """
    Provides [resource] shared across multiple tests.
    Created once per test module for efficiency.
    """
    # Heavy setup
    yield resource
    # Resource cleanup

# === TEST CATEGORIES ===

class TestInitialization:
    """Tests covering object creation and parameter validation."""
    
    def test_valid_initialization(self):
        """
        GIVEN valid initialization parameters
        WHEN the object is created
        THEN it should initialize with correct default state
        """
        # Implementation
    
    # More initialization tests...

class TestCoreOperation:
    """Tests covering primary functionality under normal conditions."""
    
    # Core operation tests...

class TestErrorHandling:
    """Tests covering failure modes and error conditions."""
    
    # Error condition tests...

class TestEdgeCases:
    """Tests covering boundary conditions and unusual inputs."""
    
    # Edge case tests...

class TestPerformance:
    """Tests covering performance characteristics (when relevant)."""
    
    # Performance tests...
```

## ADVANCED TESTING PATTERNS

### 1. Behavior Verification Pattern
```python
def test_behavior_verification(mocker):
    """Verify component interacts correctly with dependencies."""
    # Setup mocks
    mock_dependency = mocker.patch('module.Dependency')
    mock_instance = mock_dependency.return_value
    
    # Configure mock behavior
    mock_instance.method.return_value = expected_value
    
    # Execute component under test
    result = component_under_test(test_input)
    
    # Verify interactions occurred correctly
    mock_instance.method.assert_called_once_with(expected_args)
    assert result == expected_output
```

### 2. State Transition Testing
```python
def test_state_transition(self):
    """
    Verify component transitions correctly between states.
    
    State diagram being tested:
    INIT --[action]--> STATE_A --[action2]--> STATE_B
    """
    # Create component in initial state
    component = ComponentUnderTest()
    assert component.state == "INIT"
    
    # Transition to first state
    component.action()
    assert component.state == "STATE_A"
    
    # Transition to second state
    component.action2()
    assert component.state == "STATE_B"
```

### 3. Data Factory Pattern
```python
@pytest.fixture
def user_factory():
    """Factory for creating test users with customizable attributes."""
    def _create_user(**kwargs):
        defaults = {
            "username": "testuser",
            "email": "test@example.com",
            "is_active": True,
            "permissions": ["read"]
        }
        # Override defaults with any provided kwargs
        params = {**defaults, **kwargs}
        return User(**params)
    return _create_user

def test_with_factory(user_factory):
    """Test using factory pattern for flexible test data."""
    # Create different user variations
    basic_user = user_factory()
    admin_user = user_factory(permissions=["read", "write", "admin"])
    inactive_user = user_factory(is_active=False)
    
    # Use users in tests...
```

### 4. Snapshot Testing Pattern
```python
def test_complex_output_structure(snapshot):
    """
    Verify complex output structure matches expected format.
    
    Rather than manually asserting on each field, we use snapshot
    testing to verify the entire structure hasn't changed unexpectedly.
    """
    result = complex_function_under_test()
    snapshot.assert_match(result)
```

### 5. Parameterized Test Matrices
```python
@pytest.mark.parametrize("input_value,expected_status,expected_message", [
    # (input, status, message)
    ({"valid": "data"}, 200, "Success"),
    ({"missing_field": None}, 400, "Missing required field"),
    ({"invalid_type": 123}, 400, "Type error in field"),
    (None, 500, "Internal error"),
])
def test_validation_matrix(input_value, expected_status, expected_message):
    """Test validation logic against multiple input scenarios."""
    result = validate_function(input_value)
    assert result.status == expected_status
    assert expected_message in result.message
```

## MOCKING STRATEGIES

### 1. External API Mocking
```python
@pytest.fixture
def mock_external_api(mocker):
    """Mock external API responses for testing."""
    mock = mocker.patch('requests.get')
    
    # Configure the mock to return different responses based on URL
    def side_effect(url, *args, **kwargs):
        mock_response = mocker.MagicMock()
        
        if url.endswith('/users'):
            mock_response.status_code = 200
            mock_response.json.return_value = {'users': [{'id': 1, 'name': 'Test'}]}
        elif url.endswith('/error'):
            mock_response.status_code = 500
            mock_response.json.side_effect = ValueError("Invalid JSON")
        else:
            mock_response.status_code = 404
            
        return mock_response
        
    mock.side_effect = side_effect
    return mock

def test_api_client(mock_external_api):
    """Test client handles different API responses correctly."""
    client = APIClient()
    
    # Test successful response
    users = client.get_users()
    assert len(users) == 1
    assert users[0]['name'] == 'Test'
    
    # Test error handling
    with pytest.raises(APIError):
        client.get_error_endpoint()
```

### 2. Context Manager Mocking
```python
def test_context_manager_usage(mocker):
    """Test code that uses a context manager."""
    mock_context = mocker.patch('module.resource_context')
    mock_resource = mocker.MagicMock()
    mock_context.return_value.__enter__.return_value = mock_resource
    
    # Configure mock resource behavior
    mock_resource.get_data.return_value = "test_data"
    
    # Test function that uses the context manager
    result = function_under_test()
    
    # Verify context was used correctly
    mock_context.assert_called_once()
    mock_resource.get_data.assert_called_once()
    assert result == "expected_result"
```

### 3. Database Mocking
```python
@pytest.fixture
def mock_db_session(mocker):
    """Mock database session for testing database operations."""
    mock_session = mocker.MagicMock()
    mock_query = mocker.MagicMock()
    
    # Configure the mock query chain
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.first.return_value = mock_user_result()
    
    # Mock transaction context
    mock_session.begin.return_value.__enter__.return_value = None
    
    # Return pre-configured session
    return mock_session

def test_db_operations(mock_db_session):
    """Test database operations with mocked session."""
    user_service = UserService(session=mock_db_session)
    user = user_service.get_user(user_id=1)
    
    # Verify database was queried correctly
    mock_db_session.query.assert_called_once_with(User)
    mock_db_session.query().filter.assert_called_once()
    assert user.id == 1
```

## SPECIFIC TEST CATEGORIES

### 1. Input Validation Tests
Test both valid inputs and all possible invalid inputs:

```python
class TestInputValidation:
    """Verify all input validation logic."""
    
    def test_valid_input_accepted(self):
        """Valid inputs should be processed without errors."""
        result = function_under_test(valid_input)
        assert isinstance(result, ExpectedType)
    
    def test_none_input_rejected(self):
        """None input should raise TypeError with clear message."""
        with pytest.raises(TypeError) as excinfo:
            function_under_test(None)
        assert "Cannot process None" in str(excinfo.value)
    
    @pytest.mark.parametrize("invalid_input,expected_error,expected_message", [
        ("", ValueError, "Empty input not allowed"),
        ({}, TypeError, "Expected string input"),
        ("invalid_format", ValueError, "Input format invalid"),
    ])
    def test_invalid_inputs(self, invalid_input, expected_error, expected_message):
        """Each invalid input type should raise appropriate error."""
        with pytest.raises(expected_error) as excinfo:
            function_under_test(invalid_input)
        assert expected_message in str(excinfo.value)
```

### 2. Concurrency and Thread Safety Tests
For components that should be thread-safe:

```python
def test_thread_safety():
    """Verify component behaves correctly under concurrent access."""
    import concurrent.futures
    
    # Shared component instance
    component = ComponentUnderTest()
    
    # Function to execute in parallel
    def concurrent_operation(value):
        return component.process(value)
    
    # Execute operations concurrently
    values = list(range(100))
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(concurrent_operation, values))
    
    # Verify results
    assert len(results) == len(values)
    assert component.internal_state_is_valid()
```

### 3. Resource Management Tests
For components that manage resources:

```python
def test_resource_acquisition_and_release():
    """Verify resources are properly acquired and released."""
    
    # Track resource usage
    resource_tracker = mocker.patch('module.resource_tracker')
    resource_tracker.active_resources = 0
    
    original_acquire = resource_manager.acquire
    original_release = resource_manager.release
    
    def tracked_acquire(*args, **kwargs):
        result = original_acquire(*args, **kwargs)
        resource_tracker.active_resources += 1
        return result
        
    def tracked_release(*args, **kwargs):
        result = original_release(*args, **kwargs)
        resource_tracker.active_resources -= 1
        return result
    
    mocker.patch.object(resource_manager, 'acquire', tracked_acquire)
    mocker.patch.object(resource_manager, 'release', tracked_release)
    
    # Execute function that should acquire and release resources
    function_under_test()
    
    # Verify all resources were released
    assert resource_tracker.active_resources == 0
```

### 4. Exception Propagation Tests
Verify exception handling and propagation:

```python
def test_exception_handling_and_wrapping(mocker):
    """Verify component wraps and transforms exceptions appropriately."""
    
    # Mock dependency to raise exception
    mock_dependency = mocker.patch('module.dependency')
    mock_dependency.side_effect = ValueError("Original error")
    
    # Test exception wrapping
    with pytest.raises(ComponentError) as excinfo:
        component_under_test()
        
    # Verify exception was wrapped properly
    assert "Component error" in str(excinfo.value)
    assert "Original error" in str(excinfo.value.__cause__)
```

## TEST HARDENING TECHNIQUES

### 1. Defensive Assertions
Write assertions that provide maximum debugging information on failure:

```python
def test_with_defensive_assertions():
    """Use assertions that provide clear failure information."""
    result = complex_function()
    
    # Instead of just:
    # assert result["status"] == "success"
    
    # Use defensive assertion with clear error message:
    assert result.get("status") == "success", \
        f"Expected status 'success' but got '{result.get('status')}'. Full result: {result}"
```

### 2. Time-based Test Hardening
For functions with timing dependencies:

```python
def test_time_sensitive_operation(mocker):
    """Test time-dependent behavior without actual waiting."""
    # Mock time.time() to return controlled sequence
    mock_time = mocker.patch('time.time')
    time_sequence = [1000.0, 1030.0, 1060.0]  # 30 second increments
    mock_time.side_effect = time_sequence
    
    # Mock sleep to do nothing
    mocker.patch('time.sleep')
    
    # Test time-sensitive function
    result = retry_operation_with_backoff()
    
    # Verify correct number of retries occurred
    assert mock_time.call_count == len(time_sequence)
```

### 3. Property-based Testing
For functions where many input variations should be tested:

```python
from hypothesis import given, strategies as st

@given(
    value=st.integers(-1000, 1000),
    modifier=st.floats(0.1, 10.0)
)
def test_property_calculation(value, modifier):
    """Property-based test to verify calculation works for many inputs."""
    result = calculate_property(value, modifier)
    
    # Properties that should hold regardless of input values
    if value > 0:
        assert result > 0
    
    assert calculate_property(value, modifier * 2) >= result
```

## QUALITY GATES

Your tests must address these critical aspects:

1. **Correctness**: Does the component produce correct results?
2. **Robustness**: Does it handle edge cases and errors gracefully?
3. **Interface Compliance**: Does it adhere to its documented interface?
4. **Performance Characteristics**: Does it perform within acceptable parameters?
5. **Resource Management**: Does it acquire and release resources properly?

## EXAMPLE: COMPLETE TEST IMPLEMENTATION

```python
"""
Tests for the UserAuthenticator class.

This suite verifies:
1. User authentication with various credential types
2. Security policy enforcement
3. Rate limiting behavior
4. Audit logging
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from security.authentication import UserAuthenticator
from security.exceptions import AuthenticationError, RateLimitExceeded
from security.models import User, AuthenticationAttempt

class TestUserAuthenticator:
    """Test suite for UserAuthenticator."""
    
    @pytest.fixture
    def auth_service(self):
        """Standard authenticator with mocked dependencies."""
        user_repo = MagicMock()
        user_repo.get_by_username.return_value = User(
            id=1,
            username="testuser",
            password_hash="$argon2id$v=19$m=65536,t=3,p=4$salt$hash",
            is_active=True,
            failed_attempts=0
        )
        
        password_verifier = MagicMock()
        password_verifier.verify.return_value = True
        
        attempt_repo = MagicMock()
        
        audit_logger = MagicMock()
        
        return UserAuthenticator(
            user_repository=user_repo,
            password_verifier=password_verifier,
            attempt_repository=attempt_repo,
            audit_logger=audit_logger
        )
    
    class TestSuccessfulAuthentication:
        """Tests for successful authentication paths."""
        
        def test_valid_credentials(self, auth_service):
            """
            GIVEN valid username and password
            WHEN authenticate is called
            THEN it should return a user object and authentication token
            """
            result = auth_service.authenticate(
                username="testuser",
                password="correct_password"
            )
            
            assert result["success"] is True
            assert result["user"].id == 1
            assert "token" in result
            assert len(result["token"]) > 20
            
            # Verify dependencies were called correctly
            auth_service._user_repository.get_by_username.assert_called_once_with("testuser")
            auth_service._password_verifier.verify.assert_called_once()
            auth_service._audit_logger.log_success.assert_called_once()
    
    class TestFailedAuthentication:
        """Tests for authentication failure scenarios."""
        
        def test_nonexistent_user(self, auth_service):
            """
            GIVEN username that doesn't exist
            WHEN authenticate is called
            THEN it should raise AuthenticationError with appropriate message
            """
            # Configure mock to return None (user not found)
            auth_service._user_repository.get_by_username.return_value = None
            
            with pytest.raises(AuthenticationError) as excinfo:
                auth_service.authenticate(
                    username="nonexistent",
                    password="any_password"
                )
                
            assert "Invalid username or password" in str(excinfo.value)
            auth_service._audit_logger.log_failure.assert_called_once()
        
        def test_incorrect_password(self, auth_service):
            """
            GIVEN valid username but incorrect password
            WHEN authenticate is called
            THEN it should raise AuthenticationError and increment failure count
            """
            # Configure password verifier to return False
            auth_service._password_verifier.verify.return_value = False
            
            with pytest.raises(AuthenticationError) as excinfo:
                auth_service.authenticate(
                    username="testuser",
                    password="wrong_password"
                )
                
            assert "Invalid username or password" in str(excinfo.value)
            
            # Verify failed attempt was recorded
            auth_service._attempt_repository.record_attempt.assert_called_once()
            auth_service._audit_logger.log_failure.assert_called_once()
    
    class TestSecurityPolicies:
        """Tests for security policy enforcement."""
        
        def test_inactive_account(self, auth_service):
            """
            GIVEN inactive user account
            WHEN authenticate is called
            THEN it should raise AuthenticationError with account inactive message
            """
            # Configure user to be inactive
            user = auth_service._user_repository.get_by_username.return_value
            user.is_active = False
            
            with pytest.raises(AuthenticationError) as excinfo:
                auth_service.authenticate(
                    username="testuser",
                    password="correct_password"
                )
                
            assert "Account is inactive" in str(excinfo.value)
            auth_service._password_verifier.verify.assert_not_called()
        
        def test_rate_limiting(self, auth_service):
            """
            GIVEN user who has exceeded failed attempt threshold
            WHEN authenticate is called
            THEN it should raise RateLimitExceeded
            """
            # Configure user with too many failed attempts
            user = auth_service._user_repository.get_by_username.return_value
            user.failed_attempts = 5  # Assuming threshold is lower than this
            
            # Configure last attempt within lockout period
            last_attempt = AuthenticationAttempt(
                user_id=1,
                timestamp=datetime.now() - timedelta(minutes=5),
                success=False
            )
            auth_service._attempt_repository.get_last_attempt.return_value = last_attempt
            
            with pytest.raises(RateLimitExceeded) as excinfo:
                auth_service.authenticate(
                    username="testuser",
                    password="any_password"
                )
                
            assert "Too many failed attempts" in str(excinfo.value)
            assert "try again after" in str(excinfo.value)
            auth_service._audit_logger.log_rate_limit.assert_called_once()
    
    class TestEdgeCases:
        """Tests for edge cases and unusual inputs."""
        
        @pytest.mark.parametrize("username,password", [
            (None, "password"),
            ("username", None),
            ("", "password"),
            ("username", ""),
            (" ", "password")
        ])
        def test_invalid_inputs(self, auth_service, username, password):
            """
            GIVEN invalid username or password format
            WHEN authenticate is called
            THEN it should raise ValueError with clear message
            """
            with pytest.raises(ValueError) as excinfo:
                auth_service.authenticate(username=username, password=password)
                
            assert "Invalid credentials format" in str(excinfo.value)
```

This structured approach produces test suites that become living documentation of expected component behavior and serve as an early warning system for defects and regressions.