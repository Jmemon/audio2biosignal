# Unit Testing

## High-Level Objective
Create a unit test for a function, method, or class.

## Mid-Level Objectives
Write a test that covers all edge cases and error conditions, and representatively tests normal cases.
Make sure exceptions are triggered and handled correctly.
Make sure normal/common cases are tested.

## Implementation Notes
Use pytest.
Unit tests go in `tests/unit/`. The structure of this directory should mirror the structure of `src/`.
Use conftest.py files to define fixtures and other shared resources where appropriate. Make sure to place them in appropriate directories.
Use one test file per source file. All tests for classes/methods/functions in a source file should go in the same test file.
Group related tests into classes.

## Context
### Beginning Context
<FILE_CONTAINING_TARGET>
<FILES_USING_TARGET>

### Ending Context
<FILE_CONTAINING_TARGET>
<FILES_USING_TARGET>
<TEST_FILE>

## Pytest Directory Structure and Unit Testing Best Practices

This guide provides a comprehensive overview of pytest project organization and unit testing principles. The information is structured to clearly define patterns, relationships, and hierarchies within test code organization.

### Pytest Directory Structure

#### Standard Pattern

```
project_root/
├── src/  (or package_name/)
│   └── ... (source code)
└── tests/
    ├── __init__.py
    ├── conftest.py  # Global fixtures
    ├── test_module1.py
    ├── test_module2.py
    └── subdir/
        ├── __init__.py
        ├── conftest.py  # Directory-specific fixtures
        └── test_something.py
```

#### Key Structural Components

1. **Root Test Directory**: Named `tests/` at the project root level.
2. **Test Discovery Pattern**: Files named `test_*.py` contain test functions.
3. **Test Function Pattern**: Functions prefixed with `test_` are discovered as tests.
4. **Package Structure**: `__init__.py` files make directories importable packages.
5. **Fixture Files**: `conftest.py` files contain shared fixtures accessible to tests in that directory and subdirectories.

#### Unit Tests Organization

Unit tests follow a structure that maps to the source code organization:

```
project_root/
├── src/
│   ├── module1/
│   │   ├── __init__.py
│   │   ├── component1.py
│   │   └── component2.py
│   └── module2/
│       └── component3.py
└── tests/
    ├── unit/  # Contains all unit tests
    │   ├── __init__.py
    │   ├── conftest.py  # Unit test fixtures
    │   ├── module1/
    │   │   ├── __init__.py
    │   │   ├── test_component1.py  # Tests for component1.py
    │   │   └── test_component2.py
    │   └── module2/
    │       ├── __init__.py
    │       └── test_component3.py
    └── conftest.py  # Global fixtures
```

#### Test Type Segregation

For projects with multiple test types:

```
tests/
├── unit/          # Tests for individual components in isolation
├── integration/   # Tests for component interactions
├── functional/    # End-to-end tests of application features
└── conftest.py    # Global fixtures
```

### Unit Testing Rules and Principles

#### General Principles

1. **Single Responsibility**: Each test verifies exactly one behavior or functionality.
2. **AAA Pattern**:
   - **Arrange**: Set up test preconditions
   - **Act**: Execute the code under test
   - **Assert**: Verify expected outcomes
3. **Naming Convention**: Test names should describe what's being tested and expected outcome
   - Format: `test_[function]_[scenario]_[expected_result]`
   - Example: `test_withdraw_with_insufficient_funds_raises_error`
4. **Independence**: Tests must not depend on each other or execution order
5. **Coverage Spectrum**: Tests should cover both normal operation and edge cases
6. **Exception Testing**: Verify error conditions raise appropriate exceptions
7. **Fixture Usage**: Common setup code should be extracted to fixtures
8. **Logic Avoidance**: Tests should avoid conditional logic

#### Function Testing Rules

1. **Input Validation Testing**:
   - Test with valid inputs within normal range
   - Test with boundary values at the edges of valid ranges
   - Test with invalid inputs outside acceptable ranges
   - Test with empty/None values where applicable

2. **Return Value Verification**:
   - Verify exact expected return values
   - Verify return type matches expectations
   - For functions returning collections, verify contents regardless of order when appropriate

3. **Side Effect Testing**:
   - Verify any state changes that should occur
   - Use mocks to verify interactions with other components

#### Method Testing Rules

1. **State Change Verification**:
   - Verify object state before and after method call
   - Test state transitions through multiple operations
   - Verify class invariants are maintained

2. **Interaction Testing**:
   - Verify methods interact correctly with other methods
   - Mock internal dependencies when testing specific behavior

3. **Object Configuration Testing**:
   - Test with different instance configurations
   - Test behavior at state boundaries

#### Class Testing Rules

1. **Constructor Testing**:
   - Test initialization with various parameter combinations
   - Test default parameter values
   - Test with invalid initialization parameters

2. **Interface Coverage**:
   - Test all public methods 
   - Test method interactions within the class
   - Verify class invariants

3. **Inheritance Behavior**:
   - Test that subclasses maintain parent class contracts
   - Test overridden methods maintain expected behavior
   - Test interactions between overridden and parent methods

4. **Lifecycle Testing**:
   - Test object creation
   - Test state transitions throughout object lifetime
   - Test cleanup/destruction if implemented

#### Mocking and Fixture Patterns

1. **External Dependency Mocking**:
   ```python
   def test_function_uses_database(mocker):
       # Mock external dependencies
       mock_db = mocker.patch('module.database_client')
       mock_db.query.return_value = [{'id': 1, 'name': 'test'}]
       
       # Execute function that uses the database
       result = function_under_test()
       
       # Verify database was called correctly
       mock_db.query.assert_called_once_with('SELECT * FROM table')
       assert result == [{'id': 1, 'name': 'test'}]
   ```

2. **Fixture Definition Pattern**:
   ```python
   @pytest.fixture
   def user_fixture():
       """Create a test user object with predefined state."""
       user = User(
           id=1,
           username="testuser",
           email="test@example.com"
       )
       return user
       
   def test_user_authentication(user_fixture):
       """Test authentication with the user fixture."""
       assert user_fixture.authenticate("correct_password") is True
   ```

### Concrete Examples

#### Function Test Example

```python
# Function to test
def calculate_discount(price, discount_percentage):
    """Calculate discounted price."""
    if not (0 <= discount_percentage <= 100):
        raise ValueError("Discount must be between 0 and 100")
    return price * (1 - discount_percentage / 100)

# Tests
def test_calculate_discount_with_valid_values():
    # Arrange
    price = 100.0
    discount = 20.0
    
    # Act
    result = calculate_discount(price, discount)
    
    # Assert
    assert result == 80.0

def test_calculate_discount_with_zero_discount():
    assert calculate_discount(100.0, 0) == 100.0

def test_calculate_discount_with_full_discount():
    assert calculate_discount(100.0, 100) == 0.0

def test_calculate_discount_with_invalid_discount():
    with pytest.raises(ValueError):
        calculate_discount(100.0, 101)
```

#### Method Test Example

```python
# Class with method to test
class ShoppingCart:
    def __init__(self):
        self.items = {}
    
    def add_item(self, item_id, quantity=1):
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if item_id in self.items:
            self.items[item_id] += quantity
        else:
            self.items[item_id] = quantity
    
    def item_count(self):
        return sum(self.items.values())

# Tests
class TestShoppingCart:
    def test_add_new_item(self):
        # Arrange
        cart = ShoppingCart()
        
        # Act
        cart.add_item("PROD1", 1)
        
        # Assert
        assert cart.items == {"PROD1": 1}
        assert cart.item_count() == 1
    
    def test_add_existing_item(self):
        # Arrange
        cart = ShoppingCart()
        cart.add_item("PROD1", 1)
        
        # Act
        cart.add_item("PROD1", 2)
        
        # Assert
        assert cart.items == {"PROD1": 3}
        assert cart.item_count() == 3
    
    def test_add_item_with_invalid_quantity(self):
        # Arrange
        cart = ShoppingCart()
        
        # Act/Assert
        with pytest.raises(ValueError):
            cart.add_item("PROD1", 0)
```

#### Class Test Example

```python
# Class to test
class BankAccount:
    def __init__(self, account_number, balance=0):
        self.account_number = account_number
        self.balance = balance
        self.is_active = True
    
    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount
    
    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount
    
    def close(self):
        self.is_active = False

# Tests
class TestBankAccountCreation:
    def test_init_with_default_balance(self):
        account = BankAccount("12345")
        assert account.account_number == "12345"
        assert account.balance == 0
        assert account.is_active is True
    
    def test_init_with_custom_balance(self):
        account = BankAccount("12345", 100)
        assert account.balance == 100

class TestBankAccountOperations:
    @pytest.fixture
    def account(self):
        return BankAccount("12345", 100)
    
    def test_deposit_increases_balance(self, account):
        account.deposit(50)
        assert account.balance == 150
    
    def test_withdraw_decreases_balance(self, account):
        account.withdraw(50)
        assert account.balance == 50
    
    def test_withdraw_insufficient_funds(self, account):
        with pytest.raises(ValueError, match="Insufficient funds"):
            account.withdraw(200)
    
    def test_close_deactivates_account(self, account):
        account.close()
        assert account.is_active is False
```

### Dependency Mocking Example

```python
# Class with external dependency
class UserService:
    def __init__(self, db_client):
        self.db_client = db_client
    
    def get_user_by_id(self, user_id):
        user_data = self.db_client.query("users", {"id": user_id})
        if not user_data:
            return None
        return User(**user_data[0])

# Test with mocking
def test_get_user_by_id_existing_user(mocker):
    # Arrange
    mock_db = mocker.Mock()
    mock_db.query.return_value = [{"id": 1, "name": "Test User", "email": "test@example.com"}]
    
    service = UserService(mock_db)
    
    # Act
    user = service.get_user_by_id(1)
    
    # Assert
    mock_db.query.assert_called_once_with("users", {"id": 1})
    assert user.id == 1
    assert user.name == "Test User"
    assert user.email == "test@example.com"

def test_get_user_by_id_nonexistent_user(mocker):
    # Arrange
    mock_db = mocker.Mock()
    mock_db.query.return_value = []
    
    service = UserService(mock_db)
    
    # Act
    user = service.get_user_by_id(999)
    
    # Assert
    mock_db.query.assert_called_once_with("users", {"id": 999})
    assert user is None
```

This structured approach to testing ensures comprehensive coverage of code behavior while maintaining organization that scales with project complexity.







## General Principles (Apply to All)
- Test One Thing Per Test: Each test should verify a single behavior or functionality.
- Follow the AAA Pattern:
    - Arrange: Set up the test conditions
    - Act: Call the method/function being tested
    - Assert: Verify the expected outcome
- Use Descriptive Test Names: Name should clearly describe what's being tested and the expected outcome.
- Keep Tests Independent: Tests should not depend on each other or on execution order.
- Test Both Happy Paths and Edge Cases: Cover normal operation and potential failure modes.
- Test Exceptions: Verify that errors are raised when expected.
- Use Fixtures for Common Setup: Use pytest fixtures for reusable test setup.
- Avoid Logic in Tests: Tests should be straightforward without complex conditionals.

## Testing Functions
### Input Validation
- Test with valid inputs
- Test with boundary values (min/max acceptable values)
- Test with invalid inputs
- Test with empty/None values where applicable

### Return Values
- Verify correct return values for various inputs
- Check return type matches expectations
- For functions that can return multiple types, test each path

### Side Effects
- If the function modifies global state, test that the modification occurs correctly
- If the function calls other functions, use mocks to verify those calls

## Testing Methods
### State Changes
- Test how the method changes object state
- Verify state remains correct after multiple operations

### Interactions
- Test how the method interacts with other methods
- Use mocks for dependencies on other methods

### Instance-Specific Behavior
- Test with different instance configurations
- Test boundaries between different instance states

## Testing Classes
### Constructor
- Test initialization with various parameters
- Test default parameters
- Test invalid initialization

### Public Interface
- Test all public methods
- Test method interactions within the class
- Test class invariants (conditions that should always be true)

### Inheritance
- Test that subclasses maintain expected behavior
- Test overridden methods
- Test interactions with parent class methods

### Lifecycle
- Test creation, modification, and destruction (if applicable)
- Test state transitions

## Testing with Mocks and Fixtures
### Use Mocks for External Dependencies
```python
def test_function_calls_database(mocker):
    # Arrange
    mock_db = mocker.patch('module.database')
    
    # Act
    function_under_test()
    
    # Assert
    mock_db.query.assert_called_once()
```

### Use Fixtures for Setup
```python
@pytest.fixture
def populated_cart():
    cart = ShoppingCart()
    cart.add_item(Item("product1", 10.0))
    cart.add_item(Item("product2", 15.0))
    return cart
    
def test_cart_total(populated_cart):
    assert populated_cart.total() == 25.0
```