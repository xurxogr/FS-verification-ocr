# Contributing to FS-Verification-OCR

Thank you for your interest in contributing to FS-Verification-OCR! This document provides guidelines for contributing to the project.

## Ways to Contribute

- **Report Bugs**: Open an issue describing the problem, steps to reproduce, and your environment
- **Suggest Features**: Open an issue describing the feature and its use case
- **Submit Pull Requests**: Fix bugs, add features, or improve documentation
- **Improve Documentation**: Help make the docs clearer and more comprehensive

## Reporting Bugs

When reporting bugs, please include:

1. **Environment Information**:
   - Python version (`python --version`)
   - Operating system
   - Installation method (pip, Docker, etc.)
   - Relevant package versions

2. **Steps to Reproduce**:
   - Exact commands or code that trigger the issue
   - Input files (if applicable and not sensitive)
   - Expected vs actual behavior

3. **Error Output**:
   - Full error messages and stack traces
   - Log output (use `VOCR_LOGGING__LOG_LEVEL=DEBUG` for verbose logs)

## Submitting Pull Requests

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/xurxogr/FS-verification-ocr.git
   cd FS-verification-ocr
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e .[dev]
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Code Quality Standards

This project follows strict code quality guidelines:

- **Linting**: Code must pass `ruff check`
- **Type Checking**: Code must pass `mypy` type checks
- **Formatting**: Code is formatted with `ruff format`
- **Testing**: Add tests for new features and bug fixes
- **Coverage**: Maintain 100% test coverage

### Running Quality Checks

```bash
# Run linter
ruff check verification_ocr/

# Run type checker
mypy verification_ocr/

# Run all pre-commit hooks
pre-commit run --all-files

# Run tests
pytest

# Run tests with coverage
pytest --cov=verification_ocr --cov-report=html
```

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Ensure all checks pass**:
   ```bash
   ruff check verification_ocr/
   mypy verification_ocr/
   pytest
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**:
   - Provide a clear description of the changes
   - Reference any related issues
   - Ensure CI checks pass

### Commit Message Guidelines

- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Keep the first line under 72 characters
- Reference issue numbers when applicable

Examples:
```
Add support for custom resolution templates
Fix OCR detection for low-contrast screenshots
Update Docker configuration for production deployment
```

## Code Style Guidelines

- **Type Hints**: All functions must have type hints
- **Pydantic Models**: Use Pydantic for data validation and settings
- **Error Handling**: Raise specific exceptions with clear error messages
- **Async/Await**: Use `async`/`await` for I/O operations in API code
- **Code Style**: Follow existing patterns in the codebase

## Testing Guidelines

- **Test Coverage**: Maintain 100% test coverage
- **Test Organization**: Use test classes to group related tests
- **Fixtures**: Use pytest fixtures for common test setup
- **Async Tests**: Mark async tests with `@pytest.mark.asyncio`
- **Mocking**: Use `unittest.mock` or `pytest-mock` for external dependencies

Example test structure:
```python
import pytest

class TestVerificationService:
    """Test suite for verification service functionality."""

    def test_verify_success(self, service, sample_image):
        """Test successful verification from screenshots."""
        result = service.verify(sample_image, sample_image)
        assert result.success is True

    def test_verify_invalid_image(self, service):
        """Test service handles invalid image gracefully."""
        result = service.verify(b"invalid", b"invalid")
        assert result.success is False
```

## License

By contributing to FS-Verification-OCR, you agree that your contributions will be licensed under the MIT License.
