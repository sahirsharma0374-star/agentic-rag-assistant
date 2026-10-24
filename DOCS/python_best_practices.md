Follow PEP 8 for consistent code formatting. Use meaningful variable names that describe the purpose of the variable. Keep functions small and focused on a single responsibility.

## Error Handling

Always use specific exception types rather than catching generic Exception. Use try/except blocks around code that might fail, especially I/O operations and network calls. Log errors with context information.

## Virtual Environments

Always use virtual environments to isolate project dependencies. Create a requirements.txt or pyproject.toml to document dependencies. Use pip freeze to capture exact versions for reproducibility.

## Testing

Write unit tests for all public functions. Use pytest as your testing framework. Aim for at least 80% code coverage. Mock external dependencies in tests to keep them fast and reliable.