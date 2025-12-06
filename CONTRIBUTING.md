# Contributing Guidelines

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Code Style

- Follow PEP 8 Python style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused on a single responsibility
- Add comments for complex logic

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update relevant documentation in `docs/` directory
- Keep code comments clear and concise

## Testing

Before submitting changes:

1. **Test locally**: Run your changes locally to ensure they work
2. **Debug mode**: Use debug mode to quickly verify functionality:
   ```bash
   python trainer/train.py experiments/exp2_debug.yaml
   ```
3. **Check logs**: Review log files for any errors or warnings

## Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**: Follow code style guidelines
4. **Test your changes**: Ensure everything works correctly
5. **Update documentation**: If needed, update relevant docs
6. **Commit changes**: Use clear, descriptive commit messages
7. **Push to your fork**: `git push origin feature/your-feature-name`
8. **Create Pull Request**: Provide a clear description of changes

## Commit Message Format

Use clear, descriptive commit messages:

```
Add feature: Implement SE-ResNet-50 model
Fix bug: Resolve division by zero in SE block initialization
Update docs: Add getting started guide
Refactor: Improve configuration loading logic
```

## Adding New Features

### Adding a New Model Architecture

1. Create model file in `models/` directory
2. Implement model class with factory function
3. Add configuration file in `configs/`
4. Update `trainer/train.py` to support new model
5. Add documentation

### Adding New Training Features

1. Add feature to `trainer/train.py` or appropriate module
2. Add configuration options to config loader
3. Update example configs
4. Document the feature

### Adding New Metrics

1. Add metric calculation to `trainer/metrics.py`
2. Integrate into training/evaluation pipeline
3. Update documentation

## Reporting Issues

When reporting issues, please include:

- **Description**: Clear description of the issue
- **Steps to reproduce**: How to reproduce the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: Python version, OS, dependencies
- **Error messages**: Full error traceback if applicable

## Code Review

All contributions will be reviewed for:

- Code quality and style
- Functionality and correctness
- Documentation completeness
- Test coverage
- Performance implications

## Questions?

If you have questions about contributing, please:

1. Check existing documentation
2. Review example code in the repository
3. Open an issue for discussion

Thank you for contributing!

