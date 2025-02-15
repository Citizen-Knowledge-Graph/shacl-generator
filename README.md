# Legal Text to SHACL Shape Mapper

This tool helps translate legal texts describing social benefit eligibility requirements into formal SHACL shapes. It provides a user-friendly interface for mapping legal requirements to SHACL constraints.

## Features

- Upload and view legal text documents
- Upload and preview example SHACL shapes
- Interactive mapping interface for creating SHACL constraints
- Support for common SHACL constraint types
- Preview generated SHACL shapes

## Setup

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
# if only available at ~/.local/bin/poetry, link it globally: echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
```

```bash
cd shacl_generator
```

2. Install dependencies:
```bash
poetry install
```

3. Run the application:
```bash
export OPENAI_API_KEY="..."
poetry run streamlit run app.py
```

## Development

To add new dependencies:
```bash
poetry add package-name
```

To activate the virtual environment:
```bash
poetry shell
```

### Code Quality Tools

The project uses several tools to maintain code quality:

- **Formatting**: Run `poetry run black .` to format code
- **Import Sorting**: Run `poetry run isort .` to sort imports
- **Linting**: Run `poetry run flake8` to check for code style issues
- **Type Checking**: Run `poetry run mypy .` to check types
- **Testing**: Run `poetry run pytest` to run tests

## Usage

1. Upload your legal text document (txt or pdf) using the sidebar
2. Optionally upload an example SHACL shape for reference
3. Use the mapping interface to:
   - Select relevant text from the legal document
   - Choose appropriate SHACL constraint types
   - Define constraint values
4. Generate and preview the resulting SHACL shape

## Requirements

- Python 3.9+ (excluding 3.9.7)
- Poetry for dependency management 