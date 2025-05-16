# Recruitment 2025 Data Biz Hackathon

## Setup

### Installing Rye

This project uses Rye as a Python package manager. To install Rye, follow these instructions:

#### macOS/Linux

```bash
curl -sSf https://rye-up.com/get | bash
```

#### Windows

```bash
irm https://rye-up.com/get.ps1 | iex
```

For more information about Rye, visit [the official documentation](https://rye-up.com).

### Project Setup

1. Clone this repository
2. Navigate to the project directory
3. Run `rye sync` to install all dependencies
4. Start Jupyter Notebook with `rye run jupyter notebook`

## Project Structure

- `conf/` - Configuration files
- `data/` - Raw and processed data files
- `notebooks/` - Jupyter notebooks for exploration and analysis
- `src/` - Source code
  - `data_preparation/` - Data preparation scripts
  - `feature_engineering/` - Feature engineering modules
  - `evaluation/` - Model evaluation
  - `modelling/` - Model implementation
- `tests/` - Test files