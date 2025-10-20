# Testing sphinx-exercise Internationalization Support

This branch (`tst-sphinx-exercise`) is configured to test the internationalization (i18n) support for sphinx-exercise from PR #75.

## Changes Made

The environment files have been updated to install sphinx-exercise directly from the TeachBooks fork that includes i18n support:

- **environment.yml**: Updated to use `git+https://github.com/TeachBooks/sphinx-exercise.git@main`
- **environment-cn.yml**: Updated to use `git+https://github.com/TeachBooks/sphinx-exercise.git@main`

## Setup Instructions

1. **Recreate the conda environment** to install the development version:
   ```bash
   conda env remove -n quantecon
   conda env create -f environment.yml
   # Or for China mirror:
   # conda env create -f environment-cn.yml
   ```

2. **Activate the environment**:
   ```bash
   conda activate quantecon
   ```

3. **Verify the installation**:
   ```bash
   python -c "import sphinx_exercise; print(sphinx_exercise.__version__)"
   ```

## Testing i18n Support

The configuration in `lectures/_config.yml` already includes:
- `language: zh_CN` - This sets the Sphinx language to Chinese
- `sphinx_exercise` extension is loaded

With the PR #75 changes, exercise and solution labels should now automatically translate to Chinese when building the documentation.

### Build the documentation

```bash
cd lectures
jupyter-book build .
```

### What to Check

Look for translated labels in the built HTML:
- **Exercise** should appear as the Chinese equivalent (练习)
- **Solution to** should appear as the Chinese equivalent (解答)

You can inspect the HTML output in `lectures/_build/html/` or check specific lecture files that contain exercises.

## PR Reference

- **Pull Request**: https://github.com/executablebooks/sphinx-exercise/pull/75
- **Issue**: https://github.com/executablebooks/sphinx-exercise/issues/73
- **Branch**: TeachBooks:main

## Features Added by PR #75

1. Translation infrastructure using Sphinx's i18n system
2. Support for multiple languages including Chinese (zh_CN)
3. Automatic translation of "Exercise" and "Solution to" labels
4. Translation files in `.po` format for various languages

## Reverting Changes

To revert back to the stable version:

```bash
git checkout main
conda env remove -n quantecon
conda env create -f environment.yml
```
