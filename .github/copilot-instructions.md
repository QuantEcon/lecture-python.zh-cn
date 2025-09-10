# QuantEcon 中级数量经济学 (lecture-python.zh-cn)

Chinese version of QuantEcon intermediate lectures built with Jupyter Book. This repository contains Chinese translations of quantitative economics lectures designed and written by Thomas J. Sargent and John Stachurski.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

**CRITICAL**: NEVER CANCEL any build or test commands. Builds take 35-45 minutes. ALWAYS set timeouts to 60+ minutes.

### Bootstrap and Setup Environment
- **Setup conda environment (2-3 minutes):**
  - `source /usr/share/miniconda/etc/profile.d/conda.sh` (required before conda activate)
  - `conda env create -f environment.yml` -- takes 2-3 minutes
  - `conda activate quantecon`
  - **Note:** May fail in restricted networks with "Read timed out" errors. Use existing environment or retry.
- **Verify installation:**
  - `jupyter-book --version` should show "Jupyter Book : 1.0.3"
  - `python --version` should show "Python 3.12.7"

### Build the Documentation
- **Build notebooks (35-45 minutes):**
  - `jupyter-book build lectures --path-output ./ --builder=custom --custom-builder=jupyter` -- NEVER CANCEL. Set timeout to 60+ minutes.
  - Builds 80+ notebook files sequentially, ~2-3 files per minute
  - Creates `_build/jupyter/*.ipynb` files
- **Build HTML (5-10 minutes):**
  - `jupyter-book build lectures --path-output ./ --keep-going` -- Set timeout to 15+ minutes
  - Creates `_build/html/` directory with website
- **Complete build pipeline:**
  - First: notebooks, then: HTML. Total time: ~45-55 minutes

### Test Single Files (for development)
- **Build single file (fast validation):**
  - `jupyter-book build lectures/intro.md --path-output _test/` -- takes 3-4 seconds
  - `jupyter-book build lectures/linear_algebra.md --path-output _test2/` -- takes 5-6 seconds
  - Use for quick validation before full builds

## Validation

- **ALWAYS build and test your changes** before completing work.
- **Manual validation scenarios:**
  - Build a single file to test basic functionality
  - Run full notebook build to ensure all code executes properly
  - Check HTML output in `_build/html/` directory
  - Verify Chinese characters display correctly in generated pages
- **No traditional unit tests** - validation is ensuring the build completes successfully.
- **Expected warnings during builds:**
  - `WARNING: failed to reach any of the inventories` (network restrictions)
  - `WARNING: could not find bibtex key` (missing references)
  - `WARNING: Unknown directive type` (when building single files)

## Environment Options

- **Standard environment**: Use `environment.yml` (always works)
- **Chinese environment**: `environment-cn.yml` uses Chinese PyPI mirror
  - May fail in restricted networks with "No address associated with hostname"
  - Use standard environment when Chinese mirror is inaccessible

## Common Tasks

### Repository Structure
```
.
├── README.md
├── environment.yml              # Conda environment (use this)
├── environment-cn.yml           # Chinese mirror version
├── lectures/                    # Main content directory
│   ├── _config.yml             # Jupyter Book configuration
│   ├── _toc.yml                # Table of contents
│   ├── intro.md                # Introduction page
│   ├── linear_algebra.md       # Example lecture file
│   └── [80+ other .md files]   # Lecture content
├── _build/                      # Generated during build
│   ├── jupyter/                # Generated notebooks
│   └── html/                   # Generated website
└── tools/                      # Translation utilities
```

### Key Configuration Files
- `lectures/_config.yml`: Jupyter Book settings, theme, extensions
- `lectures/_toc.yml`: Book structure with 14 chapters covering tools, statistics, optimization, etc.
- `environment.yml`: Python 3.12, Jupyter Book 1.0.3, quantecon-book-theme

### GitHub Workflows
- **CI Build**: `.github/workflows/ci.yml` - builds on PRs, deploys to Netlify preview
- **Publish**: `.github/workflows/publish.yml` - publishes to GitHub Pages on tags
- **Cache**: `.github/workflows/cache.yml` - caches build artifacts

### Common File Types
- **`.md` files**: MyST Markdown with executable code blocks
- **Python files in `_static/`**: Supporting code for lectures
- **Translation tools**: `tools/translation.py`, `tools/translation_openai.py`

## Troubleshooting

- **Build failures**: Check for Python syntax errors in code blocks within `.md` files
- **Long build times**: Normal - full builds take 45+ minutes due to notebook execution
- **Network errors**: Expected in sandboxed environments, builds continue despite warnings
- **Environment setup failures**: May fail with "Read timed out" in restricted networks - retry or use existing environment
- **Missing LaTeX**: LaTeX/PDF builds not supported in basic setup (HTML builds work fine)
- **Environment activation**: Always source conda profile before activating environment

## Working with Content

- **Lecture files**: All in `lectures/` directory, numbered by chapter
- **Chinese content**: All lectures are in Chinese with embedded Python code
- **Code execution**: Code blocks execute during build and results are cached
- **Images**: Generated plots saved to `_build/html/_images/` during build
- **References**: Bibliography in `lectures/_static/quant-econ.bib`

## Build Artifacts

After successful build:
- `_build/jupyter/`: Individual notebook files (.ipynb)
- `_build/html/`: Complete website with Chinese content
- `_build/.jupyter_cache/`: Execution cache (speeds up subsequent builds)

Always ensure builds complete successfully before pushing changes.