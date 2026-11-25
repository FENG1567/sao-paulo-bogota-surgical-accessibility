# Portable reproduction environment

The figure and table entry points were validated with Python 3.12.3. Install the pinned Python packages from this directory with:

```sh
python -m pip install -r requirements-lock.txt
```

The table renderer uses Python's standard library. The figure renderer uses the supplied layout wrappers and CSV data; it does not estimate the study results again. Arial regular and bold font files are not distributed. Pass their directory with `--font-dir`; Windows checks the standard `C:/Windows/Fonts` directory automatically. No font is downloaded. Runs stop with a setup message when Arial is unavailable.

The extracted raw-pipeline source files need additional inputs and software beyond this lock. This package does not claim a full raw-data rerun.
