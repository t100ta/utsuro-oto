"""Convenience launcher: ``python main.py`` behaves like ``python -m thereminvox.main``."""

from runpy import run_module

if __name__ == "__main__":
    run_module("thereminvox.main", run_name="__main__")
