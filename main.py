"""Convenience launcher: ``python main.py`` behaves like ``python -m utsuro_oto.main``."""

from runpy import run_module

if __name__ == "__main__":
    run_module("utsuro_oto.main", run_name="__main__")
