"""
Copy each example script into its cell in the group notebook, so the
notebooks show the code they run. The scripts are the source: edit a
script, then run

    python sync_notebooks.py           # rewrite the scripts' notebook cells
    python sync_notebooks.py --check   # only report stale notebooks (exit 1); check_all.py runs this

A script's cell is found by its first line, `__file__ = "NN_name.py"` (or
`%run NN_name.py`, to add a new script's cell). That line is the one addition: `python` and `%run` set `__file__`, a notebook
cell does not, and the scripts use it to find their output directory. The
one change: `parser.parse_args()` becomes `parser.parse_args([])`, since a
notebook has no command line.
"""

import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
SCRIPT_CELL = re.compile(r'^(?:%run |__file__ = ")(\d\d_\w+\.py)')


def cell_source(script):
    code = re.sub(r"parser\.parse_args\(\)(.*)$", r"parser.parse_args([])\1  # no command line here: options go in the []",
                  script.read_text(), flags=re.M)
    return f'__file__ = "{script.name}"  # python and %run set this; a notebook cell does not\n' + code.rstrip("\n")


def main():
    check = "--check" in sys.argv[1:]
    stale = []
    for path in sorted(ROOT.glob("[0-9][0-9]_*/*.ipynb")):
        book = nbformat.read(path, 4)
        changed = False
        for cell in book.cells:
            match = cell.cell_type == "code" and SCRIPT_CELL.match(cell.source)
            if match and cell.source != (source := cell_source(path.parent / match.group(1))):
                cell.source, changed = source, True
        if changed:
            stale.append(path.relative_to(ROOT).as_posix())
            if not check:
                nbformat.write(book, path)
    if stale:
        print(("out of date (run python sync_notebooks.py): " if check else "updated: ") + ", ".join(stale))
    sys.exit(1 if check and stale else 0)


if __name__ == "__main__":
    main()
