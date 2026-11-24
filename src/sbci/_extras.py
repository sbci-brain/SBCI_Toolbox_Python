"""How to install an optional extra, said the same way in every message that needs one.

The package is installed from its repository on GitHub, not from PyPI, which holds no
``sbci`` of ours: asked for the package by its name alone, pip finds nothing there, or
would install another project's package of that name. Every error that asks for an extra
takes its command from :func:`install_hint`.
"""

from __future__ import annotations

#: Where the package is installed from.
REPOSITORY = "https://github.com/sbci-brain/SBCI_Toolbox_Python.git"


def install_hint(extra: str, indent: str = "    ") -> str:
    """The commands that install ``extra``: from the repository, or in a checkout of it.

    Examples
    --------
    >>> print(install_hint("plotting"))
        pip install "sbci[plotting] @ git+https://github.com/sbci-brain/SBCI_Toolbox_Python.git"
        or, in a checkout of the repository: pip install -e ".[plotting]"
    """
    return (
        f'{indent}pip install "sbci[{extra}] @ git+{REPOSITORY}"\n'
        f'{indent}or, in a checkout of the repository: pip install -e ".[{extra}]"'
    )
