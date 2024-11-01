# Probabilistic Turing Machines

## Installation
I like [uv](https://docs.astral.sh/uv/) for package management and never needing to activate virtual envs.

```bash
uv sync # install packages in pyproject.toml to .venv
uv run gps.py # run gps.py inside virtual-env
```

Alternatively, look at [pyproject.toml](./pyproject.toml) and install the packages however you like.

## Contents

Each of these implementations has some kind of terminal-based interactive execution of a TM

- [single_tape.py](./single_tape.py) Classical Turing machines with 1 tape
    - Design guided by aim to implement Alan Turing's UTM from [On Computable Numbers, with an Application to the Entscheidungsproblem](https://www.cs.virginia.edu/~robins/Turing_Paper_1936.pdf)
    - Has exotic shorthands for certain kinds of transitions that appear a lot.
- [multi_tape.py](./multi_tape.py) Classical multi-tape Turing machines
    - Main purpose is to implement the pseudo-UTM from GPS
- [gps.py](./gps.py) Probabilistic Turing machines
    - Implements the direct-simulation (see GPS, Appendix I) of a smooth Turing machine.
    - The ANSI-code based highlighting may not be supported/compatible with your terminal and colour scheme.

## References
- GPS = [Clift, Murfet, Wallbridge (2022) - Geometry of Program Synthesis](https://arxiv.org/pdf/2103.16080)