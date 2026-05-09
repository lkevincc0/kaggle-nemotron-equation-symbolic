# Equation Symbolic Solver

Solver for Alice's Wonderland `equation_symbolic` puzzles in the [NVIDIA NeMoTron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview).

## Layout

```text
kaggle-nemotron-equation-symbolic/
├── src/
│   ├── solver_eq_symbolic.py
│   └── solve_all_symbolic.py
├── data/
│   └──train.csv
├── docs/
│   └── index.html
├── main.py
├── pyproject.toml
└── rust/
    └── alice_sovler_helper/
        └── src/lib.rs
```

## Install

```bash
uv sync
uv run maturin develop --release
```

## Test

```bash
uv run python main.py
```

## Generate Solver Results

```bash
uv run python src/solve_all_symbolic.py
```

Output:

```bash
data/solver_results.parquet
```

Quick test:

```bash
uv run python src/solve_all_symbolic.py --limit 3 --workers 1
```

## Visualization

```bash
uv run python -m http.server 8000
```

Open:

```text
http://localhost:8000/docs/
```

## Use

```python
from src.solver_eq_symbolic import AliceEquationSolver

solver = AliceEquationSolver(prompt_text, answer_hint=gold_answer)
answer, details = solver.solve()
```
