"""Generate solver_results.parquet for equation_symbolic rows.

Default paths are relative to the open_source directory:

    uv run python data/solve_all_symbolic.py --limit 3 --workers 1

Expected input:
    data/train.csv

Default output:
    data/solver_results.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.solver_eq_symbolic import AliceEquationSolver


DEFAULT_CSV = _ROOT / "data/train.csv"
DEFAULT_OUTPUT = _ROOT / "data/solver_results.parquet"


def categorize(prompt: str, answer: str) -> str:
    if not prompt:
        return "other"
    first = prompt.split("\n")[0].lower()
    if "transformation rules" in first and "equation" in first:
        ans = str(answer).strip()
        if bool(re.fullmatch(r"[\d.\-]+", ans)) and any(c.isdigit() for c in ans):
            return "equation_numeric"
        if any(c.isdigit() for c in ans) and re.search(r"[^\d\s.\-]", ans):
            return "equation_numeric_symbol"
        return "equation_symbolic"
    return "other"


def count_op_appearances(prompt: str) -> dict[str, int]:
    ex_start = max(prompt.find("examples:"), prompt.find("Examples:"), 0)
    q_start = prompt.find("Now, determine")
    if q_start < 0:
        q_start = prompt.find("determine the result for:")
    body = prompt[ex_start:q_start] if q_start > 0 else prompt[ex_start:]
    counts: dict[str, int] = {}
    for line in body.split("\n"):
        line = line.strip()
        if not line or "example" in line.lower() or "=" not in line:
            continue
        lhs = line.split("=", 1)[0].strip()
        if len(lhs) >= 3:
            op = lhs[2]
            counts[op] = counts.get(op, 0) + 1
    return counts


def solve_row(row: pd.Series) -> dict:
    solver = AliceEquationSolver(row["prompt"], answer_hint=row["answer"])
    answer, details = solver.solve()
    op_counts = count_op_appearances(row["prompt"])
    solver_category = details.get("category") if details else None
    if solver_category is None and details:
        solver_category = details.get("type")
    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "answer": row["answer"],
        "solver_answer": answer,
        "solver_correct": answer == row["answer"],
        "solver_type": details.get("type") if details else None,
        "solver_category": solver_category,
        "conditioned_on_answer": bool(details.get("conditioned_on_answer"))
        if details
        else False,
        "solver_mode": details.get("mode") if details else None,
        "solver_ops": json.dumps(details.get("ops"), ensure_ascii=False)
        if details and "ops" in details
        else None,
        "solver_mapping": json.dumps(details.get("mapping"), ensure_ascii=False)
        if details and "mapping" in details
        else None,
        "solver_tier": details.get("tier") if details else None,
        "solver_level": details.get("level") if details else None,
        "solver_radix": details.get("solver_radix") if details else None,
        "solver_numeric_answer": details.get("numeric_answer") if details else None,
        "op_counts": json.dumps(op_counts, ensure_ascii=False),
        "min_op_count": min(op_counts.values()) if op_counts else 0,
        "num_ops": len(op_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--limit", type=int, default=0, help="cap rows for smoke tests")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["cat"] = df.apply(lambda r: categorize(r["prompt"], r["answer"]), axis=1)
    sym = df[df["cat"] == "equation_symbolic"].copy()
    if args.limit:
        sym = sym.head(args.limit)
    print(f"equation_symbolic: {len(sym)} rows  workers={args.workers}")

    rows: list[dict] = [None] * len(sym)  # type: ignore[list-item]
    sym_rows = list(sym.iterrows())

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fut_to_idx = {pool.submit(solve_row, row): i for i, (_, row) in enumerate(sym_rows)}
        for fut in as_completed(fut_to_idx):
            rows[fut_to_idx[fut]] = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(sym)} ...")

    out = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)

    total = len(out)
    correct = int(out["solver_correct"].sum()) if total else 0
    no_solution = int(out["solver_answer"].isna().sum()) if total else 0
    wrong = total - correct - no_solution

    print("\n=== results ===")
    print(f"total        : {total}")
    print(
        f"correct      : {correct}  ({correct / total * 100:.1f}%)"
        if total
        else "correct      : 0"
    )
    print(
        f"wrong answer : {wrong}  ({wrong / total * 100:.1f}%)"
        if total
        else "wrong answer : 0"
    )
    print(
        f"no solution  : {no_solution}  ({no_solution / total * 100:.1f}%)"
        if total
        else "no solution  : 0"
    )
    print(f"\nsaved -> {output}")


if __name__ == "__main__":
    main()
