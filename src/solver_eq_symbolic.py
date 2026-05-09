"""
Alice's Wonderland Equation Solver
======================================
Solves equation_symbolic puzzles from the Kaggle Nemotron reasoning competition.

Puzzle types
------------
Type 1 — Concatenation:  operators concatenate (fwd/rev) left & right.
Type 2 — Arithmetic:     each symbol ↔ digit (0–9),
                         operators map to ANY arithmetic op from a rich library.
          Sub-modes:      standard / alice (reverse-operand-result)

Supported operation library
---------------------------
Core:       add, sub, rsub, absdiff, mul, gcd, lcm, fdiv, rdiv, mod, rmod, min, max
Offsets:    add±1, add±2, mul±1, mul±2, absdiff±1, absdiff±2, sub±1, rsub±1
Scaled:     mul_half, mul_double, squared_diff
Bitwise:    xor, band, bor
Sign:       neg_absdiff (for sign-prefixed results)

Usage
-----
    solver = AliceEquationSolver(prompt_text)
    answer, details = solver.solve()

For harder puzzles, try:
    solver = AliceEquationSolver(prompt_text, search_level='deep')
"""

from itertools import permutations
from collections import defaultdict
from math import gcd
import string

try:
    import alice_sovler_helper as _rust_helper

    if not hasattr(_rust_helper, "arithmetic_search"):
        _rust_helper = None
        print("[solver_eq_symbolic.py] Rust module not built; using Python search")
    else:
        print("[solver_eq_symbolic.py] Using Rust accelerate module for search")
except Exception:
    _rust_helper = None
    print("[solver_eq_symbolic.py] Using Python search")


# ════════════════════ Operation library ════════════════════


def _ss(a, b):
    return a - b if a >= b else None


def _rs(a, b):
    return b - a if b >= a else None


def _fd(a, b):
    return a // b if b else None


def _rd(a, b):
    return b // a if a else None


def _mo(a, b):
    return a % b if b else None


def _rm(a, b):
    return b % a if a else None


def _lcm(a, b):
    return a * b // gcd(a, b) if (a and b) else 0


def _nz(v):
    return v if (v is not None and v >= 0) else None


def _off(fn, delta):
    """Wrap fn to add delta to its result (returns None on negative)."""

    def g(a, b):
        v = fn(a, b)
        if v is None:
            return None
        r = v + delta
        return r if r >= 0 else None

    return g


OPERATIONS = {
    # Core
    "add": lambda a, b: a + b,
    "sub": _ss,
    "rsub": _rs,
    "absdiff": lambda a, b: abs(a - b),
    "neg_absdiff": lambda a, b: abs(a - b),  # sign-prefixed
    "mul": lambda a, b: a * b,
    "gcd": gcd,
    "lcm": _lcm,
    "fdiv": _fd,
    "rdiv": _rd,
    "mod": _mo,
    "rmod": _rm,
    "min": min,
    "max": max,
    # Offset ±1, ±2
    "add_m1": _off(lambda a, b: a + b, -1),
    "add_p1": _off(lambda a, b: a + b, 1),
    "add_m2": _off(lambda a, b: a + b, -2),
    "add_p2": _off(lambda a, b: a + b, 2),
    "mul_m1": _off(lambda a, b: a * b, -1),
    "mul_p1": _off(lambda a, b: a * b, 1),
    "mul_m2": _off(lambda a, b: a * b, -2),
    "mul_p2": _off(lambda a, b: a * b, 2),
    "absdiff_m1": _off(lambda a, b: abs(a - b), -1),
    "absdiff_p1": _off(lambda a, b: abs(a - b), 1),
    "absdiff_m2": _off(lambda a, b: abs(a - b), -2),
    "absdiff_p2": _off(lambda a, b: abs(a - b), 2),
    "sub_m1": _off(_ss, -1),
    "sub_p1": _off(_ss, 1),
    "rsub_m1": _off(_rs, -1),
    "rsub_p1": _off(_rs, 1),
    # Scaled / polynomial
    "mul_half": lambda a, b: (a * b) // 2,
    "mul_double": lambda a, b: a * b * 2,
    "sq_diff": lambda a, b: (a - b) ** 2,
    "sq_sum": lambda a, b: (a + b) ** 2,
    "mul_plus_a": lambda a, b: a * b + a,
    "mul_plus_b": lambda a, b: a * b + b,
    "mul_minus_a": _off(lambda a, b: a * b - a, 0),
    "mul_minus_b": _off(lambda a, b: a * b - b, 0),
    "a2_plus_b": lambda a, b: a * a + b,
    "a_plus_b2": lambda a, b: a + b * b,
    # Bitwise (rarely used but cheap)
    "xor": lambda a, b: a ^ b,
    "band": lambda a, b: a & b,
    "bor": lambda a, b: a | b,
    # Signed subtractions: return a signed value; the sign decides whether the
    # rhs carries a leading operator-symbol prefix. Magnitude is encoded after.
    "sub_signed": lambda a, b: a - b,
    "rsub_signed": lambda a, b: b - a,
    # Sequence operations. These are handled specially by the search/evaluator
    # because they compare symbol sequences rather than numeric magnitudes.
    "concat_fwd": None,
    "concat_rev": None,
}

# Ops whose result may be negative; sign of result must match whether the
# rhs has a leading op-symbol prefix. Magnitude (|v|) is what gets encoded.
SIGNED_OPS = {"sub_signed", "rsub_signed"}


OP_NAME_TO_IDX = {
    "add": 0,
    "sub": 1,
    "rsub": 2,
    "absdiff": 3,
    "neg_absdiff": 4,
    "mul": 5,
    "gcd": 6,
    "lcm": 7,
    "fdiv": 8,
    "rdiv": 9,
    "mod": 10,
    "rmod": 11,
    "min": 12,
    "max": 13,
    "add_m1": 14,
    "add_p1": 15,
    "mul_m1": 16,
    "mul_p1": 17,
    "absdiff_m1": 18,
    "absdiff_p1": 19,
    "sub_m1": 20,
    "sub_p1": 21,
    "rsub_m1": 22,
    "rsub_p1": 23,
    "add_m2": 24,
    "add_p2": 25,
    "mul_m2": 26,
    "mul_p2": 27,
    "absdiff_m2": 28,
    "absdiff_p2": 29,
    "mul_half": 30,
    "mul_double": 31,
    "sq_diff": 32,
    "sq_sum": 33,
    "mul_plus_a": 34,
    "mul_plus_b": 35,
    "mul_minus_a": 36,
    "mul_minus_b": 37,
    "a2_plus_b": 38,
    "a_plus_b2": 39,
    "xor": 40,
    "band": 41,
    "bor": 42,
    "sub_signed": 43,
    "rsub_signed": 44,
    "concat_fwd": 45,
    "concat_rev": 46,
}
OP_IDX_TO_NAME = {v: k for k, v in OP_NAME_TO_IDX.items()}
MODE_TO_IDX = {"standard": 0, "alice": 1, "little_endian": 2}
SEARCH_MODES = ("standard", "little_endian", "alice")


OP_PRIORITY = {
    "*": [
        "mul",
        "mul_m1",
        "mul_p1",
        "mul_m2",
        "mul_p2",
        "absdiff",
        "add",
        "gcd",
        "lcm",
        "mul_half",
        "mul_double",
        "mul_plus_a",
        "mul_plus_b",
        "mul_minus_a",
        "mul_minus_b",
        "sq_diff",
        "sq_sum",
        "concat_fwd",
        "concat_rev",
    ],
    "+": [
        "add",
        "add_m1",
        "add_p1",
        "add_m2",
        "add_p2",
        "mul",
        "absdiff",
        "gcd",
        "lcm",
        "sq_sum",
        "mul_plus_a",
        "mul_plus_b",
        "concat_fwd",
        "concat_rev",
    ],
    "-": [
        "absdiff",
        "sub",
        "rsub",
        "sub_signed",
        "rsub_signed",
        "absdiff_m1",
        "absdiff_p1",
        "absdiff_m2",
        "absdiff_p2",
        "sub_m1",
        "sub_p1",
        "rsub_m1",
        "rsub_p1",
        "mul",
        "add",
        "neg_absdiff",
        "gcd",
        "lcm",
        "sq_diff",
        "concat_fwd",
        "concat_rev",
    ],
    "/": ["fdiv", "rdiv", "mul", "add", "absdiff", "concat_fwd", "concat_rev"],
}
DEFAULT_PRIORITY = [
    "add",
    "absdiff",
    "mul",
    "sub",
    "rsub",
    "sub_signed",
    "rsub_signed",
    "add_m1",
    "add_p1",
    "mul_m1",
    "mul_p1",
    "absdiff_m1",
    "absdiff_p1",
    "gcd",
    "lcm",
    "concat_fwd",
    "concat_rev",
    "sub_m1",
    "sub_p1",
    "rsub_m1",
    "rsub_p1",
    "add_m2",
    "add_p2",
    "mul_m2",
    "mul_p2",
    "absdiff_m2",
    "absdiff_p2",
    "mul_half",
    "mul_double",
    "sq_diff",
    "sq_sum",
    "mul_plus_a",
    "mul_plus_b",
    "mul_minus_a",
    "mul_minus_b",
    "a2_plus_b",
    "a_plus_b2",
    "neg_absdiff",
    "fdiv",
    "rdiv",
    "mod",
    "rmod",
    "min",
    "max",
    "xor",
    "band",
    "bor",
]


# Levels: which ops to enable. 'fast' = core, 'normal' = +offsets, 'deep' = all
OP_LEVELS = {
    "fast": [
        "add",
        "sub",
        "rsub",
        "sub_signed",
        "rsub_signed",
        "absdiff",
        "neg_absdiff",
        "mul",
        "gcd",
        "lcm",
        "concat_fwd",
        "concat_rev",
        "fdiv",
        "rdiv",
        "mod",
        "rmod",
        "min",
        "max",
    ],
    "normal": None,  # means "all except bitwise+polynomial" (set below)
    "deep": None,  # means "everything"
}
OP_LEVELS["normal"] = OP_LEVELS["fast"] + [
    "add_m1",
    "add_p1",
    "mul_m1",
    "mul_p1",
    "absdiff_m1",
    "absdiff_p1",
    "sub_m1",
    "sub_p1",
    "rsub_m1",
    "rsub_p1",
]
OP_LEVELS["deep"] = list(OPERATIONS.keys())


def _digits_to_int(digits, base: int) -> int:
    v = 0
    for d in digits:
        v = v * base + int(d)
    return v


def _int_to_base_digits(value: int, base: int) -> list[int]:
    if value == 0:
        return [0]
    out = []
    while value:
        out.append(value % base)
        value //= base
    return out[::-1]


def _is_reversed_digit_mode(mode: str) -> bool:
    return mode in ("alice", "little_endian")


def _solver_category(full_ops: dict[str, str], mode: str) -> str:
    has_concat = any(op.startswith("concat_") for op in full_ops.values())
    if has_concat and mode == "little_endian":
        return "mixed_concat_little_endian"
    if has_concat:
        return "mixed_concat"
    if mode == "little_endian":
        return "little_endian"
    return "arithmetic"


class AliceEquationSolver:
    def __init__(
        self,
        prompt: str,
        search_level: str = "normal",
        op_types=None,
        answer_hint: str | None = None,
    ):
        """
        search_level:
            'fast'   — core ops only (≈12 types, fastest)
            'normal' — core + ±1 offsets (≈22 types, default)
            'deep'   — everything incl. bitwise/polynomial (≈38 types, slowest)
        op_types: override with explicit list if provided.
        answer_hint: if provided (e.g. the gold answer from train.csv), any
            symbols appearing in it that are NOT in the examples/query are added
            to content_symbols so the search can bind them to digits. This
            rescues "orphan-symbol" puzzles where the answer introduces a new
            character never seen in training examples (the digit that happens
            to be the one unused by the example mapping).
        """
        self.prompt = prompt
        if op_types is not None:
            self.op_types = op_types
        else:
            self.op_types = OP_LEVELS.get(search_level, OP_LEVELS["normal"])
        self.search_level = search_level
        self.answer_hint = answer_hint
        self.examples, self.query = self._parse(prompt)
        self._analyze()

    # ═════════════════ Parsing ═════════════════

    @staticmethod
    def _parse(prompt):
        lines = prompt.strip().split("\n")
        examples, query = [], None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            low = line.lower()
            if "determine the result for:" in low:
                idx = low.index("determine the result for:")
                query = line[idx + len("determine the result for:") :].strip()
                continue
            if any(
                kw in low
                for kw in [
                    "alice",
                    "wonderland",
                    "transformation",
                    "secret",
                    "examples",
                    "below",
                ]
            ):
                continue
            if " = " in line:
                lhs, rhs = line.split(" = ", 1)
                examples.append((lhs.strip(), rhs.strip()))
        return examples, query

    def _analyze(self):
        self.op_examples = defaultdict(list)
        self.example_op_chars = set()
        for lhs, rhs in self.examples:
            if len(lhs) >= 5:
                self.example_op_chars.add(lhs[2])
                self.op_examples[lhs[2]].append((lhs, rhs))
        self.op_chars = set(self.example_op_chars)
        if self.query and len(self.query) >= 5:
            self.op_chars.add(self.query[2])

        all_chars = set()
        for lhs, rhs in self.examples:
            all_chars.update(lhs)
            all_chars.update(rhs)
        if self.query:
            all_chars.update(self.query)
        # Rescue orphan symbols that only appear in the gold answer (so the
        # search can bind them to the unused digit). Skip op chars / whitespace.
        if self.answer_hint:
            for ch in self.answer_hint:
                if ch.strip() and ch not in self.op_chars:
                    all_chars.add(ch)
        self.content_symbols = sorted(all_chars - self.op_chars)
        self.base = len(self.content_symbols)

    def _priority(self, oc):
        pri = OP_PRIORITY.get(oc, DEFAULT_PRIORITY)
        allowed = set(self.op_types)
        p1 = [t for t in pri if t in allowed]
        p2 = [t for t in self.op_types if t not in set(p1)]
        return p1 + p2

    # ═════════════════ Main Solve ═════════════════

    def solve(self):
        """
        Returns (answer: str, details: dict) or (None, None).
        If the default search_level fails, auto-escalates to 'deep'.
        """
        ans, det = self._try_concat()
        if ans is not None:
            if self.answer_hint is None or ans == self.answer_hint:
                return ans, det

        if self.answer_hint:
            ans, det = self._try_unseen_query_concat()
            if ans is not None:
                return ans, det
            ans, det = self._try_arithmetic_gold_conditioned()
            if ans is not None:
                return ans, det
            if self.search_level != "deep":
                self.op_types = OP_LEVELS["deep"]
                self.search_level = "deep"
                ans, det = self._try_arithmetic_gold_conditioned()
                if ans is not None:
                    return ans, det
            return None, None

        ans, det = self._try_arithmetic()
        if ans is not None:
            return ans, det

        if self.search_level != "deep":
            self.op_types = OP_LEVELS["deep"]
            self.search_level = "deep"
            ans, det = self._try_arithmetic()
            if ans is not None:
                return ans, det

        return None, None

    # ═════════════ Type 1: Concatenation ═════════════

    def _try_concat(self):
        concat_ops = {}
        for op, exs in self.op_examples.items():
            if all(rhs == lhs[:2] + lhs[3:] for lhs, rhs in exs):
                concat_ops[op] = "fwd"
            elif all(rhs == lhs[3:] + lhs[:2] for lhs, rhs in exs):
                concat_ops[op] = "rev"
        if self.query and len(self.query) >= 5:
            qo = self.query[2]
            if qo in concat_ops:
                ans = (
                    self.query[:2] + self.query[3:]
                    if concat_ops[qo] == "fwd"
                    else self.query[3:] + self.query[:2]
                )
                return (
                    ans,
                    {
                        "type": "concat",
                        "category": "pure_concat",
                        "concat_mode": concat_ops[qo],
                    },
                )
        return None, None

    def _try_unseen_query_concat(self):
        """Gold-only fallback for query ops absent from examples.

        Some puzzles use an operator in the query that never appears in the
        examples. In those cases the query op can act as a plain concatenator.
        This is only accepted when the provided answer_hint exactly matches
        one of the two concatenation directions.
        """
        if not self.answer_hint or not self.query or len(self.query) < 5:
            return None, None
        qo = self.query[2]
        if qo in self.example_op_chars:
            return None, None
        gold = str(self.answer_hint).strip()
        fwd = self.query[:2] + self.query[3:]
        rev = self.query[3:] + self.query[:2]
        if gold == fwd:
            return gold, {
                "type": "concat",
                "category": "query_unseen_concat",
                "conditioned_on_answer": True,
                "query_op_unseen": True,
                "concat_mode": "fwd",
            }
        if gold == rev:
            return gold, {
                "type": "concat",
                "category": "query_unseen_concat",
                "conditioned_on_answer": True,
                "query_op_unseen": True,
                "concat_mode": "rev",
            }
        return None, None

    # ═════════════ Type 2: Arithmetic ═════════════

    def _try_arithmetic_gold_conditioned(self):
        """Search latent mapping/op/mode constrained by the known answer.

        The gold answer is encoded as an extra equation for the query op, so the
        Rust brute-force search returns only programs that satisfy both examples
        and the query outcome. This is for data generation; normal inference
        still uses _try_arithmetic().
        """
        if self.base > 12 or not self.answer_hint:
            return None, None
        sym2i = {s: i for i, s in enumerate(self.content_symbols)}

        op_eqs = defaultdict(list)
        for lhs, rhs in self.examples:
            if len(lhs) < 5:
                continue
            op = lhs[2]
            has_sign = len(rhs) > 1 and rhs[0] == op
            res_str = rhs[1:] if has_sign else rhs
            chars = [lhs[0], lhs[1], lhs[3], lhs[4]] + list(res_str)
            if not all(c in sym2i for c in chars):
                continue
            op_eqs[op].append(
                (
                    sym2i[lhs[0]],
                    sym2i[lhs[1]],
                    sym2i[lhs[3]],
                    sym2i[lhs[4]],
                    tuple(sym2i[c] for c in res_str),
                    has_sign,
                    len(res_str),
                )
            )
        if not self.query or len(self.query) < 5:
            return None, None

        qo = self.query[2]
        ql = (sym2i[self.query[0]], sym2i[self.query[1]])
        qr = (sym2i[self.query[3]], sym2i[self.query[4]])

        gold = str(self.answer_hint).strip()
        if not gold:
            return None, None
        gold_has_sign = gold[0] == qo and len(gold) > 1
        gold_res = gold[1:] if gold_has_sign else gold
        q_chars = [self.query[0], self.query[1], self.query[3], self.query[4]]
        if not all(c in sym2i for c in q_chars + list(gold_res)):
            return None, None
        op_eqs[qo].append(
            (
                sym2i[self.query[0]],
                sym2i[self.query[1]],
                sym2i[self.query[3]],
                sym2i[self.query[4]],
                tuple(sym2i[c] for c in gold_res),
                gold_has_sign,
                len(gold_res),
            )
        )
        if not op_eqs:
            return None, None

        op_candidates = {}
        for op, eqs in op_eqs.items():
            signs = {hs for _, _, _, _, _, hs, _ in eqs}
            pri = self._priority(op)
            if signs == {True}:
                op_candidates[op] = [
                    t for t in pri if t in ("neg_absdiff", "sub_signed", "rsub_signed")
                ]
            elif signs == {False}:
                op_candidates[op] = [t for t in pri if t != "neg_absdiff"]
            else:
                op_candidates[op] = [
                    t for t in pri if t in ("sub_signed", "rsub_signed")
                ]
            if all((not hs and rl == 4) for _, _, _, _, _, hs, rl in eqs):
                for t in ("concat_fwd", "concat_rev"):
                    if t in self.op_types and t not in op_candidates[op]:
                        op_candidates[op].append(t)

        n = self.base
        sorted_ops = sorted(
            op_eqs.keys(), key=lambda op: (len(op_candidates[op]), -len(op_eqs[op]))
        )

        TIER0 = {
            "add",
            "sub",
            "rsub",
            "absdiff",
            "mul",
            "gcd",
            "lcm",
            "concat_fwd",
            "concat_rev",
            "neg_absdiff",
            "sub_signed",
            "rsub_signed",
        }
        TIER1 = TIER0 | {"fdiv", "rdiv", "mod", "rmod", "min", "max"}
        TIER2 = TIER1 | {
            "add_m1",
            "add_p1",
            "mul_m1",
            "mul_p1",
            "absdiff_m1",
            "absdiff_p1",
            "sub_m1",
            "sub_p1",
            "rsub_m1",
            "rsub_p1",
        }
        full_set = set(self.op_types)
        tiers = []
        for t in (TIER0 & full_set, TIER1 & full_set, TIER2 & full_set, full_set):
            if t and t not in tiers:
                tiers.append(t)

        for tier in tiers:
            tier_cands = {
                op: [t for t in cands if t in tier]
                for op, cands in op_candidates.items()
            }
            if not all(tier_cands.values()):
                continue

            radix_candidates = [(self.base, self.base)]
            if self.base != 10:
                radix_candidates.append((10, 10))
            for radix, digit_count in radix_candidates:
                for mode in SEARCH_MODES:
                    sol = self._search(
                        op_eqs, tier_cands, sorted_ops, n, mode, radix, digit_count
                    )
                    if sol is None:
                        continue
                    mapping, ops_valid = sol
                    full_ops = {op: vs[0] for op, vs in ops_valid.items()}
                    ans, num = self._encode_answer(
                        mapping, ql, qr, qo, full_ops, mode, radix
                    )
                    if ans != gold:
                        continue
                    mapping_dict = {
                        s: mapping[i] for i, s in enumerate(self.content_symbols)
                    }
                    if (
                        radix != 10
                        or mode == "little_endian"
                        or any(op.startswith("concat_") for op in full_ops.values())
                    ):
                        ded_order, ded_trace = [], []
                    else:
                        ded_order, ded_trace = self._derive_order(
                            mapping_dict, full_ops, mode
                        )
                    return ans, {
                        "type": "arithmetic",
                        "category": _solver_category(full_ops, mode),
                        "conditioned_on_answer": True,
                        "mode": mode,
                        "ops": full_ops,
                        "mapping": mapping_dict,
                        "numeric_answer": num,
                        "solver_radix": radix,
                        "level": self.search_level,
                        "tier": len(tier),
                        "deduction_order": ded_order,
                        "deduction_trace": ded_trace,
                    }
        return None, None

    def _try_arithmetic(self):
        if self.base > 12:
            return None, None
        sym2i = {s: i for i, s in enumerate(self.content_symbols)}

        op_eqs = defaultdict(list)
        for lhs, rhs in self.examples:
            if len(lhs) < 5:
                continue
            op = lhs[2]
            has_sign = len(rhs) > 1 and rhs[0] == op
            res_str = rhs[1:] if has_sign else rhs
            chars = [lhs[0], lhs[1], lhs[3], lhs[4]] + list(res_str)
            if not all(c in sym2i for c in chars):
                continue
            op_eqs[op].append(
                (
                    sym2i[lhs[0]],
                    sym2i[lhs[1]],
                    sym2i[lhs[3]],
                    sym2i[lhs[4]],
                    tuple(sym2i[c] for c in res_str),
                    has_sign,
                    len(res_str),
                )
            )
        if not op_eqs:
            return None, None

        qo = self.query[2]
        ql = (sym2i[self.query[0]], sym2i[self.query[1]])
        qr = (sym2i[self.query[3]], sym2i[self.query[4]])

        # Per-op candidate op types, filtered by sign-prefix pattern across examples.
        # Sign pattern interpretations:
        #   all signed    -> neg_absdiff (forced -) OR signed_sub/rsub (if all a<b or all b<a)
        #   all unsigned  -> all ops except neg_absdiff (signed ops still OK; positive branch)
        #   mixed         -> only signed_sub / rsub_signed can explain both branches
        op_candidates = {}
        for op, eqs in op_eqs.items():
            signs = {hs for _, _, _, _, _, hs, _ in eqs}
            pri = self._priority(op)
            if signs == {True}:
                op_candidates[op] = [
                    t for t in pri if t in ("neg_absdiff", "sub_signed", "rsub_signed")
                ]
            elif signs == {False}:
                op_candidates[op] = [t for t in pri if t != "neg_absdiff"]
            else:  # mixed
                op_candidates[op] = [
                    t for t in pri if t in ("sub_signed", "rsub_signed")
                ]
            if all((not hs and rl == 4) for _, _, _, _, _, hs, rl in eqs):
                for t in ("concat_fwd", "concat_rev"):
                    if t in self.op_types and t not in op_candidates[op]:
                        op_candidates[op].append(t)

        n = self.base
        # Sort ops: fewer candidates + more equations → check first
        sorted_ops = sorted(
            op_eqs.keys(), key=lambda op: (len(op_candidates[op]), -len(op_eqs[op]))
        )

        # Tiered search: prefer simple ops. If a (mapping, ops) solution exists
        # using only simple ops, use it. Offset/polynomial ops are generic
        # enough that they can spuriously fit multiple permutations; restricting
        # the op pool first breaks ambiguity in favour of the canonical answer.
        TIER0 = {
            "add",
            "sub",
            "rsub",
            "absdiff",
            "mul",
            "gcd",
            "lcm",
            "concat_fwd",
            "concat_rev",
            "neg_absdiff",
            "sub_signed",
            "rsub_signed",
        }
        TIER1 = TIER0 | {"fdiv", "rdiv", "mod", "rmod", "min", "max"}
        TIER2 = TIER1 | {
            "add_m1",
            "add_p1",
            "mul_m1",
            "mul_p1",
            "absdiff_m1",
            "absdiff_p1",
            "sub_m1",
            "sub_p1",
            "rsub_m1",
            "rsub_p1",
        }
        full_set = set(self.op_types)
        tiers = []
        for t in (TIER0 & full_set, TIER1 & full_set, TIER2 & full_set, full_set):
            if t and t not in tiers:
                tiers.append(t)

        for tier in tiers:
            tier_cands = {
                op: [t for t in cands if t in tier]
                for op, cands in op_candidates.items()
            }
            if not all(tier_cands.values()):
                continue

            radix_candidates = [(self.base, self.base)]
            if self.base != 10:
                radix_candidates.append((10, 10))
            for radix, digit_count in radix_candidates:
                for mode in SEARCH_MODES:
                    sol = self._search(
                        op_eqs, tier_cands, sorted_ops, n, mode, radix, digit_count
                    )
                    if sol is None:
                        continue
                    mapping, ops_valid = sol

                    qop_cands_base = (
                        ops_valid[qo] if qo in ops_valid else self._priority(qo)
                    )
                    qop_cands = [
                        t for t in qop_cands_base if t in tier
                    ] or qop_cands_base
                    ex_ops = {op: vs[0] for op, vs in ops_valid.items()}

                    for qt in qop_cands:
                        full_ops = {**ex_ops, qo: qt}
                        ans, num = self._encode_answer(
                            mapping, ql, qr, qo, full_ops, mode, radix
                        )
                        if ans is not None:
                            mapping_dict = {
                                s: mapping[i]
                                for i, s in enumerate(self.content_symbols)
                            }
                            if (
                                radix != 10
                                or mode == "little_endian"
                                or any(
                                    op.startswith("concat_") for op in full_ops.values()
                                )
                            ):
                                ded_order, ded_trace = [], []
                            else:
                                ded_order, ded_trace = self._derive_order(
                                    mapping_dict, full_ops, mode
                                )
                            return ans, {
                                "type": "arithmetic",
                                "category": _solver_category(full_ops, mode),
                                "mode": mode,
                                "ops": full_ops,
                                "mapping": mapping_dict,
                                "numeric_answer": num,
                                "solver_radix": radix,
                                "level": self.search_level,
                                "tier": len(tier),
                                "deduction_order": ded_order,
                                "deduction_trace": ded_trace,
                            }
        return None, None

    # ─── forward-propagation order ───

    def _derive_order(self, mapping_dict, ops, mode):
        """Order in which content symbols are committed by a CSP search using
        MRV (smallest current domain) with **max-propagation tie-break**:
        among MRV-tied candidates, simulate committing each (using its gold
        digit), run forward-checking to fixed point, and pick the candidate
        whose commit causes the largest cascade — i.e. opens the most "doors"
        for the rest of the variables. This yields the path along which a
        learner sees the most forced (|dom|=1) deductions after each step.

        Returns list[str] of content_symbols in commit order.
        """
        from itertools import permutations as _perm

        symbols = list(mapping_dict.keys())
        if not symbols:
            return []
        alice = mode == "alice"
        parsed = []
        parsed_meta = []  # parallel: (ex_idx_1based, lhs_str, rhs_str, op_char)
        for ex_idx, (lhs, rhs) in enumerate(self.examples, 1):
            if len(lhs) < 5:
                continue
            oc = lhs[2]
            if oc not in ops:
                continue
            has_sign = len(rhs) > 1 and rhs[0] == oc
            res_str = rhs[1:] if has_sign else rhs
            lhs_chars = (lhs[0], lhs[1], lhs[3], lhs[4])
            if not all(c in mapping_dict for c in lhs_chars + tuple(res_str)):
                continue
            parsed.append((lhs_chars, has_sign, tuple(res_str), ops[oc]))
            parsed_meta.append((ex_idx, lhs, rhs, oc))

        def compute_v(L, R, ot, has_sign):
            fn = OPERATIONS.get(ot)
            if fn is None:
                return None, False
            v = fn(L, R)
            if v is None:
                return None, False
            if ot in SIGNED_OPS:
                if (v < 0) != has_sign:
                    return None, False
                return (-v if v < 0 else v), True
            if ot == "neg_absdiff":
                if not has_sign:
                    return None, False
                return v, True
            if has_sign or v < 0:
                return None, False
            return v, True

        def update_feasible(
            determined, domains, lhs_chars, has_sign, res_str, ot, feasible
        ):
            rl = len(res_str)
            avail = sorted(set(range(10)) - set(determined.values()))
            unk_order = []
            seen = set()
            for c in lhs_chars:
                if c not in determined and c not in seen:
                    unk_order.append(c)
                    seen.add(c)
            k = len(unk_order)
            if k > len(avail):
                return

            def commit_row(assign):
                full = dict(determined)
                full.update(assign)
                L = full[lhs_chars[0]] * 10 + full[lhs_chars[1]]
                R = full[lhs_chars[2]] * 10 + full[lhs_chars[3]]
                if alice:
                    L = (L % 10) * 10 + L // 10
                    R = (R % 10) * 10 + R // 10
                v, ok = compute_v(L, R, ot, has_sign)
                if not ok:
                    return
                s_str = str(v)
                if len(s_str) > rl:
                    return
                s_str = s_str.zfill(rl)
                if alice:
                    s_str = s_str[::-1]
                used_local = set(full.values())
                rhs_extra = {}
                for kk in range(rl):
                    c = res_str[kk]
                    d = int(s_str[kk])
                    if c in full:
                        if full[c] != d:
                            return
                    elif c in rhs_extra:
                        if rhs_extra[c] != d:
                            return
                    else:
                        if d not in domains[c]:
                            return
                        if d in used_local:
                            return
                        rhs_extra[c] = d
                        used_local.add(d)
                for c, d in assign.items():
                    feasible[c].add(d)
                for c, d in rhs_extra.items():
                    feasible[c].add(d)

            if k == 0:
                commit_row({})
                return
            for perm in _perm(avail, k):
                ok = True
                for i, val in enumerate(perm):
                    if val not in domains[unk_order[i]]:
                        ok = False
                        break
                if not ok:
                    continue
                commit_row(dict(zip(unk_order, perm)))

        def propagate(determined, domains):
            """Run forward-checking + AllDifferent to fixed point. Mutates domains."""
            used = set(determined.values())
            for s in symbols:
                if s not in determined:
                    domains[s] = domains[s] - used
            changed = True
            while changed:
                changed = False
                for lhs_chars, has_sign, res_str, ot in parsed:
                    feasible = defaultdict(set)
                    update_feasible(
                        determined, domains, lhs_chars, has_sign, res_str, ot, feasible
                    )
                    involved = set()
                    for c in lhs_chars:
                        if c not in determined:
                            involved.add(c)
                    for c in res_str:
                        if c not in determined:
                            involved.add(c)
                    for u in involved:
                        new_d = domains[u] & feasible.get(u, set())
                        if new_d != domains[u]:
                            domains[u] = new_d
                            changed = True
            return domains

        def per_example_pass(det_state, dom_state):
            """Iterate forward passes (each = process every example in document
            order, intersecting feasibility into running domains) until fixed
            point. Returns a structured trace mirroring what propagate() does
            internally, but with per-example records visible.

            For CoT rendering, generator typically shows pass 1 in full and
            mentions later passes only if they produce additional narrowing."""
            running = {s: set(dom_state[s]) for s in symbols}
            used = set(det_state.values())
            for s in symbols:
                if s not in det_state:
                    running[s] = running[s] - used  # AllDifferent step

            initial_running = {s: sorted(running[s]) for s in symbols}

            passes = []
            iter_no = 0
            while True:
                iter_no += 1
                pass_records = []
                pass_changed = False
                for (lhs_chars, has_sign, res_str, ot), (
                    ex_idx,
                    lhs_orig,
                    rhs_orig,
                    op_char,
                ) in zip(parsed, parsed_meta):
                    feasible = defaultdict(set)
                    update_feasible(
                        det_state, running, lhs_chars, has_sign, res_str, ot, feasible
                    )
                    involved = set()
                    for c in lhs_chars:
                        if c not in det_state:
                            involved.add(c)
                    for c in res_str:
                        if c not in det_state:
                            involved.add(c)
                    involved_sorted = sorted(involved)
                    per_sym = {
                        c: sorted(feasible.get(c, set())) for c in involved_sorted
                    }
                    before_this = {c: sorted(running[c]) for c in involved_sorted}
                    new_running = {s: set(running[s]) for s in symbols}
                    ex_changed = False
                    for u in involved:
                        nd = running[u] & feasible.get(u, set())
                        if nd != running[u]:
                            ex_changed = True
                            pass_changed = True
                        new_running[u] = nd
                    intersected = {c: sorted(new_running[c]) for c in involved_sorted}
                    pass_records.append(
                        {
                            "ex_idx": ex_idx,
                            "label": f"ex{ex_idx}",
                            "lhs": lhs_orig,
                            "rhs": rhs_orig,
                            "op_char": op_char,
                            "op_type": ot,
                            "syms_unknown": involved_sorted,
                            "feasible_per_sym": per_sym,
                            "running_before_this_ex": before_this,
                            "intersected_after": intersected,
                            "narrowed": ex_changed,
                        }
                    )
                    running = new_running

                passes.append(
                    {
                        "iter": iter_no,
                        "examples": pass_records,
                        "any_change": pass_changed,
                    }
                )
                if not pass_changed or iter_no >= 5:
                    break

            final = {s: sorted(running[s]) for s in symbols}
            return {
                "initial": initial_running,
                "passes": passes,
                "final": final,
            }

        domains = {s: set(range(10)) for s in symbols}
        determined = {}
        order = []
        trace = []

        while len(determined) < len(symbols):
            # Capture per-example feasibility for this step's deduction.
            # Done BEFORE propagate (so the pass sees pristine starting domains).
            pre_propagate_domains = {s: set(domains[s]) for s in symbols}
            per_ex = per_example_pass(determined, pre_propagate_domains)

            propagate(determined, domains)

            rem = [s for s in symbols if s not in determined]
            if not rem:
                break
            min_dom = min(len(domains[s]) for s in rem)
            tied = [s for s in rem if len(domains[s]) == min_dom]

            cascade_info = []
            if len(tied) == 1:
                chosen = tied[0]
            else:
                # Max-propagation tie-break: simulate commit on each tied
                # candidate, score by total reduction in the OTHERS' domain
                # sizes (more reduction = more "doors opened"). Forced steps
                # (others dropping to |dom|=1) get extra weight via the sum
                # already (since 10→1 reduces 9, more than 10→3 reducing 7).
                best = None
                best_score = -1
                best_forced = -1
                for cand in tied:
                    sim_det = dict(determined)
                    sim_det[cand] = mapping_dict[cand]
                    sim_dom = {s: set(domains[s]) for s in symbols}
                    propagate(sim_det, sim_dom)
                    score = 0
                    forced = 0
                    for s in symbols:
                        if s in sim_det:
                            continue
                        score += len(domains[s]) - len(sim_dom[s])
                        if len(sim_dom[s]) == 1:
                            forced += 1
                    cascade_info.append({"sym": cand, "score": score, "forced": forced})
                    # Prefer most reduction; tie → most newly-forced; tie → char
                    key = (score, forced, -ord(cand[0]) if cand else 0)
                    if best is None or key > (best_score, best_forced, -ord(best[0])):
                        best = cand
                        best_score = score
                        best_forced = forced
                chosen = best

            dom_at_commit = sorted(domains[chosen])
            domains_before_full = {s: sorted(domains[s]) for s in symbols}
            order.append(chosen)

            # Snapshot domains after committing chosen + propagation (for
            # generator's <domain_after> tag).
            sim_det_after = dict(determined)
            sim_det_after[chosen] = mapping_dict[chosen]
            sim_dom_after = {s: set(domains[s]) for s in symbols}
            sim_dom_after[chosen] = {mapping_dict[chosen]}
            propagate(sim_det_after, sim_dom_after)
            domains_after_full = {s: sorted(sim_dom_after[s]) for s in symbols}

            trace.append(
                {
                    "sym": chosen,
                    "digit": mapping_dict[chosen],
                    "domain": dom_at_commit,
                    "kind": "forced" if len(dom_at_commit) == 1 else "guess",
                    "tied": list(tied),
                    "cascade": cascade_info,
                    "domains_before": domains_before_full,
                    "domains_after": domains_after_full,
                    "determined_before": dict(determined),
                    "per_example_feasibility": per_ex,
                }
            )
            determined[chosen] = mapping_dict[chosen]
            domains = {s: set(sim_dom_after[s]) for s in symbols}

        return order, trace

    # ─── brute-force search (with early pruning) ───

    def _search(
        self, op_eqs, op_cands, sorted_ops, n, mode, radix=None, digit_count=None
    ):
        if radix is None:
            radix = n
        if digit_count is None:
            digit_count = n
        if _rust_helper is None:
            return self._search_python(
                op_eqs, op_cands, sorted_ops, n, mode, radix, digit_count
            )
        rust = _rust_helper

        groups = []
        for op in sorted_ops:
            cands = [OP_NAME_TO_IDX[ot] for ot in op_cands[op]]
            eq_list = []
            for l0, l1, r0, r1, ridx, hs, rl in op_eqs[op]:
                # Format: [l0, l1, r0, r1, rl, has_sign] + res_syms
                eq_list.append([l0, l1, r0, r1, rl, 1 if hs else 0] + list(ridx))
            groups.append((cands, eq_list))

        result = rust.arithmetic_search(
            n,
            mode == "alice",
            groups,
            radix,
            digit_count,
            MODE_TO_IDX.get(mode, 0),
        )
        if result is None:
            return None
        perm, ops_valid_raw = result
        ops_valid = {}
        for op, valid_idx in zip(sorted_ops, ops_valid_raw):
            ops_valid[op] = [OP_IDX_TO_NAME[i] for i in valid_idx]
        return list(perm), ops_valid

    def _search_python(
        self, op_eqs, op_cands, sorted_ops, n, mode, radix=None, digit_count=None
    ):
        reverse_digits = _is_reversed_digit_mode(mode)
        base = radix or n
        pool = digit_count or n
        for perm in permutations(range(pool), n):
            ops_valid = {}
            ok = True
            for op in sorted_ops:
                valid = []
                for ot in op_cands[op]:
                    if ot == "concat_fwd" or ot == "concat_rev":
                        all_match = True
                        for l0, l1, r0, r1, ridx, hs, rl in op_eqs[op]:
                            if hs or rl != 4:
                                all_match = False
                                break
                            want = (
                                (l0, l1, r0, r1)
                                if ot == "concat_fwd"
                                else (r0, r1, l0, l1)
                            )
                            if tuple(ridx) != want:
                                all_match = False
                                break
                        if all_match:
                            valid.append(ot)
                        continue
                    fn = OPERATIONS[ot]
                    is_signed = ot in SIGNED_OPS
                    all_match = True
                    for l0, l1, r0, r1, ridx, hs, rl in op_eqs[op]:
                        L = perm[l0] * base + perm[l1]
                        R = perm[r0] * base + perm[r1]
                        if reverse_digits:
                            L = (L % base) * base + L // base
                            R = (R % base) * base + R // base
                        v = fn(L, R)
                        if v is None:
                            all_match = False
                            break
                        if is_signed:
                            # Sign of raw result must match whether rhs has a prefix
                            if (v < 0) != hs:
                                all_match = False
                                break
                            v = -v if v < 0 else v
                        else:
                            if v < 0:
                                all_match = False
                                break
                        digits = _int_to_base_digits(v, base)
                        if len(digits) > rl:
                            all_match = False
                            break
                        if reverse_digits:
                            digits = ([0] * (rl - len(digits)) + digits)[::-1]
                        else:
                            digits = [0] * (rl - len(digits)) + digits
                        expected_digits = [perm[i] for i in ridx]
                        if digits != expected_digits:
                            all_match = False
                            break
                    if all_match:
                        valid.append(ot)
                if not valid:
                    ok = False
                    break
                ops_valid[op] = valid
            if ok:
                return list(perm), ops_valid
        return None

    # ─── encode answer back to symbols ───

    def _encode_answer(self, mapping, ql, qr, q_op, ops, mode, radix=None):
        # Only symbols that appear in the puzzle (content_symbols) are trusted.
        # If the query result needs a digit that never appeared in any example,
        # the system is underdetermined → reject it rather than guess a fallback.
        rev = {mapping[i]: s for i, s in enumerate(self.content_symbols)}

        base = radix or self.base
        L = mapping[ql[0]] * base + mapping[ql[1]]
        R = mapping[qr[0]] * base + mapping[qr[1]]
        if _is_reversed_digit_mode(mode):
            L = (L % base) * base + L // base
            R = (R % base) * base + R // base

        ot = ops.get(q_op, "absdiff")
        if ot == "concat_fwd":
            return rev[mapping[ql[0]]] + rev[mapping[ql[1]]] + rev[
                mapping[qr[0]]
            ] + rev[mapping[qr[1]]], None
        if ot == "concat_rev":
            return rev[mapping[qr[0]]] + rev[mapping[qr[1]]] + rev[
                mapping[ql[0]]
            ] + rev[mapping[ql[1]]], None
        fn = OPERATIONS.get(ot, OPERATIONS["absdiff"])
        res = fn(L, R)
        if res is None:
            return None, None

        if ot in SIGNED_OPS:
            prefix = q_op if res < 0 else ""
            mag = -res if res < 0 else res
            numeric = res
        elif ot == "neg_absdiff":
            if res < 0:
                return None, None
            prefix = q_op
            mag = res
            numeric = -res
        else:
            if res < 0:
                return None, None
            prefix = ""
            mag = res
            numeric = res

        digits = _int_to_base_digits(mag, base)
        target_len = None
        if self.answer_hint:
            gold = str(self.answer_hint).strip()
            if gold:
                target_len = len(
                    gold[1:] if len(gold) > 1 and gold[0] == q_op else gold
                )
        if target_len is not None and len(digits) < target_len:
            digits = [0] * (target_len - len(digits)) + digits
        if _is_reversed_digit_mode(mode):
            digits = digits[::-1]

        ans = prefix
        for d in digits:
            if d in rev:
                ans += rev[d]
            else:
                # Digit not covered by any content symbol → underdetermined.
                return None, None
        return ans, numeric


# ═══════════════════════════ CLI ═══════════════════════════

if __name__ == "__main__":
    import time
    import pandas as pd

    df = pd.read_csv("data/train.csv")
    df = df[df["category"] == "equation_symbolic"]

    concat_ok = arith_ok = fail = none = 0
    t0 = time.time()
    for idx in range(len(df)):
        s = AliceEquationSolver(df["prompt"].iloc[idx])
        ans, det = s.solve()
        actual = df["answer"].iloc[idx]

        if ans is None:
            none += 1
        elif ans == actual:
            if det and det["type"] == "concat":
                concat_ok += 1
            else:
                arith_ok += 1
        else:
            fail += 1
            print(f"FAIL idx={idx} got={ans!r} want={actual!r} det={det}")

        if idx % 50 == 0:
            print(
                f"idx={idx} concat={concat_ok} arith={arith_ok} fail={fail} none={none} {time.time() - t0:.1f}s"
            )

    total = concat_ok + arith_ok
    print(
        f"\nDone: {total}/{len(df)} correct  (concat={concat_ok} arith={arith_ok})  fail={fail}  none={none}  time={time.time() - t0:.1f}s"
    )


def solve_with_trace(self):
    """
    Like solve(), but records the real search trace.
    Returns (answer, details) where details['trace'] contains:
    {
        'branches': [
            {
                'tier': 'TIER0',
                'tier_ops': [...],         # ops allowed in this tier
                'mode': 'standard',
                'result': 'no_solution',   # or 'solved'
                'failed_example': {...},   # first example that no perm could satisfy (if available)
            },
            ...
        ],
        'winning_branch': {
            'tier': 'TIER2',
            'mode': 'alice',
            'ops_per_symbol': {
                '*': {
                    'candidates_tested': ['mul', 'mul_p1', 'mul_m1', ...],
                    'survivors': ['mul_p1'],
                },
                '-': {
                    'candidates_tested': ['absdiff', 'sub', 'rsub', ...],
                    'survivors': ['absdiff', 'rsub', 'rsub_signed'],
                },
            },
            'permutations_searched': 3628800,
            'mapping': [5, 6, 4, 9, 2, 7, 3, 0, 1, 8],
        },
    }
    """
    trace = {"branches": [], "winning_branch": None}

    # Try concat first (no trace needed, trivial)
    ans, det = self._try_concat()
    if ans is not None:
        return ans, det

    # Arithmetic search with trace
    ans, det = self._try_arithmetic_traced(trace)
    if ans is not None:
        det["trace"] = trace
        return ans, det

    # Auto-escalate
    if self.search_level != "deep":
        self.op_types = OP_LEVELS["deep"]
        self.search_level = "deep"
        ans, det = self._try_arithmetic_traced(trace)
        if ans is not None:
            det["trace"] = trace
            return ans, det

    return None, None


def _try_arithmetic_traced(self, trace):
    """_try_arithmetic with trace logging."""
    if self.base > 12:
        return None, None
    sym2i = {s: i for i, s in enumerate(self.content_symbols)}

    op_eqs = defaultdict(list)
    for lhs, rhs in self.examples:
        if len(lhs) < 5:
            continue
        op = lhs[2]
        has_sign = len(rhs) > 1 and rhs[0] == op
        res_str = rhs[1:] if has_sign else rhs
        chars = [lhs[0], lhs[1], lhs[3], lhs[4]] + list(res_str)
        if not all(c in sym2i for c in chars):
            continue
        op_eqs[op].append(
            (
                sym2i[lhs[0]],
                sym2i[lhs[1]],
                sym2i[lhs[3]],
                sym2i[lhs[4]],
                tuple(sym2i[c] for c in res_str),
                has_sign,
                len(res_str),
            )
        )
    if not op_eqs:
        return None, None

    qo = self.query[2]
    ql = (sym2i[self.query[0]], sym2i[self.query[1]])
    qr = (sym2i[self.query[3]], sym2i[self.query[4]])

    # Build op candidates (same logic as original)
    op_candidates = {}
    for op, eqs in op_eqs.items():
        signs = {hs for _, _, _, _, _, hs, _ in eqs}
        pri = self._priority(op)
        if signs == {True}:
            op_candidates[op] = [
                t for t in pri if t in ("neg_absdiff", "sub_signed", "rsub_signed")
            ]
        elif signs == {False}:
            op_candidates[op] = [t for t in pri if t != "neg_absdiff"]
        else:
            op_candidates[op] = [t for t in pri if t in ("sub_signed", "rsub_signed")]

    n = self.base
    sorted_ops = sorted(
        op_eqs.keys(), key=lambda op: (len(op_candidates[op]), -len(op_eqs[op]))
    )

    # Build tiers
    TIER0 = {
        "add",
        "sub",
        "rsub",
        "absdiff",
        "mul",
        "gcd",
        "lcm",
        "neg_absdiff",
        "sub_signed",
        "rsub_signed",
    }
    TIER1 = TIER0 | {"fdiv", "rdiv", "mod", "rmod", "min", "max"}
    TIER2 = TIER1 | {
        "add_m1",
        "add_p1",
        "mul_m1",
        "mul_p1",
        "absdiff_m1",
        "absdiff_p1",
        "sub_m1",
        "sub_p1",
        "rsub_m1",
        "rsub_p1",
    }
    full_set = set(self.op_types)
    tier_defs = [
        ("TIER0_core", TIER0 & full_set),
        ("TIER1_divmod", TIER1 & full_set),
        ("TIER2_offsets", TIER2 & full_set),
        ("FULL", full_set),
    ]
    tiers = []
    seen = set()
    for name, t in tier_defs:
        key = frozenset(t)
        if t and key not in seen:
            tiers.append((name, t))
            seen.add(key)

    for tier_name, tier in tiers:
        tier_cands = {
            op: [t for t in cands if t in tier] for op, cands in op_candidates.items()
        }
        if not all(tier_cands.values()):
            # Some op has zero candidates in this tier → skip
            trace["branches"].append(
                {
                    "tier": tier_name,
                    "tier_size": len(tier),
                    "mode": "skipped",
                    "result": "no_candidates",
                    "reason": f"op(s) {[op for op, c in tier_cands.items() if not c]} "
                    f"have no candidates in this tier",
                }
            )
            continue

        for mode in ["standard", "alice"]:
            # Count permutations that will be searched
            from math import perm as math_perm

            n_perms = math_perm(10, n)

            sol = self._search_traced(
                op_eqs,
                tier_cands,
                sorted_ops,
                n,
                mode,
                trace,
                tier_name,
                n_perms,
            )

            if sol is None:
                trace["branches"].append(
                    {
                        "tier": tier_name,
                        "tier_size": len(tier),
                        "mode": mode,
                        "result": "no_solution",
                        "candidates_per_op": {
                            op: list(cands) for op, cands in tier_cands.items()
                        },
                        "permutations_searched": n_perms,
                    }
                )
                continue

            # Found solution!
            mapping, ops_valid = sol

            # Record op-level detail: which candidates survived
            ops_detail = {}
            for op in sorted_ops:
                ops_detail[op] = {
                    "candidates_tested": list(tier_cands[op]),
                    "survivors": list(ops_valid[op]),
                }

            # Try query op candidates
            qop_cands_base = ops_valid[qo] if qo in ops_valid else self._priority(qo)
            qop_cands = [t for t in qop_cands_base if t in tier] or qop_cands_base
            ex_ops = {op: vs[0] for op, vs in ops_valid.items()}

            for qt in qop_cands:
                full_ops = {**ex_ops, qo: qt}
                ans, num = self._encode_answer(mapping, ql, qr, qo, full_ops, mode)
                if ans is not None:
                    trace["winning_branch"] = {
                        "tier": tier_name,
                        "tier_size": len(tier),
                        "mode": mode,
                        "ops_per_symbol": ops_detail,
                        "permutations_searched": n_perms,
                        "mapping": list(mapping),
                    }
                    mapping_dict = {
                        s: mapping[i] for i, s in enumerate(self.content_symbols)
                    }
                    ded_order, ded_trace = self._derive_order(
                        mapping_dict, full_ops, mode
                    )
                    return ans, {
                        "type": "arithmetic",
                        "mode": mode,
                        "ops": full_ops,
                        "mapping": mapping_dict,
                        "numeric_answer": num,
                        "level": self.search_level,
                        "tier": len(tier),
                        "deduction_order": ded_order,
                        "deduction_trace": ded_trace,
                    }

    return None, None


def _search_traced(
    self, op_eqs, op_cands, sorted_ops, n, mode, trace, tier_name, n_perms
):
    """Search with trace — delegates to existing _search."""
    # The actual search is unchanged; we just wrap it.
    # The real search already tries all permutations and all op candidates.
    # What matters for the trace is the RESULT: which ops survived.
    return self._search(op_eqs, op_cands, sorted_ops, n, mode)


# ─── Bind trace helpers as methods ───
AliceEquationSolver.solve_with_trace = solve_with_trace
AliceEquationSolver._try_arithmetic_traced = _try_arithmetic_traced
AliceEquationSolver._search_traced = _search_traced


def apply_query_trace(self, mapping_dict, ops, mode):
    """
    Mirror _encode_answer but emit a structured trace for <apply_query>.

    Returns dict with:
        L_letters, R_letters: 2-char strings from query
        L_digits_raw, R_digits_raw: 2-digit ints before alice flip
        L_digits, R_digits: 2-digit ints fed into op (alice-flipped if alice)
        op_char, op_type, mode
        raw_result: signed/unsigned int from op
        sign_prefix: '' or op_char
        magnitude: |result| (or None if unsolvable)
        magnitude_str: zero-padded? — solver doesn't pad, it uses str(mag)
        magnitude_alice: alice-reversed magnitude string (or same as str if standard)
        encoded_answer: final symbol string
        digit_to_symbol: rev-mapping
    """
    if not self.query or len(self.query) < 5:
        return None
    q = self.query
    L_letters = q[0] + q[1]
    R_letters = q[3] + q[4]
    op_char = q[2]
    if any(c not in mapping_dict for c in (q[0], q[1], q[3], q[4])):
        return None
    L_raw = mapping_dict[q[0]] * 10 + mapping_dict[q[1]]
    R_raw = mapping_dict[q[3]] * 10 + mapping_dict[q[4]]
    if mode == "alice":
        L_use = (L_raw % 10) * 10 + L_raw // 10
        R_use = (R_raw % 10) * 10 + R_raw // 10
    else:
        L_use = L_raw
        R_use = R_raw

    ot = ops.get(op_char, "absdiff")
    fn = OPERATIONS.get(ot, OPERATIONS["absdiff"])
    res = fn(L_use, R_use)
    if res is None:
        return None

    if ot in SIGNED_OPS:
        sign_prefix = op_char if res < 0 else ""
        mag = -res if res < 0 else res
    elif ot == "neg_absdiff":
        if res < 0:
            return None
        sign_prefix = op_char
        mag = res
    else:
        if res < 0:
            return None
        sign_prefix = ""
        mag = res

    rs = str(mag)
    rs_alice = rs[::-1] if mode == "alice" else rs
    rev = {mapping_dict[s]: s for s in mapping_dict}
    if not all(int(c) in rev for c in rs_alice):
        return None
    encoded = sign_prefix + "".join(rev[int(c)] for c in rs_alice)

    return {
        "query": q,
        "L_letters": L_letters,
        "R_letters": R_letters,
        "op_char": op_char,
        "op_type": ot,
        "mode": mode,
        "L_digits_raw": L_raw,
        "R_digits_raw": R_raw,
        "L_digits": L_use,
        "R_digits": R_use,
        "raw_result": res,
        "sign_prefix": sign_prefix,
        "magnitude": mag,
        "magnitude_str": rs,
        "magnitude_alice": rs_alice,
        "digit_to_symbol": rev,
        "encoded_answer": encoded,
    }


AliceEquationSolver.apply_query_trace = apply_query_trace


def narrow_op_candidates_structural(self):
    """Return per-op-char narrowed candidates from STRUCTURAL features only
    (no digit search). Used by generator's <analyze_ops>.

    Returns:
      {
        op_char: {
          "sign_pattern": "all_signed" | "all_unsigned" | "mixed",
          "result_lens": [...],
          "candidates": [list of op_type names that survive structural filter],
          "examples_using_this_op": [(lhs, rhs), ...]
        },
        ...
      }
    """
    result = {}
    for op_char, exs in self.op_examples.items():
        # --- sign-pattern detection ---
        signs = []
        result_lens = set()
        for lhs, rhs in exs:
            has_sign = len(rhs) > 1 and rhs[0] == op_char
            signs.append(has_sign)
            mag = rhs[1:] if has_sign else rhs
            result_lens.add(len(mag))

        if all(signs):
            sign_pattern = "all_signed"
        elif not any(signs):
            sign_pattern = "all_unsigned"
        else:
            sign_pattern = "mixed"

        # --- per-op narrowing from sign pattern ---
        pri = self._priority(op_char)
        if sign_pattern == "all_signed":
            candidates = [
                t for t in pri if t in ("neg_absdiff", "sub_signed", "rsub_signed")
            ]
        elif sign_pattern == "mixed":
            candidates = [t for t in pri if t in ("sub_signed", "rsub_signed")]
        else:
            # all_unsigned: exclude neg_absdiff
            candidates = [t for t in pri if t != "neg_absdiff"]

        # --- result-length magnitude filtering ---
        # For 2-digit operands (max 99 each):
        #   result_len >= 3: only ops that can produce 3+ digit results
        #   result_len == 2: keep ops bounded by 99
        #   result_len == 1: keep ops that always produce single digits
        max_result_len = max(result_lens) if result_lens else 1
        min_result_len = min(result_lens) if result_lens else 1

        # Helper to check if an op CAN produce a result of given digit length
        # with 2-digit operands (0-99 each).
        def _op_can_produce_result_len(op_name: str, target_len: int) -> bool:
            if op_name in (
                "add",
                "add_p1",
                "add_p2",
                "add_m1",
                "add_m2",
                "sub_signed",
                "rsub_signed",
            ):
                # a+b max 198 (3 digits), min 0 (1 digit)
                return target_len <= 3
            if op_name in (
                "mul",
                "mul_p1",
                "mul_p2",
                "mul_m1",
                "mul_m2",
                "mul_half",
                "mul_double",
                "mul_plus_a",
                "mul_plus_b",
                "mul_minus_a",
                "mul_minus_b",
                "sq_diff",
                "sq_sum",
                "a2_plus_b",
                "a_plus_b2",
            ):
                # can produce up to 4 digits (99*99=9801)
                return target_len <= 4
            if op_name in (
                "sub",
                "rsub",
                "absdiff",
                "neg_absdiff",
                "absdiff_m1",
                "absdiff_p1",
                "absdiff_m2",
                "absdiff_p2",
                "sub_m1",
                "sub_p1",
                "rsub_m1",
                "rsub_p1",
            ):
                # |diff| max 99 (2 digits), can be 1 digit
                return target_len <= 2
            if op_name in ("gcd", "lcm", "fdiv", "rdiv", "mod", "rmod", "min", "max"):
                # all bounded by max(a,b) ≤ 99 (2 digits)
                return target_len <= 2
            if op_name in ("xor", "band", "bor"):
                # bitwise ops ≤ 127 (2^7-1 for 0-99), so ≤ 3 digits but mostly 2
                return target_len <= 2
            # Unknown ops: be permissive
            return True

        if max_result_len >= 3:
            # Need ops that can produce 3+ digit results
            candidates = [
                t for t in candidates if _op_can_produce_result_len(t, max_result_len)
            ]
        elif max_result_len == 2:
            # Keep ops bounded by 99
            candidates = [t for t in candidates if _op_can_produce_result_len(t, 2)]

        result[op_char] = {
            "sign_pattern": sign_pattern,
            "result_lens": sorted(result_lens),
            "candidates": candidates,
            "examples_using_this_op": list(exs),
        }

    return result


AliceEquationSolver.narrow_op_candidates_structural = narrow_op_candidates_structural
