from src.solver_eq_symbolic import AliceEquationSolver


PROMPT = r"""In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
`!*[{ = '"[`
\'*'> = ![@
\'-!` = \\
`!*\& = '@'{
Now, determine the result for: [[-!'"""


def main():
    expected = "@&"
    solver = AliceEquationSolver(PROMPT, answer_hint=expected)
    answer, details = solver.solve()

    print(f"answer: {answer}")
    print(f"expected: {expected}")
    print(f"correct: {answer == expected}")
    if details:
        print(f"type: {details.get('type')}")
        print(f"mode: {details.get('mode')}")


if __name__ == "__main__":
    main()
