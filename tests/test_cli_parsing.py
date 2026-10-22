import unittest

from pde_prism.cli import build_parser, validate_screen_args


class CLIParsingTests(unittest.TestCase):
    def test_required_combinations_parse(self):
        combinations = [
            ["--task", "PDE7", "--threshold", "0.85"],
            ["--task", "PDE7", "--threshold", "0.85", "--max-sd", "0.15"],
            ["--task", "PDE7", "--exclude", "PDE4:0.30"],
            ["--include", "PDE4:0.8", "--include", "PDE7:0.8"],
            ["--task", "PDE7", "--condition", "PDE7>=0.85"],
            ["--task", "PDE7", "--selectivity-margin", "0.4"],
            ["--task", "PDE7", "--mw-min", "150", "--mw-max", "500"],
            ["--task", "PDE7", "--threshold", "0.85", "--max-sd", "0.15", "--exclude", "PDE4:0.3", "--selectivity-margin", "0.4", "--qed-min", "0.3", "--top", "100"],
        ]
        for extra in combinations:
            with self.subTest(extra=extra):
                args = build_parser().parse_args(["screen", "--input", "x.csv", "--output", "y.csv", *extra])
                validate_screen_args(args)

    def test_sd_range_validation(self):
        args = build_parser().parse_args(["screen", "--input", "x.csv", "--output", "y.csv", "--task", "PDE7", "--min-sd", "0.2", "--max-sd", "0.1"])
        with self.assertRaises(ValueError):
            validate_screen_args(args)


if __name__ == "__main__":
    unittest.main()

