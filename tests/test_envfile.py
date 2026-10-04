"""`.env` loading: parsing rules and precedence of real environment variables."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from signature_verification_system.envfile import load_env_file, parse_env_file


class TestParseEnvFile(unittest.TestCase):
    def _write(self, tmp: str, text: str) -> Path:
        path = Path(tmp) / ".env"
        path.write_text(text, encoding="utf-8")
        return path

    def test_parses_comments_blanks_quotes_and_export(self) -> None:
        with TemporaryDirectory() as tmp:
            path = self._write(tmp, "# comment\n\nA=1\nexport B = two \nC=\"x y\"\nD='z'\nE=\n")
            self.assertEqual(parse_env_file(path), {"A": "1", "B": "two", "C": "x y", "D": "z", "E": ""})

    def test_value_may_contain_equals(self) -> None:
        with TemporaryDirectory() as tmp:
            self.assertEqual(parse_env_file(self._write(tmp, "URL=a=b\n")), {"URL": "a=b"})

    def test_rejects_line_without_equals(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                parse_env_file(self._write(tmp, "NOT_A_PAIR\n"))


class TestLoadEnvFile(unittest.TestCase):
    def test_existing_environment_wins(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("SIGV_TEST_KEEP=file\nSIGV_TEST_NEW=file\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"SIGV_TEST_KEEP": "shell"}, clear=False):
                applied = load_env_file(path)
                self.assertEqual(os.environ["SIGV_TEST_KEEP"], "shell")
                self.assertEqual(os.environ["SIGV_TEST_NEW"], "file")
                self.assertEqual(applied, {"SIGV_TEST_NEW": "file"})
            os.environ.pop("SIGV_TEST_NEW", None)

    def test_missing_file_is_not_an_error(self) -> None:
        self.assertEqual(load_env_file(Path("/nonexistent/.env")), {})


if __name__ == "__main__":
    unittest.main()
