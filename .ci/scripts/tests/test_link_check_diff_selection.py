import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(sys.platform == "linux", "The scripts under test run on Linux runners only")
class TestLinkCheckDiffSelection(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmpdir.name)
        self.run_cmd("git init -b main")
        self.run_cmd('git config user.name "Test User"')
        self.run_cmd('git config user.email "test@example.com"')
        self.write("README.md", "base\n")
        self.write("docs/feature-target.md", "feature target\n")
        self.write("docs/main-target.md", "main target\n")
        self.run_cmd("git add README.md docs/feature-target.md docs/main-target.md")
        self.run_cmd('git commit -m "base"')
        self.run_cmd("git checkout -b feature")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def run_cmd(
        self, command: str, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            shell=True,
            cwd=self.repo,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )

    def write(self, relative_path: str, content: str) -> None:
        path = self.repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def create_diverged_history(
        self, feature_path: str, feature_content: str, main_path: str, main_content: str
    ) -> tuple[str, str]:
        self.write(feature_path, feature_content)
        self.run_cmd(f"git add {feature_path}")
        self.run_cmd('git commit -m "feature change"')
        head_sha = self.run_cmd("git rev-parse HEAD").stdout.strip()

        self.run_cmd("git checkout main")
        self.write(main_path, main_content)
        self.run_cmd(f"git add {main_path}")
        self.run_cmd('git commit -m "main change"')
        base_sha = self.run_cmd("git rev-parse HEAD").stdout.strip()
        return base_sha, head_sha

    def make_fake_curl(self) -> dict[str, str]:
        bin_dir = self.repo / "bin"
        bin_dir.mkdir()
        curl = bin_dir / "curl"
        curl.write_text("#!/bin/sh\nprintf '200'\n")
        curl.chmod(curl.stat().st_mode | stat.S_IEXEC)
        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        return env

    def test_lint_urls_uses_merge_base_for_changed_lines(self) -> None:
        base_sha, head_sha = self.create_diverged_history(
            "feature.md",
            "https://example.com/feature\n",
            "main.md",
            "https://example.com/main\n",
        )

        result = self.run_cmd(
            f"bash {REPO_ROOT / 'scripts' / 'lint_urls.sh'} {base_sha} {head_sha}",
            env=self.make_fake_curl(),
        )

        self.assertIn("feature.md", result.stdout)
        self.assertNotIn("main.md", result.stdout)

    def test_lint_xrefs_uses_merge_base_for_changed_lines(self) -> None:
        base_sha, head_sha = self.create_diverged_history(
            "feature.md",
            textwrap.dedent(
                """\
                [feature](docs/feature-target.md)
                """
            ),
            "main.md",
            textwrap.dedent(
                """\
                [main](docs/main-target.md)
                """
            ),
        )

        result = self.run_cmd(
            f"bash {REPO_ROOT / 'scripts' / 'lint_xrefs.sh'} {base_sha} {head_sha}"
        )

        self.assertIn("feature.md", result.stdout)
        self.assertNotIn("main.md", result.stdout)
