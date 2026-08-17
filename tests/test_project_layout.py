from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectLayoutSmokeTest(unittest.TestCase):
    def test_expected_research_areas_exist(self) -> None:
        for relative in (
            "README.md",
            "AGENTS.md",
            "docs/ARCHITECTURE.md",
            "experiments/README.md",
            "results/README.md",
            "config/node.example.toml",
            "src/infra_lab/__init__.py",
        ):
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).is_file(), relative)

    def test_large_local_artifacts_are_ignored(self) -> None:
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in ("models/", "data/", "artifacts/", "*.gguf", "*.safetensors"):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, ignore)


if __name__ == "__main__":
    unittest.main()
