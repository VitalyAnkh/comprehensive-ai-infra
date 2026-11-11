"""Independent checks for the F04 delivery audit."""
import json
import re
import unittest
from pathlib import Path
import sys

for ancestor in Path(__file__).resolve().parents:
    if (ancestor / "src/infra_calc/sources.py").is_file():
        sys.path.insert(0, str(ancestor / "src"))
        break
from infra_calc import delivery
from infra_calc.sources import is_lfs_pointer


class ScenarioArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = delivery.scenario_artifacts()

    def test_every_scenario_id_has_both_outputs(self):
        self.assertEqual(self.row["scenarios_without_results"], [])
        manifest = json.loads((delivery.PROJECT / "results/manifest.json").read_text())
        produced = {entry["file"] for entry in manifest["artifacts"]}
        book_path = delivery.PROJECT / "scenarios/book.json"
        if is_lfs_pointer(book_path):
            h2_scenarios = json.loads((delivery.PROJECT / "configs/h2-scenarios.json").read_text())
            sample = h2_scenarios["h2_topics"][0]["id"]
            self.assertIn(f"results/{sample}.json", produced)
            self.assertIn(f"results/{sample}.md", produced)
            return
        scenarios = json.loads(book_path.read_text())
        sample = next(row["id"] for rows in scenarios.values() if isinstance(rows, list)
                      for row in rows if isinstance(row, dict) and row.get("id"))
        self.assertIn(f"results/{sample}.json", produced)
        self.assertIn(f"results/{sample}.md", produced)

    def test_derived_names_cover_the_groups_without_an_id_field(self):
        book_path = delivery.PROJECT / "scenarios/book.json"
        if is_lfs_pointer(book_path):
            self.skipTest("scenarios/book.json is an unpulled Git LFS pointer")
        scenarios = json.loads(book_path.read_text())
        for name, builder in delivery.DERIVED_NAMES.items():
            self.assertIn(name, scenarios, name)
            derived = builder(scenarios[name][0])
            self.assertTrue((delivery.PROJECT / "results" / (derived + ".json")).exists(), derived)

    def test_no_result_is_left_without_a_scenario(self):
        self.assertEqual(self.row["results_without_a_scenario"], [])
        self.assertTrue(self.row["passed"])

    def test_a_missing_scenario_output_would_be_caught(self):
        # The check compares against the manifest, so an id with no entry must fail.
        manifest = json.loads((delivery.PROJECT / "results/manifest.json").read_text())
        produced = {entry["file"] for entry in manifest["artifacts"]}
        self.assertNotIn("results/this-scenario-does-not-exist.json", produced)


class ReaderEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = delivery.reader_entry()

    def test_every_subcommand_is_documented(self):
        self.assertEqual(self.row["undocumented_subcommands"], [])
        self.assertGreater(self.row["subcommands"], 60)

    def test_every_topic_module_is_reachable(self):
        self.assertEqual(self.row["topics_reachable_from_nowhere"], [])
        self.assertEqual(self.row["reachable_topics"], self.row["topic_modules"])

    def test_reachability_counts_a_helper_imported_by_another_topic(self):
        # image_stages has no subcommand; image_generation imports it.
        source = (delivery.PROJECT / "src/infra_calc/topics/image_generation.py").read_text()
        self.assertIn("from .image_stages import", source)
        cli = (delivery.PROJECT / "src/infra_calc/cli.py").read_text()
        self.assertNotIn("image_stages", re.sub(r"^from \.topics import.*$", "", cli, flags=re.M))

    def test_documented_means_the_runnable_form(self):
        readme = (delivery.PROJECT / "README.md").read_text()
        for name in ("delivery", "coverage", "verify-results", "fetch"):
            self.assertIn(f"calc.py {name}", readme, name)


class SourceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = delivery.source_audit()

    def setUp(self):
        if "unpulled Git LFS pointer" in self.row.get("reason", ""):
            self.skipTest("inventory/sources.json is an unpulled Git LFS pointer")

    def test_the_saved_report_is_current(self):
        self.assertTrue(self.row["saved_report_is_current"])
        self.assertEqual(self.row["blocks_citing_a_missing_result"], 0)

    def test_it_cannot_pass_while_blocks_are_unreviewed(self):
        self.assertGreater(self.row["blocks_pending_review"], 0)
        self.assertFalse(self.row["passed"])
        self.assertIn("mechanical map does not close F01", self.row["reason"])

    def test_the_only_sectioned_gap_is_the_orientation_section(self):
        self.assertTrue(set(self.row["sectioned_gaps"]) <= {"1.1", "1.1.1", "1.1.2", "1.1.3"},
                        self.row["sectioned_gaps"])


class AuditTests(unittest.TestCase):
    def test_checks_are_reported_separately_and_never_averaged(self):
        result = delivery.calculate(include_rendered_book=False)
        self.assertEqual(set(result["checks"]),
                         {"results_regenerable", "source_audited", "reader_entry"})
        self.assertEqual(sorted(result["passed"] + result["failed"]), sorted(result["checks"]))
        # The verdict is a set of named booleans, not a score: no top-level
        # number summarises across the checks.
        self.assertIsInstance(result["delivered"], bool)
        numeric = [key for key, value in result.items()
                   if isinstance(value, (int, float)) and not isinstance(value, bool)]
        self.assertEqual(numeric, [])
        for row in result["checks"].values():
            self.assertIsInstance(row["passed"], bool)

    def test_delivered_requires_every_check(self):
        result = delivery.calculate(include_rendered_book=False)
        self.assertEqual(result["delivered"], not result["failed"])
        self.assertFalse(result["delivered"])
        self.assertIn("source_audited", result["failed"])

    def test_result_is_json_serialisable(self):
        text = json.dumps(delivery.calculate(include_rendered_book=False),
                          ensure_ascii=False, allow_nan=False)
        self.assertIn("f04-delivery-audit", text)

    def test_markdown_names_each_failing_check(self):
        result = delivery.calculate(include_rendered_book=False)
        text = delivery.markdown(result)
        self.assertIn("四项检查各自独立报告", text)
        for name in result["failed"]:
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
