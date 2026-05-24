import subprocess
import os
import tempfile
import unittest


class TestSQLFluffLinterConflict(unittest.TestCase):
    def test_l003_l010_conflict(self):
        """
        This test checks for conflicting fix suggestions between L003 (Indentation) and L010 (Keywords uppercase)
        when linting a multi-line JOIN statement.
        """
        sql_code = """
SELECT
    *
FROM
    table1
JOIN
    table2
        ON table1.id = table2.table1_id;
"""

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as temp_file:
            temp_file.write(sql_code)
            temp_file_path = temp_file.name

        try:
            # Run SQLFluff lint with L003 and L010 enabled
            result = subprocess.run(
                [
                    "sqlfluff",
                    "lint",
                    temp_file_path,
                    "--rules",
                    "L003,L010",  # Explicitly specify the conflicting rules
                ],
                capture_output=True,
                text=True,
            )

            # Check for errors.  Exit code != 0 indicates errors.
            self.assertEqual(result.returncode, 1, "SQLFluff lint should have returned an error (code 1).")

            # Check if output indicates conflict (e.g., contradictory fix suggestions).
            output_str = result.stdout
            self.assertTrue(
                "L003" in output_str and "L010" in output_str,
                "Both L003 and L010 rules should have triggered.",
            )

            # Check the specific fixes being suggested. This is not an ideal check, as the wording
            # can change.
            self.assertTrue("Fix L003" in output_str, "L003 fix should be suggested.")
            self.assertTrue("Fix L010" in output_str, "L010 fix should be suggested.")

            # Run SQLFluff fix (autofix) and check the result
            fix_result = subprocess.run(
                [
                    "sqlfluff",
                    "fix",
                    temp_file_path,
                    "--rules",
                    "L003,L010",
                    "--fixed-output-file", "-", # Output the fixed result to stdout
                ],
                capture_output=True,
                text=True,
            )

            # Validate that SQLFluff fix returns 0 (no unfixable errors).
            self.assertEqual(fix_result.returncode, 0, "SQLFluff fix should have returned 0 (success).")
            fixed_sql_code = fix_result.stdout

            expected_fixed_code = """SELECT
    *
FROM
    table1
JOIN
    table2
    ON table1.id = table2.table1_id;
"""

            # We assert here that the final fixed code matches the expected output.
            # This serves as the ultimate verification that the autofix resolves the conflicts.

            self.assertEqual(
                fixed_sql_code.strip(), expected_fixed_code.strip(), "Fixed SQL code should match the expected code."
            )


        finally:
            # Clean up the temporary file
            os.remove(temp_file_path)


if __name__ == "__main__":
    unittest.main()