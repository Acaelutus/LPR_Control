import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from demo.setup_whitelist import AUTHORIZED_PLATES, setup_whitelist
from src.database.access_db import create_database
from src.utils.config import Config
from src.utils.logger import logger
from src.utils.plate_text import normalize_plate_text


logger.disabled = True


class CoreBehaviorTest(unittest.TestCase):
    def test_normalize_plate_text(self):
        self.assertEqual(normalize_plate_text(" h-386 ce 76 "), "H386CE76")
        self.assertEqual(normalize_plate_text("T899CM76\n"), "T899CM76")

    def test_default_config_points_to_existing_assets(self):
        config = Config("configs/config.yaml")

        self.assertEqual(config.plate_detector.model_format, "yolov5")
        self.assertTrue(Path(config.plate_detector.weights_path).exists())
        self.assertTrue(Path(config.ocr.model_path).exists())
        self.assertEqual(config.database.type, "sqlite")
        self.assertEqual(config.controller.type, "simulated")

    def test_setup_whitelist_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "access_list.db")

            setup_whitelist(db_path=db_path, verbose=False)
            setup_whitelist(db_path=db_path, verbose=False)

            with closing(sqlite3.connect(db_path)) as conn:
                rows = [
                    row[0]
                    for row in conn.execute("SELECT plate FROM whitelist ORDER BY plate")
                ]

        self.assertEqual(rows, sorted(AUTHORIZED_PLATES))
        self.assertEqual(len(rows), 22)

    def test_database_logs_access_attempts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "access_list.db")
            db = create_database("sqlite", db_path=db_path)

            db.add_plate("H386CE76", "Demo Plate")
            db.log_access("H386CE76", db.is_authorized("H386CE76"))
            db.log_access("UNKNOWN", db.is_authorized("UNKNOWN"))

            with closing(sqlite3.connect(db_path)) as conn:
                rows = list(
                    conn.execute(
                        "SELECT plate, authorized FROM access_log ORDER BY id"
                    )
                )

        self.assertEqual(rows, [("H386CE76", 1), ("UNKNOWN", 0)])


if __name__ == "__main__":
    unittest.main()
