"""Initialize the local whitelist used by the demo."""

import sqlite3
import sys
from contextlib import closing
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.access_db import create_database
from src.utils.logger import logger


AUTHORIZED_PLATES = (
    "H386CE76",
    "T899CM76",
    "O718CY777",
    "B098EO76",
    "X313HO76",
    "A280PM76",
    "K768XO76",
    "E684AA11",
    "M897YO76",
    "H216AP76",
    "P115PC799",
    "Y472MP76",
    "B882TE196",
    "P793XO76",
    "K542HP76",
    "A371MO76",
    "K348TO76",
    "M802EY797",
    "X803KP76",
    "M847TP76",
    "P073MP76",
    "H795KM76",
)


def setup_whitelist(db_path: str = "data/access_list.db", verbose: bool = True) -> int:
    """Reset whitelist and add the allowed plate numbers."""

    logger.info("Setting up access control whitelist...")

    db = create_database("sqlite", db_path=db_path)

    # Keep the whitelist deterministic: only the list above is authorized.
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("DELETE FROM whitelist")
        conn.commit()

    added = 0
    for plate in AUTHORIZED_PLATES:
        if db.add_plate(plate, "Demo Plate"):
            added += 1
            if verbose:
                logger.info(f"Added {plate} (Demo Plate)")
        else:
            logger.warning(f"Plate {plate} already exists")

    if verbose:
        logger.info("\n" + "=" * 60)
        logger.info("Current Whitelist:")
        logger.info("=" * 60)

        for plate_info in db.get_all_plates():
            logger.info(
                f"  {plate_info['plate']:12} | Owner: {plate_info['owner']:10} | "
                f"Added: {plate_info['added_at']}"
            )

        logger.info("=" * 60)
        logger.info(f"\nWhitelist setup complete: {len(AUTHORIZED_PLATES)} plates authorized\n")

    return added


if __name__ == "__main__":
    setup_whitelist()
