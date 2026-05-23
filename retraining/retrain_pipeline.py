import subprocess
from pathlib import Path
import logging
from 

def trigger_retrain(self, reason="unspecified"):

    lock = Path("flags/retrain.lock")
    lock.parent.mkdir(exist_ok=True)

    if lock.exists():
        logging.warning("Retrain already in progress, skipping")
        return

    # create lock BEFORE starting process
    lock.touch()

    try:
        subprocess.Popen(
            [
                "python",
                "pipeline/retrain_pipeline.py",
                "--reason",
                reason
            ],
            stdout=open("logs/retrain.log", "a"),
            stderr=subprocess.STDOUT
        )

        logging.info("Retrain triggered successfully (async)")

    except Exception as e:
        logging.error(f"Failed to start retrain: {e}")
        lock.unlink(missing_ok=True)
















































from pathlib import Path
import logging

def main():

    lock = Path("flags/retrain.lock")

    try:
        logging.info("Starting retraining...")

        # -----------------------
        # 1. Load data
        # 2. Train model
        # 3. Evaluate
        # 4. Save new model
        # 5. Update registry
        # -----------------------

        run_retraining_pipeline()

        logging.info("Retraining completed successfully")

    except Exception as e:
        logging.error(f"Retraining failed: {e}")

    finally:
        # ALWAYS remove lock
        if lock.exists():
            lock.unlink()
            logging.info("Lock file removed")


if __name__ == "__main__":
    main()