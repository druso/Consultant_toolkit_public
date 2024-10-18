import logging
logger = logging.getLogger(__name__)
import time
import os
import json
from src.external_tools import OxyLabsManager, LlmManager
from datetime import datetime
from src.setup import scheduler_setup, CredentialManager
from src.file_manager import AppLogger, DataFrameProcessor, FileLockManager
from src.batch_handler import DfBatchesConstructor, BatchManager
import pandas as pd


tool_config = scheduler_setup()
batch_manager = BatchManager(tool_config)
credential_manager = CredentialManager(tool_config, use_streamlit=False)



def run_scheduler(frequency=tool_config['scheduler_frequency'], process_newest_first=False):
    logger.info("Cronjob: Scheduler started, looking for pending jobs")
    
    while True:
        pending_files_paths = [f for f in os.listdir(batch_manager.batches_folder) if f.endswith("PENDING.json")]
        if process_newest_first:
            pending_files_paths.reverse()

        if not pending_files_paths:
            logger.info(f"No pending jobs found.")


        for file_path in pending_files_paths:
            job_payload = batch_manager.load_payload(file_path)
            if job_payload:
                logger.info(f"Found new file to process: {file_path}")
                batch_manager.execute_job(job_payload,file_path, credential_manager)

        logger.info(f"Will sleep for {frequency} seconds, good night")
        time.sleep(frequency)


if __name__ == "__main__":
    run_scheduler()