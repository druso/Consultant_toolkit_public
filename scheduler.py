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



def run_scheduler(frequency=tool_config['scheduler_frequency']):
    logger.info("Cronjob: Scheduler started")
    
    while True:
        logger.info("Checking for new jobs")
        pending_files = [f for f in os.listdir(batch_manager.batches_folder) if f.endswith("PENDING.json")]

        for file in pending_files:
            job_payload = batch_manager.load_payload(file)
            if job_payload:
                batch_manager.execute_job(job_payload, credential_manager)
                batch_manager.rename_completed_payload(file)
            
        batch_manager.update_batches_summary()
        time.sleep(frequency)


if __name__ == "__main__":
    run_scheduler()