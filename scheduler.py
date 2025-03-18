import logging
logger = logging.getLogger(__name__)

import threading
import time
import os
from queue import Queue
from datetime import datetime
from src.setup import scheduler_setup, CredentialManager
from src.file_manager import FolderSetupMixin
from src.batch_handler import BatchManager
from src.file_manager import BatchSummaryLogger



class BatchManagerFactory:
    def __init__(self, tool_config):
        self.tool_config = tool_config

    def create_batch_manager(self):
        return BatchManager(self.tool_config)


class CredentialManagerFactory:
    def __init__(self, tool_config):
        self.tool_config = tool_config

    def create_credential_manager(self):
        return CredentialManager(self.tool_config)


class SchedulerFolder(FolderSetupMixin):
    def __init__(self, tool_config):
        self.config = tool_config
        self.setup_shared_folders(self.config)


# Initialize configuration and folder setup
tool_config = scheduler_setup()
folder_setup = SchedulerFolder(tool_config)

# Initialize factories
batch_manager_factory = BatchManagerFactory(tool_config)
credential_manager_factory = CredentialManagerFactory(tool_config)


def worker(job_queue, credential_manager_factory, batch_manager_factory):
    thread_credential_manager = credential_manager_factory.create_credential_manager()
    while True:
        file_path = job_queue.get()
        if file_path is None:
            break
        try:
            logger.info(f"Worker processing file: {file_path}")
            # Create a new BatchManager instance for the current thread
            with batch_manager_factory.create_batch_manager() as thread_batch_manager:
                thread_batch_manager.execute_job(file_path, thread_credential_manager)
            logger.info(f"Worker completed job for file: {file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
            # Handle exception as needed
        finally:
            job_queue.task_done()
    thread_credential_manager.close()


def run_scheduler(tool_config, credential_manager_factory, batch_manager_factory, process_newest_first=False):
    logger.info("Cronjob: Scheduler started, looking for pending jobs")

    job_queue = Queue()

    threads = []
    for i in range(tool_config['max_cuncurrent_jobs']):
        t = threading.Thread(
            target=worker, 
            args=(job_queue, credential_manager_factory, batch_manager_factory)
        )
        t.daemon = True  
        t.start()
        threads.append(t)

    try:
        while True:
            pending_files = [
                f
                for f in os.listdir(folder_setup.batches_folder)
                if f.endswith("PENDING.json")
            ]
            if process_newest_first:
                pending_files.sort(reverse=True)

            if not pending_files:
                logger.info("No pending jobs found.")
                logger.info(f"Queue size: {job_queue.qsize()}") 
                batch_summary_logger = BatchSummaryLogger(tool_config)
                batch_summary_logger.cleanup_stuck_progress_files(timeout_seconds=tool_config.get('stuck_file_timeout',7200))     

            for filename in pending_files:
                job_queue.put(filename)
                logger.info(f"Added file to job queue: {filename}")

       
            logger.info(f"Will sleep for {tool_config['scheduler_frequency']} seconds, good night")
            time.sleep(tool_config['scheduler_frequency'])
    except KeyboardInterrupt:
        logger.info("Scheduler interrupted. Shutting down.")
        
    finally:
        for _ in threads:
            job_queue.put(None)
        for t in threads:
            t.join()


if __name__ == "__main__":
    run_scheduler(tool_config, credential_manager_factory, batch_manager_factory)