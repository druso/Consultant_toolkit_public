import time
import os
import json
from src.external_tools import OxyLabsManager, LlmManager
from datetime import datetime
from src.setup import scheduler_setup, CredentialManager
from src.file_manager import AppLogger, DataFrameProcessor, FileLockManager
from src.batch_handler import DfBatchesConstructor
import pandas as pd


tool_config = scheduler_setup()
credential_manager = CredentialManager(tool_config, use_streamlit=False)
batched_folder= os.path.join(tool_config['shared_folder'], 'batches')

def check_payloads(directory=batched_folder, load_latest=True):
    pending_files = [f for f in os.listdir(directory) if f.endswith("PENDING.json")]
    
    if not pending_files:
        print("No pending files found")
        return None, None
    
    target_file = pending_files[-1] if load_latest else pending_files[0]
    
    file_path = os.path.join(directory, target_file)
    file_content = FileLockManager(file_path).secure_read()
    return json.loads(file_content), target_file


def rename_completed_payload(payload_filename, directory=batched_folder):
    os.rename(os.path.join(directory, payload_filename), os.path.join(directory, payload_filename.replace("PENDING", "COMPLETED")))

def dataframe_loader(filepath) -> pd.DataFrame:
    try:
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            return pd.read_excel(filepath, engine='openpyxl', dtype=str)       
    except Exception as e:
        print(f"Failed to load data: {e}")

def execute_job(job_payload):

    use_streamlit = False
    app_logger = AppLogger(job_payload.get("user_id"),tool_config, use_streamlit=use_streamlit)
    app_logger.session_id = job_payload.get("session_id")
    app_logger.file_name = job_payload.get("input_file")

    if job_payload.get("function") == "llm_request":
        llm_manager = LlmManager(job_payload['kwargs'].get("llm_model", "gpt-4o-mini"), 
                                app_logger, 
                                credential_manager, 
                                tool_config, 
                                use_streamlit)
        llm_manager.llm_temp = job_payload['kwargs'].get("llm_temp", 0.3)
        function = llm_manager.llm_request

    else:
        oxylabs_manager = OxyLabsManager(app_logger, credential_manager)

        available_functions = {
            "serp_crawler": oxylabs_manager.serp_crawler,
            "get_amazon_product_info": oxylabs_manager.get_amazon_product_info,
            "get_amazon_review": oxylabs_manager.get_amazon_review,
            "web_crawler": oxylabs_manager.web_crawler,
            }
        function = available_functions.get(job_payload.get("function"), None)

    if not function:
        print(f"Function {job_payload.get('function')} not available")
        return None

    filepath = os.path.join(app_logger.session_files_folder(), job_payload.get("input_file",""))
    dataframe = dataframe_loader(filepath)
    df_processor= DataFrameProcessor(dataframe, use_streamlit=False)

    batches_constructor = DfBatchesConstructor(df_processor, app_logger)

    try:
        for progress in batches_constructor.batch_requests(
            func=function, 
            batch_size=job_payload.get("batch_size"), 
            response_column=job_payload.get("response_column"), 
            query_column=job_payload.get("query_column"), 
            streamlit=False,
            **job_payload.get("kwargs")
        ):
            if isinstance(progress, str):
                print(progress)
            elif isinstance(progress, pd.DataFrame):
                print("Processing complete.")
    except Exception as e:
        print(f"Error details: {e}")


def run_scheduler(frequency=tool_config['scheduler_frequency']):
    print(f"{datetime.now()} - Cronjob: Scheduler started")
    
    while True:
        print(f"{datetime.now()} - Checking for new jobs")
        
        job_payload, payload_filename= check_payloads()
        if job_payload:
            execute_job(job_payload)
            rename_completed_payload(payload_filename)
        time.sleep(frequency)


if __name__ == "__main__":
    run_scheduler()