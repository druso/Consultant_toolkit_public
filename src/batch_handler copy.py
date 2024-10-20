import logging
logger = logging.getLogger(__name__)

from src.external_tools import( 
    LlmManager, 
    OxyLabsManager, 
    StopProcessingError, 
    RetryableError, 
    SkippableError
    )
from src.file_manager import(
    DataFrameProcessor,
    FileLockManager, 
    SessionLogger, 
    FolderSetupMixin, 
    BatchSummaryLogger,
    BatchProgressLogger,
    )
from src.dataformats import BatchRequestPayload
import pandas as pd
import json
import os
import time
import re


class DfBatchesConstructor():
    def __init__(self, df_processor:DataFrameProcessor, session_logger:SessionLogger):
        self.df_processor = df_processor
        self.session_logger = session_logger

    def df_batches_handler(self, 
                       func, 
                       response_column, 
                       query_column, 
                       batch_size=0, 
                       force_string=True, 
                       save_interval=5,
                       *args, 
                       **kwargs):
        df = self.df_processor.processed_df

        if query_column not in df.columns:
            raise StopProcessingError(
                f"Column '{query_column}' not found in DataFrame. Available columns are: {', '.join(df.columns)}"
            )
        
        elif query_column == "Select a column...":
            raise StopProcessingError(
                f"Please select a column"
            )
        
        if response_column not in df.columns:
            df[response_column] = pd.NA

        # Only process rows where response_column is NaN
        unprocessed_indices = df[df[response_column].isna()].index
        total_unprocessed = len(unprocessed_indices)
        if batch_size == 0 or batch_size > total_unprocessed:
            batch_size = total_unprocessed

        total_to_process = len(unprocessed_indices)
        logger.info(f"Total rows to process: {batch_size} out of {total_to_process} available")

        processed_count = 0
        start_time = time.time()
        progress_saved = None
        for idx in unprocessed_indices:
            if batch_size > 0 and processed_count >= batch_size:
                logger.info(f"Reached batch size limit of {batch_size}")
                break

            query_value = df.at[idx, query_column]
            try:
                if pd.isna(query_value):
                    message = "Query value is NaN or None"
                    if force_string:
                        message = str(message)
                    df.at[idx, response_column] = message
                    logger.warning(f"Skipped row {idx}: {message}")
                else:
                    result = func(query_value, *args, **kwargs)
                    if force_string:
                        result = str(result) if result is not None else ""
                    df.at[idx, response_column] = result
            except Exception as e:
                error_message = f"Error: {e}"
                if force_string:
                    error_message = str(error_message)
                df.at[idx, response_column] = error_message
                logger.error(f"Error processing row {idx}: {e}")

            processed_count += 1

            progress_saved = False
            if processed_count % save_interval == 0: 
                save_path = self.session_logger.log_excel(df, version="in_progress", return_path=True )
                progress_saved = True
                logger.info(f"Saved progress at {processed_count} rows at {save_path}")

            yield {
                "processed_count": processed_count,
                "progress_saved": progress_saved,
                "df": None
            }

        save_path = self.session_logger.log_excel(df, version="processed", return_path=True)
        end_time = time.time()
        logger.info(f"Processing complete! Processed {processed_count} rows in {end_time - start_time:.2f} seconds. Saved at {save_path}")

        yield {
                "processed_count": processed_count,
                "progress_saved": progress_saved,
                "df": df
            }




class BatchManager(FolderSetupMixin):

    def __init__(self, tool_config):
        self.setup_shared_folders(tool_config)
        self.tool_config = tool_config
        self.progress_logger = BatchProgressLogger(self.tool_config)
        self.batch_summary_logger = BatchSummaryLogger(self.tool_config)

    def load_payload(self, payload_filename) -> BatchRequestPayload:
        file_path = os.path.join(self.batches_folder, payload_filename)
        file_content = FileLockManager(file_path).secure_read()
        data = json.loads(file_content)
        return BatchRequestPayload.from_dict(data)

    def dataframe_loader(self, filepath) -> pd.DataFrame:
        try:
            if filepath.endswith('.csv'):
                return pd.read_csv(filepath)
            elif filepath.endswith(('.xls', '.xlsx')):
                return pd.read_excel(filepath, engine='openpyxl', dtype=str)       
        except Exception as e:
            logger.error(f"Failed to load data: {e}")

    def execute_job(self,payload_filename, credential_manager):

        try:
            job_payload = self.load_payload(payload_filename)
        except Exception as e:
            logger.error(f"Failed to load payload: {e}")
            return None

        session_logger = SessionLogger(job_payload.user_id,self.tool_config)
        session_logger.session_id = job_payload.session_id
        session_logger.file_name = job_payload.input_file

        if job_payload.function == "llm_request":
            llm_manager = LlmManager(
                                    session_logger, 
                                    credential_manager, 
                                    job_payload.kwargs.get("llm_model", "gpt-4o-mini"), 
                                    job_payload.kwargs.get("llm_temp", 0.3)
                                    )

            function = llm_manager.llm_request

        else:
            oxylabs_manager = OxyLabsManager(session_logger, credential_manager)

            available_functions = {
                "serp_paginator": oxylabs_manager.serp_paginator,
                "get_amazon_product_info": oxylabs_manager.get_amazon_product_info,
                "get_amazon_review": oxylabs_manager.get_amazon_review,
                "web_crawler": oxylabs_manager.web_crawler,
                }
            function = available_functions.get(job_payload.function, None)

        if not function:
            logger.error(f"Function {job_payload.function} not available")
            return None

        filepath_toprocess = os.path.join(session_logger.session_files_folder(), job_payload.input_file)
        dataframe = self.dataframe_loader(filepath_toprocess)
        df_processor= DataFrameProcessor(dataframe)

        batches_constructor = DfBatchesConstructor(df_processor, session_logger)

        try:
            total_rows = df_processor.processed_df[job_payload.query_column].notna().sum()
            current_payload_filename = payload_filename
            self.batch_summary_logger.update_batch_summary(job_payload, status="WIP", total_rows=total_rows)
            current_payload_filename = self.update_payload_status(current_payload_filename, "WIP")
            self.update_wip_progress(job_payload.user_id, job_payload.batch_id, 0)
            for progress in batches_constructor.df_batches_handler(
                func=function, 
                batch_size=job_payload.batch_size, 
                response_column=job_payload.response_column, 
                query_column=job_payload.query_column, 
                **job_payload.kwargs
            ):
                if progress["df"] is None and not progress["progress_saved"]:
                    pass
                
                if progress["df"] is None and progress["progress_saved"]:
                    print (progress)
                    current_percentage = (progress["processed_count"] / total_rows) * 100
                    status = int(current_percentage)
                    self.update_wip_progress(job_payload.user_id, job_payload.batch_id, status)
                    logging.info(f"updated processing status")

                elif progress["df"] is not None:
                    self.update_wip_progress(job_payload.user_id, job_payload.batch_id, 100)
                    completed_filename = self.update_payload_status(current_payload_filename, "COMPLETED")
                    self.batch_summary_logger.update_batch_summary(job_payload, status="COMPLETED", filename=completed_filename)
                    
                    logging.info(f"Ended processing executing function")
                    
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")




















    def update_wip_progress(self, user_id, batch_id, wip_percentage):
        """
        Creates or updates an empty file with a name that reflects the batch progress.
        """
        if not 0 <= wip_percentage <= 100:
            raise ValueError("Progress must be between 0 and 100")
        
        for existing_file in os.listdir(self.wip_folder):
            if existing_file.startswith(f"{user_id}_{batch_id}_"):
                os.remove(os.path.join(self.wip_folder, existing_file))

        file_name = f"{user_id}_{batch_id}_{wip_percentage}"
        file_path = os.path.join(self.wip_folder, file_name)

        if not wip_percentage == 100:
            open(file_path, 'w').close()
            print(f"Progress updated: {file_name}")
    
    def post_stop_request(self, user_id, batch_id):
        file_name = f"{user_id}_{batch_id}_STOP"
        file_path = os.path.join(self.batches_folder, file_name)
        open(file_path, 'w').close()
    
    def handled_stop_request(self, user_id, batch_id):
        file_name = f"{user_id}_{batch_id}_STOP"
        file_path = os.path.join(self.batches_folder, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

    def update_payload_status(self, payload_filename, status):
        payload_filepath = os.path.join(self.batches_folder, payload_filename)
        
        if not status in ["PENDING", "WIP", "COMPLETED"]:
            raise ValueError(f"Invalid status: {status}. Must be 'PENDING', 'WIP', or 'COMPLETED'.")

        payload_dir_path, payload_filename = os.path.split(payload_filepath)

        base_name = re.sub(r'_(PENDING|WIP|COMPLETED)\.json$', '', payload_filename)
        
        new_filename = f"{base_name}_{status}.json"
        new_payload_path = os.path.join(payload_dir_path, new_filename)
        file_lock = FileLockManager(payload_filepath)

        try:
            file_lock.secure_rename(new_payload_path)
            logger.info(f"Updated {payload_filename} to {new_filename}")        
            if status == "COMPLETED":
                archived_payload_path = os.path.join(self.completed_dir,payload_filename)
                FileLockManager(payload_dir_path).secure_move(archived_payload_path)
            logger.info(f"Successfully updated payload status for '{payload_filename}'.")
            return new_filename 
                  
        except Exception as e:
            logger.error(f"Error updating payload status: {str(e)}")
            return payload_filename