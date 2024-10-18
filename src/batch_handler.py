import logging
logger = logging.getLogger(__name__)

from src.external_tools import LlmManager, OxyLabsManager, StopProcessingError, RetryableError, SkippableError
from src.file_manager import DataFrameProcessor, FileLockManager, AppLogger
import pandas as pd
import json
import os
import time
import re


class DfBatchesConstructor():
    def __init__(self, df_processor:DataFrameProcessor, app_logger:AppLogger):
        self.df_processor = df_processor
        self.app_logger = app_logger

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

            if processed_count % save_interval == 0:
                save_path = self.app_logger.log_excel(df, version="in_progress", return_path=True )
                logger.info(f"Saved progress at {processed_count} rows at {save_path}")

            yield processed_count

        save_path = self.app_logger.log_excel(df, version="processed", return_path=True)
        end_time = time.time()
        logger.info(f"Processing complete! Processed {processed_count} rows in {end_time - start_time:.2f} seconds. Saved at {save_path}")

        yield df



class BatchManager:
    def __init__(self, tool_config):
        self.tool_config = tool_config
        self.batches_folder = os.path.join(tool_config['shared_folder'], 'batches')
        os.makedirs(self.batches_folder, exist_ok=True)
        self.summaries_dir = os.path.join( self.batches_folder,self.tool_config['shared_summaries_folder'])
        os.makedirs(self.summaries_dir, exist_ok=True)
        self.completed_dir = os.path.join(self.batches_folder, self.tool_config['shared_completed_folder'])
        os.makedirs(self.completed_dir, exist_ok=True)

    def load_payload(self, payload_filename):
        file_path = os.path.join(self.batches_folder, payload_filename)
        file_content = FileLockManager(file_path).secure_read()
        return json.loads(file_content)

    def dataframe_loader(self, filepath) -> pd.DataFrame:
        try:
            if filepath.endswith('.csv'):
                return pd.read_csv(filepath)
            elif filepath.endswith(('.xls', '.xlsx')):
                return pd.read_excel(filepath, engine='openpyxl', dtype=str)       
        except Exception as e:
            logger.error(f"Failed to load data: {e}")

    def execute_job(self,job_payload,filename, credential_manager):

        app_logger = AppLogger(job_payload.get("user_id"),self.tool_config)
        app_logger.session_id = job_payload.get("session_id")
        app_logger.file_name = job_payload.get("input_file")

        if job_payload.get("function") == "llm_request":
            llm_manager = LlmManager(
                                    app_logger, 
                                    credential_manager, 
                                    job_payload['kwargs'].get("llm_model", "gpt-4o-mini"), 
                                    job_payload['kwargs'].get("llm_temp", 0.3)
                                    )

            function = llm_manager.llm_request

        else:
            oxylabs_manager = OxyLabsManager(app_logger, credential_manager)

            available_functions = {
                "serp_paginator": oxylabs_manager.serp_paginator,
                "get_amazon_product_info": oxylabs_manager.get_amazon_product_info,
                "get_amazon_review": oxylabs_manager.get_amazon_review,
                "web_crawler": oxylabs_manager.web_crawler,
                }
            function = available_functions.get(job_payload.get("function"), None)

        if not function:
            logger.error(f"Function {job_payload.get('function')} not available")
            return None

        filepath = os.path.join(app_logger.session_files_folder(), job_payload.get("input_file",""))
        dataframe = self.dataframe_loader(filepath)
        df_processor= DataFrameProcessor(dataframe)

        batches_constructor = DfBatchesConstructor(df_processor, app_logger)

        try:
            total_rows = len(df_processor.processed_df)
            last_percentage = 0
            current_file_path = os.path.join(self.batches_folder, filename) 
            current_file_path = self.update_payload_status(current_file_path, 0)
            for progress in batches_constructor.df_batches_handler(
                func=function, 
                batch_size=job_payload.get("batch_size"), 
                response_column=job_payload.get("response_column"), 
                query_column=job_payload.get("query_column"), 
                **job_payload.get("kwargs")
            ):
                if isinstance(progress, int):
                    current_percentage = (progress / total_rows) * 100
                    if current_percentage - last_percentage >= 5:
                        status = int(current_percentage)
                        current_file_path = self.update_payload_status(current_file_path, status)
                        logging.info(f"{current_file_path}")
                        last_percentage = current_percentage
                elif isinstance(progress, pd.DataFrame):
                    dir_path, current_filename = os.path.split(current_file_path)
                    self.finalize_job(current_filename, job_payload)
                    logging.info(f"Ended processing executing function")
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")

    def update_payload_status(self, file_path, status):
        if isinstance(status, int):
            new_status = f"PROCESSING_{status}"
        elif status.upper() == "COMPLETED":
            new_status = "COMPLETED"
        else:
            raise ValueError(f"Invalid status: {status}. Must be an integer or 'COMPLETED'.")
        
        # Get the directory and filename separately
        dir_path, filename = os.path.split(file_path)
        
        # Remove any existing status from the filename
        base_name = re.sub(r'(_(PENDING|PROCESSING_\d+|COMPLETED))?\.json$', '', filename)
        
        new_filename = f"{base_name}_{new_status}.json"
        new_path = os.path.join(dir_path, new_filename)
        file_lock = FileLockManager(file_path)

        try:
            file_lock.secure_rename(new_path)
            logger.info(f"Updated {file_path} to {new_filename}")
            return new_path  # Return the full path, not just the filename
        
        except Exception as e:
            logger.error(f"Error updating payload status: {str(e)}")
            return file_path

    def finalize_job(self, payload_filename, job_payload):
        """
        Finalizes the job by updating the batches summary CSV after processing a batch.
        
        Args:
            file_path (str): Path to the processed batch JSON file.
        """
        

        try:

            flattened_data = {
                'filename': payload_filename,
                'status': "COMPLETED"
            }
            for key, value in job_payload.items():
                if key != 'kwargs':
                    if isinstance(value, dict):
                        flattened_data.update({f"{key}_{sub_key}": sub_value for sub_key, sub_value in value.items()})
                    else:
                        flattened_data[key] = value
            self.update_batch_summary(flattened_data, job_payload)
            self.finalize_payload(payload_filename)

        except Exception as e:
            logger.error(f"Failed to finalize job {payload_filename}: {str(e)}")


    def update_batch_summary(self, flattened_data, job_payload):
            # Create a DataFrame from the flattened data
            new_df = pd.DataFrame([flattened_data])
            columns_to_save = [col for col in new_df.columns if not col.startswith('kwargs_')]

            csv_path = os.path.join(
                self.summaries_dir,
                f'{job_payload.get("user_id")}_batches_summary.csv'
            )
            lock_manager = FileLockManager(csv_path)

            if not os.path.exists(csv_path):
                csv_content = new_df[columns_to_save].to_csv(index=False)
                lock_manager.secure_write(csv_content.encode())
                logger.info(f"Created new CSV file: {csv_path}")
            else:
                try:
                    existing_content = lock_manager.secure_read()
                    existing_df = pd.read_csv(csv_path) if existing_content else pd.DataFrame()
                    
                    updated_df = pd.concat([existing_df, new_df[columns_to_save]], ignore_index=True)

                    csv_content = updated_df.to_csv(index=False)
                    lock_manager.secure_write(csv_content.encode())
                except Exception as e:
                    logger.error(f"Error updating existing CSV: {str(e)}")
                    raise


    def finalize_payload(self, payload_filename):
            payload_path = os.path.join(
                self.batches_folder,
                payload_filename
            )

            lock_manager = FileLockManager(payload_path)

            final_payload_path = os.path.join(
                self.completed_dir,
                payload_filename
            )

            lock_manager.secure_move(final_payload_path)

            logger.info(f"Successfully updated and 'payload' to '{final_payload_path}'.")

            # Update the payload status to COMPLETED
            self.update_payload_status(final_payload_path, "COMPLETED")