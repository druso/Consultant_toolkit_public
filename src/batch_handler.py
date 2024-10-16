import logging
logger = logging.getLogger(__name__)

from src.external_tools import LlmManager, OxyLabsManager, StopProcessingError, RetryableError, SkippableError
from src.file_manager import DataFrameProcessor, FileLockManager, AppLogger
import pandas as pd
import json
import os
import time
from src.prompts import sysmsg_keyword_categorizer,sysmsg_review_analyst_template, sysmsg_summarizer_template


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

    def update_batches_summary(self):
        """
        Updates the batches summary CSV by only processing new completed batches.
        """
        folder = self.tool_config['shared_folder']
        batches_folder = os.path.join(folder, 'batches')
        csv_path = os.path.join(folder, 'batches_summary.csv')

        if os.path.exists(csv_path):
            processed_files = set(pd.read_csv(csv_path, usecols=['filename'])['filename'])
        else:
            processed_files = set()

        batch_files = set(os.listdir(batches_folder))

        new_batches = []

        for batch_file in batch_files:
            if batch_file.endswith('.json') and batch_file not in processed_files:
                file_path = os.path.join(batches_folder, batch_file)
                lock_manager = FileLockManager(file_path)
                
                try:
                    json_data = lock_manager.secure_read()
                    batch_data = json.loads(json_data)
                    
                    status = batch_file.split('_')[-1].split('.')[0]
                    
                    flattened_data = {
                        'filename': batch_file,
                        'status': status
                    }
                    for key, value in batch_data.items():
                        if key != 'kwargs':
                            if isinstance(value, dict):
                                flattened_data.update({f"{key}_{sub_key}": sub_value for sub_key, sub_value in value.items()})
                            else:
                                flattened_data[key] = value
                    
                    new_batches.append(flattened_data)
                except json.JSONDecodeError:
                    logger.error(f"Error reading JSON file: {batch_file}")
                except Exception as e:
                    logger.error(f"Error processing file {batch_file}: {str(e)}")
        new_df = pd.DataFrame(new_batches)

        if not new_df.empty:
            columns_to_save = [col for col in new_df.columns if not col.startswith('kwargs_')]
            new_df[columns_to_save].to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
            logger.info(f"Batches summary updated and saved to {csv_path}")
            logger.info(f"Added {len(new_batches)} new batches.")
        else:
            logger.info("No new batches to add.")


    def load_payload(self, payload_filename):
        file_path = os.path.join(self.batches_folder, payload_filename)
        file_content = FileLockManager(file_path).secure_read()
        return json.loads(file_content)

    def rename_completed_payload(self, payload_filename):
        os.rename(os.path.join(self.batches_folder, payload_filename), os.path.join(self.batches_folder, payload_filename.replace("PENDING", "COMPLETED")))
        logger.info(f"Updated {payload_filename} to COMPLETED")

    def dataframe_loader(self, filepath) -> pd.DataFrame:
        try:
            if filepath.endswith('.csv'):
                return pd.read_csv(filepath)
            elif filepath.endswith(('.xls', '.xlsx')):
                return pd.read_excel(filepath, engine='openpyxl', dtype=str)       
        except Exception as e:
            logger.error(f"Failed to load data: {e}")

    def execute_job(self,job_payload, credential_manager):

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
            for progress in batches_constructor.df_batches_handler(
                func=function, 
                batch_size=job_payload.get("batch_size"), 
                response_column=job_payload.get("response_column"), 
                query_column=job_payload.get("query_column"), 
                **job_payload.get("kwargs")
            ):
                if isinstance(progress, int):
                    logging.info(f"Processed {progress} rows")
                elif isinstance(progress, pd.DataFrame):
                    logging.info(f"Ended processing executing function")
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")