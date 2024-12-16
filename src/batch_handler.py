import logging
logger = logging.getLogger(__name__)

from src.external_tools import( 
    LlmManager, 
    OxyLabsManager, 
    StopProcessingError, 
    SerpApiManager,
    universal_paginator,
    RetryableError, 
    SkippableError
    )
from src.file_manager import(
    DataFrameProcessor,
    FileLockManager, 
    SessionLogger, 
    FolderSetupMixin, 
    BatchSummaryLogger,
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

    def list_batches_handler(self, 
                             func,  
                             save_interval=5,
                             *args, 
                             **kwargs
                             ):
        pass
        

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
        self.batch_summary_logger = BatchSummaryLogger(self.tool_config)

    def __enter__(self):
        #
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        #
        self.close()

    def close(self):
        # 
        pass

    def read_list_file(self, filepath_toprocess):    #should I move this to a different class?
        """
        Reads a list from a JSON file and returns a list.
        """
        try:
            with open(filepath_toprocess, 'r', encoding='utf-8') as file:
                input_list = json.load(file)
                
            if not isinstance(input_list, list):
                raise ValueError("JSON file must contain a list")
                
            logger.info(f"Successfully loaded list with {len(input_list)} items from {filepath_toprocess}")
            return input_list
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading list file: {e}")
            raise

    def load_payload(self, payload_filename) -> BatchRequestPayload:
        logger.info(f"Attempting to load payload from: {payload_filename}")
        file_path = os.path.join(self.batches_folder, payload_filename)
        logger.info(f"filepath: {file_path}")
        file_content = FileLockManager(file_path).secure_read()
        data = json.loads(file_content)
        logger.info(f"Payload loaded successfully")
        return BatchRequestPayload.from_dict(data)

    def dataframe_loader(self, filepath) -> pd.DataFrame:
        try:
            if filepath.endswith('.csv'):
                return pd.read_csv(filepath)
            elif filepath.endswith(('.xls', '.xlsx')):
                return pd.read_excel(filepath, engine='openpyxl', dtype=str)       
        except Exception as e:
            logger.error(f"Failed to load data: {e}")

    def execute_df_job(self, job_payload, filepath_toprocess, payload_filename, session_logger, selected_function):
        dataframe = self.dataframe_loader(filepath_toprocess)
        df_processor= DataFrameProcessor(dataframe)

        batches_constructor = DfBatchesConstructor(df_processor, session_logger)

        try:
            #total_rows = df_processor.processed_df[job_payload.query_column].notna().sum()
            total_rows = len(df_processor.processed_df)
            current_payload_filename = payload_filename
            self.batch_summary_logger.update_batch_summary(job_payload, status="WIP", total_rows=total_rows)
            current_payload_filename = self.batch_summary_logger.update_payload_status(current_payload_filename, "WIP")
            self.batch_summary_logger.update_wip_progress(job_payload.user_id, job_payload.batch_id, 0)
            for progress in batches_constructor.df_batches_handler(
                func=selected_function, 
                #batch_size=job_payload.batch_size, 
                response_column=job_payload.response_column, 
                query_column=job_payload.query_column, 
                **job_payload.kwargs
            ):
                if progress["df"] is not None:
                    #progress completed
                    self.batch_summary_logger.update_wip_progress(job_payload.user_id, job_payload.batch_id, 100)
                    completed_filename = self.batch_summary_logger.update_payload_status(current_payload_filename, "COMPLETED")
                    self.batch_summary_logger.update_batch_summary(job_payload, status="COMPLETED", filename=completed_filename)
                    
                    logging.info(f"Ended processing executing function")
                    
                elif progress["progress_saved"]:
                    logger.warning(f"Standard progress saved progress = {progress['processed_count']}")
                    #progress file saved but process ongoing
                    if self.check_batch_stop(job_payload, payload_filename, "WIP"):
                        return None
                    current_percentage = (progress["processed_count"] / total_rows) * 100
                    status = int(current_percentage)
                    self.batch_summary_logger.update_wip_progress(job_payload.user_id, job_payload.batch_id, status)
                    
                else:
                    #standard progress received
                    pass

        except Exception as e:
            logger.error(f"An error occurred during list processing: {str(e)}")
            error_filename = self.batch_summary_logger.update_payload_status(current_payload_filename, "FAILED")
            self.batch_summary_logger.update_batch_summary(job_payload, status="FAILED", filename=error_filename)
        
        
    def execute_list_job(self, job_payload, filepath_toprocess, payload_filename, session_logger, selected_function):
        input_list = self.read_list_file(filepath_toprocess)
        expected_total_rows = len(input_list)*job_payload.batch_size
        
        try:
            # Initialize progress tracking
            self.batch_summary_logger.update_batch_summary(job_payload, status="WIP", total_rows=expected_total_rows)
            current_payload_filename = self.batch_summary_logger.update_payload_status(payload_filename, "WIP")
            self.batch_summary_logger.update_wip_progress(job_payload.user_id, job_payload.batch_id, 0)
            
            results_df = pd.DataFrame()  
            
            for query in input_list:
                if self.check_batch_stop(job_payload, payload_filename, "WIP"):
                    return None
                    
                try:
                    for page_results, progress_info in universal_paginator(
                        selected_function,
                        query=query,
                        num_results=job_payload.batch_size,
                        **job_payload.kwargs
                    ):
                        # Convert page_results to DataFrame and append
                        try:
                            page_df = pd.DataFrame(page_results)
                            page_df['query'] = query
                            results_df = pd.concat([results_df, page_df], ignore_index=True)
                            
                            # Calculate progress based on actual results vs expected total
                            current_percentage = min(int((len(results_df) / expected_total_rows) * 100), 99)
                            self.batch_summary_logger.update_wip_progress(
                                job_payload.user_id, 
                                job_payload.batch_id, 
                                current_percentage
                            )
                            
                            # Save progress periodically
                            if len(results_df) % 5 == 0:
                                save_path = session_logger.log_excel(
                                    results_df, 
                                    version="in_progress", 
                                    return_path=True
                                )
                                logger.info(f"Saved progress: {len(results_df)}/{expected_total_rows} rows to {save_path}")
                                
                        except Exception as df_error:
                            logger.error(f"Failed to process results for query '{query}': {str(df_error)}")
                            logger.error(f"Problematic results: {page_results}")
                            continue 
                        
                except Exception as e:
                    logger.error(f"Error processing query '{query}': {str(e)}")
                    
            
            # Save final results and mark as complete
            save_path = session_logger.log_excel(results_df, version="processed", return_path=True)
            self.batch_summary_logger.update_wip_progress(job_payload.user_id, job_payload.batch_id, 100)
            completed_filename = self.batch_summary_logger.update_payload_status(current_payload_filename, "COMPLETED")
            self.batch_summary_logger.update_batch_summary(
                job_payload, 
                status="COMPLETED", 
                filename=completed_filename
            )
            
            logger.info(f"Processing complete! Processed {len(input_list)} queries for a total of {expected_total_rows} results. Saved at {save_path}")
            
        except Exception as e:
            logger.error(f"An error occurred during list processing: {str(e)}")
            error_filename = self.batch_summary_logger.update_payload_status(current_payload_filename, "FAILED")
            self.batch_summary_logger.update_batch_summary(job_payload, status="FAILED", filename=error_filename)

    def execute_job(self,payload_filename, credential_manager):

        try:
            logger.info(f"Attempting to load payload: {payload_filename}")
            job_payload = self.load_payload(payload_filename)
        except Exception as e:
            logger.error(f"Failed to load payload: {e}")
            return None
        
        if self.check_batch_stop(job_payload, payload_filename, "PENDING"):
            return None
        
        session_logger = SessionLogger(job_payload.user_id,self.tool_config)
        session_logger.session_id = job_payload.session_id
        session_logger.file_name = job_payload.input_file

        #create managers
        serpapi_manager = SerpApiManager(session_logger, credential_manager)
        oxylabs_manager = OxyLabsManager(session_logger, credential_manager)
        llm_manager = LlmManager(
            session_logger, 
            credential_manager, 
            job_payload.kwargs.get("llm_model", "gpt-4o-mini"), 
            job_payload.kwargs.get("llm_temp", 0.3)
        )

        # Map function names to their implementations
        available_functions = {
            "llm_request": llm_manager.llm_request,
            "serpapi_serp_crawler": serpapi_manager.serpapi_serp_crawler,
            "oxylab_serp_crawler": oxylabs_manager.oxylab_serp_crawler,
            "get_amazon_product_info": oxylabs_manager.get_amazon_product_info,
            "get_amazon_review": oxylabs_manager.get_amazon_review,
            "web_crawler": oxylabs_manager.web_crawler,
            "serpapi_review_crawler": serpapi_manager.serpapi_review_crawler,
            "serpapi_maps_reviews_crawler": serpapi_manager.serpapi_maps_reviews_crawler,
        }        
        
        list_supported_functions = {
            "serpapi_serp_crawler",
            "oxylab_serp_crawler",
            "get_amazon_review",
            "serpapi_review_crawler",
            "serpapi_maps_reviews_crawler",
            "get_google_product_info",
            "get_amazon_product_info",
        }
        
        selected_function = available_functions.get(job_payload.function)

        if not selected_function:
            logger.error(f"Function {job_payload.function} not available")
            return None

        filepath_toprocess = os.path.join(session_logger.session_files_folder(), job_payload.input_file)

        
        if job_payload.type == "df":
            self.execute_df_job(job_payload, filepath_toprocess, payload_filename, session_logger, selected_function)
   
        elif job_payload.type == "list" and job_payload.function in list_supported_functions:
            self.execute_list_job(job_payload, filepath_toprocess, payload_filename, session_logger, selected_function)
        
        else:            
            logger.error(f"Function {job_payload.function} is not supported for {job_payload.type } type processing")
            return None

    def check_batch_stop(self, job_payload, payload_filename, status):

        stop_check_filename = job_payload.user_id + "_" + job_payload.batch_id + "_STOP"
        if stop_check_filename in self.batch_summary_logger.check_stop_requests():
            logger.info(f"Stop request found for batch {job_payload.batch_id}")
            if status == "PENDING":
                completed_filename = self.batch_summary_logger.update_payload_status(payload_filename, "CANCELLED")
                self.batch_summary_logger.update_batch_summary(job_payload, status="CANCELLED", filename=completed_filename)
            elif status == "WIP":
                completed_filename = self.batch_summary_logger.update_payload_status(payload_filename, "STOPPED")
                self.batch_summary_logger.update_batch_summary(job_payload, status="STOPPED", filename=completed_filename)
            self.batch_summary_logger.handled_stop_request(job_payload.user_id, job_payload.batch_id)
            return True
        else: 
            return False
