from src.external_tools import LlmManager, StopProcessingError, RetryableError, SkippableError, openai_advanced_uses, OxyLabsManager
import streamlit as st
from src.file_manager import DataFrameProcessor, FileLockManager, AppLogger
import pandas as pd
import json
import os
from src.prompts import sysmsg_keyword_categorizer,sysmsg_review_analyst_template, sysmsg_summarizer_template


class StreamlitProgressHandler:
    def __init__(self, total_rows):
        self.total_rows = total_rows
        self.progress_bar = st.progress(0, text="Preparing the operation")

    def update(self, processed_count):
        progress_percent = int(processed_count / self.total_rows * 100)
        self.progress_bar.progress(progress_percent, text=f"Processing... {processed_count}/{self.total_rows}")

    def finalize(self):
        self.progress_bar.empty()
        st.success('Processing complete! All rows have been processed.')


class StandardProgressHandler:
    def __init__(self, total_rows):
        self.total_rows = total_rows
        self.processed_count = 0

    def update(self, processed_count):
        self.processed_count = processed_count
        #print(f"Processed {self.processed_count}/{self.total_rows} rows", end='\r')

    def finalize(self):
        #print(f"\nProcessing complete! All {self.total_rows} rows have been processed.")
        pass

class DfBatchesConstructor():
    def __init__(self, df_processor:DataFrameProcessor, app_logger=None):
        """
        Initialize the batches constructor
        """

        self.df_processor = df_processor
        self.app_logger = app_logger

    
    def batch_requests(self, 
                       func, 
                       response_column, 
                       query_column, 
                       batch_size=0, 
                       force_string=True, 
                       max_retries=3, 
                       retry_delay=1, 
                       save_interval=5,
                       streamlit=True, 
                       *args, 
                       **kwargs):
        if query_column not in self.df_processor.processed_df.columns:
            raise StopProcessingError(f"Column '{query_column}' not found in DataFrame. Available columns are: {', '.join(self.df_processor.processed_df.columns)}")       
        self._prepare_dataframe(response_column)
        valid_indices = self._get_valid_indices(response_column)
        total_rows = len(valid_indices)
        #print(f"total rows to process: {total_rows}, batch size: {batch_size}")

        progress_handler = self._get_progress_handler(streamlit, total_rows)

        processed_count = 0
        for batch in self._process_batches(valid_indices, batch_size):
            batch_progress = self._process_batch(batch, func, response_column, query_column, 
                                                 force_string, max_retries, retry_delay, 
                                                 save_interval, total_rows, progress_handler, 
                                                 *args, **kwargs)
            processed_count += len(batch)
            yield f"Processed {processed_count} out of {total_rows} rows"
            
            if batch_size > 0 and processed_count >= batch_size:
                #print(f"Reached batch size limit of {batch_size}")
                break
        print ("finalizing processing")
        self._finalize_processing(progress_handler)
        yield self.df_processor.processed_df

    def _prepare_dataframe(self, response_column):
        if response_column not in self.df_processor.processed_df.columns:
            self.df_processor.processed_df[response_column] = pd.NA

    def _get_valid_indices(self, response_column):
        mask = self.df_processor.processed_df[response_column].isna()
        return self.df_processor.processed_df[mask].index

    def _get_progress_handler(self, use_streamlit, total_rows):
        if use_streamlit:
            return StreamlitProgressHandler(total_rows)
        else:
            return StandardProgressHandler(total_rows)

    def _process_batches(self, valid_indices, batch_size):
        start_idx = 0
        while start_idx < len(valid_indices):
            end_idx = start_idx + batch_size if batch_size > 0 else len(valid_indices)
            yield valid_indices[start_idx:end_idx]
            start_idx = end_idx
            if batch_size > 0 and start_idx >= batch_size:
                break

    def _process_batch(self, batch, func, response_column, query_column, 
                       force_string, max_retries, retry_delay, 
                       save_interval, total_rows, progress_handler, 
                       *args, **kwargs):
        for i, idx in enumerate(batch, 1):
            if not self._process_row(idx, func, response_column, query_column, 
                                     force_string, max_retries, retry_delay, 
                                     *args, **kwargs):
                return i

            progress_handler.update(i)
            if i % save_interval == 0:
                self.app_logger.log_excel(self.df_processor.processed_df, version="batch_processing")

        return len(batch)

    def _process_row(self, idx, func, response_column, query_column, 
                     force_string, max_retries, retry_delay, 
                     *args, **kwargs):    
        try:
            if idx not in self.df_processor.processed_df.index:
                raise ValueError(f"Index {idx} not found in DataFrame")
            if query_column not in self.df_processor.processed_df.columns:
                raise ValueError(f"Column '{query_column}' not found in DataFrame")
            
            value = self.df_processor.processed_df.loc[idx, query_column]
            
            if pd.isna(value):
                raise SkippableError(f"Value is NaN or None for row {idx}")
            
            result = func(value, *args, **kwargs)
            
            if force_string:
                result = str(result) if result is not None else ""
            
            self.df_processor.processed_df.at[idx, response_column] = result
            return True
        except SkippableError as e:
            self.df_processor.processed_df.at[idx, response_column] = str(e)
            print(f"Skipped row {idx}: {e}")
        except Exception as e:
            self.df_processor.processed_df.at[idx, response_column] = f"Error: {e}"
            print(f"Error processing row {idx}: {e}")
        return True


    def _finalize_processing(self, progress_handler):
        self.app_logger.log_excel(self.df_processor.processed_df, version="processed")
        progress_handler.finalize()


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

        # Load only the 'filename' column from existing CSV or create an empty set if it doesn't exist
        if os.path.exists(csv_path):
            processed_files = set(pd.read_csv(csv_path, usecols=['filename'])['filename'])
        else:
            processed_files = set()

        # Get the set of batch files
        batch_files = set(os.listdir(batches_folder))

        # List to store new batch data
        new_batches = []

        # Process only new completed batches
        for batch_file in batch_files:
            if batch_file.endswith('.json') and batch_file not in processed_files:
                file_path = os.path.join(batches_folder, batch_file)
                lock_manager = FileLockManager(file_path)
                
                try:
                    json_data = lock_manager.secure_read()
                    batch_data = json.loads(json_data)
                    
                    # Extract status from filename
                    status = batch_file.split('_')[-1].split('.')[0]
                    
                    # Flatten nested dictionaries (e.g., kwargs), excluding 'kwargs' itself
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
                    print(f"Error reading JSON file: {batch_file}")
                except Exception as e:
                    print(f"Error processing file {batch_file}: {str(e)}")
        # Convert new batches to DataFrame
        new_df = pd.DataFrame(new_batches)

        # Append new data to CSV if there are new batches
        if not new_df.empty:
            # Ensure we're not writing any 'kwargs' columns
            columns_to_save = [col for col in new_df.columns if not col.startswith('kwargs_')]
            new_df[columns_to_save].to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
            print(f"Batches summary updated and saved to {csv_path}")
            print(f"Added {len(new_batches)} new completed batches.")
        else:
            print("No new batches to add.")


    def load_payload(self, payload_filename):
        file_path = os.path.join(self.batches_folder, payload_filename)
        file_content = FileLockManager(file_path).secure_read()
        return json.loads(file_content)

    def rename_completed_payload(self, payload_filename):
        os.rename(os.path.join(self.batches_folder, payload_filename), os.path.join(self.batches_folder, payload_filename.replace("PENDING", "COMPLETED")))

    def dataframe_loader(self, filepath) -> pd.DataFrame:
        try:
            if filepath.endswith('.csv'):
                return pd.read_csv(filepath)
            elif filepath.endswith(('.xls', '.xlsx')):
                return pd.read_excel(filepath, engine='openpyxl', dtype=str)       
        except Exception as e:
            print(f"Failed to load data: {e}")

    def execute_job(self,job_payload, credential_manager):

        use_streamlit = False
        app_logger = AppLogger(job_payload.get("user_id"),self.tool_config, use_streamlit=use_streamlit)
        app_logger.session_id = job_payload.get("session_id")
        app_logger.file_name = job_payload.get("input_file")

        if job_payload.get("function") == "llm_request":
            llm_manager = LlmManager(job_payload['kwargs'].get("llm_model", "gpt-4o-mini"), 
                                    app_logger, 
                                    credential_manager, 
                                    self.tool_config, 
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
        dataframe = self.dataframe_loader(filepath)
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