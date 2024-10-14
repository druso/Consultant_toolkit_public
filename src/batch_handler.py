from src.external_tools import LlmManager, StopProcessingError, RetryableError, SkippableError, openai_advanced_uses
import streamlit as st
from src.file_manager import DataFrameProcessor
import pandas as pd
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