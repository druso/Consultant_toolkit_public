
from io import BytesIO, StringIO
import pandas as pd
import json
import textract
import os
from datetime import datetime
import uuid
import zipfile
import time
from typing import Tuple, Any, Optional, List
import shutil

def import_streamlit():
    import streamlit as st
    return st

class FileLockManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.lock_file = file_path + ".lock"

    def _acquire_lock(self):
        """Try to acquire a lock by creating a lock file."""
        while True:
            try:
                # Try to create a lock file exclusively
                fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                # Close the file descriptor to release it after lock acquisition
                os.close(fd)
                #print(f"Lock acquired on {self.lock_file}")
                return True
            except FileExistsError:
                # If the lock file already exists, wait and retry
                #print(f"Lock already exists. Waiting to acquire lock on {self.lock_file}...")
                time.sleep(1)  # Wait before retrying

    def _release_lock(self):
        """Release the lock by deleting the lock file."""
        if os.path.exists(self.lock_file):
            os.remove(self.lock_file)
            #print(f"Lock released on {self.lock_file}")
        #else:
            #print(f"No lock to release on {self.lock_file}")

    def secure_write(self, data):
        """Automatically acquire and release the lock for binary writing."""
        self._acquire_lock()
        try:
            with open(self.file_path, "wb") as file:
                file.write(data)
            #print(f"Finished writing to {self.file_path}")
        finally:
            self._release_lock()

    def secure_read(self):
        """Automatically acquire and release the lock for reading the file as raw text."""
        self._acquire_lock()
        try:
            with open(self.file_path, "r") as file:
                data = file.read()  # Reads the file as raw text
            #print(f"Finished reading from {self.file_path}")
            return data
        finally:
            self._release_lock()


class DataFrameProcessor:
    def __init__(self, df: pd.DataFrame, use_streamlit=True):
        self.use_streamlit = use_streamlit
        self.user_df = df.copy()
        self.processed_df = df.copy()
        if self.use_streamlit:
            self.st = import_streamlit()
            self._initialize_streamlit()

    def _initialize_streamlit(self):
        if 'processed_df' not in self.st.session_state:
            self.st.session_state['processed_df'] = self.processed_df

        self.processed_df = self.st.session_state['processed_df']

        if 'available_columns' not in self.st.session_state:
            self.st.session_state['available_columns'] = self.processed_df.columns.tolist()

        self._add_streamlit_buttons()

    def _add_streamlit_buttons(self):
        if self.st.sidebar.button("Refresh columns list", use_container_width=True, 
                                  help="Will force the refresh of the lists of available columns throughout the application"):
            self.update_columns_list()

        if self.st.sidebar.button('Restore Processing', use_container_width=True, 
                                  help="Will restore the file to the originally uploaded file"):
            self.st.session_state["app_logger"].reinitialize_session_id()
            self.st.session_state["app_logger"].log_excel(self.user_df, "original")
            self.reset_to_original()

    def update_columns_list(self):
        if self.use_streamlit:
            self.st.session_state['available_columns'] = self.processed_df.columns.tolist()
    def unroll_json(self, column_name: str) -> pd.DataFrame:
        """
        Unrolls (flattens) JSON data within a specified column of the DataFrame.
        """
        import ast

        # Helper function to safely load JSON
        def safe_json_loads(val):
            if isinstance(val, str):
                try:
                    return ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    return val  # Return the original value if parsing fails
            else:
                return val

        # Make a copy of the original column to preserve the original data
        self.processed_df[column_name + '_original'] = self.processed_df[column_name]

        # Apply the safe JSON loading to the specified column
        self.processed_df[column_name] = self.processed_df[column_name].apply(safe_json_loads)

        # Helper function to explode and normalize data
        def explode_and_normalize(data):
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.json_normalize(data)
            else:
                return None

        result_rows = []
        for index, row in self.processed_df.iterrows():
            json_data = row[column_name]

            # Check if any value within the JSON data is a list or dict
            expanded_data = explode_and_normalize(json_data)
            if expanded_data is not None:
                for _, expanded_row in expanded_data.iterrows():
                    new_row = row.to_dict()  # Keep all original columns
                    # Optionally add a prefix to expanded columns to avoid conflicts
                    # expanded_row = expanded_row.add_prefix('expanded_')
                    new_row.update(expanded_row.to_dict())  # Update the row with exploded data
                    result_rows.append(new_row)
            else:
                # Straight values are left as-is
                result_rows.append(row.to_dict())

        result_df = pd.DataFrame(result_rows)

        # Ensure consistent type across all columns (convert lists and non-lists to lists)
        for col in result_df.columns:
            if any(isinstance(i, list) for i in result_df[col]):
                result_df[col] = result_df[col].apply(lambda x: x if isinstance(x, list) else [x])

        self.processed_df = result_df
        return self.processed_df

    def try_parse_column(self, column_name: str) -> pd.Series:
        """
        Attempts to parse a column into different data types.
        """
        series = self.processed_df[column_name]

        # Try parsing as JSON first
        if series.dtype == 'object':
            try:
                return series.apply(json.loads)
            except (ValueError, json.JSONDecodeError):
                pass

        # Try parsing as datetime
        try:
            return pd.to_datetime(series, errors='coerce')
        except (ValueError, TypeError):
            pass

        # Try parsing as numeric
        try:
            return pd.to_numeric(series, errors='coerce')
        except ValueError:
            pass

        # Try parsing as list (assuming string representation of lists)
        if series.dtype == 'object':
            try:
                return series.apply(eval)
            except:
                pass

        # If all parsing attempts fail, return the original series
        return series

    def get_data_type(self, column_name: str) -> str:
        """
        Returns the data type of a column as a string.
        """
        series = self.processed_df[column_name]
        if pd.api.types.is_datetime64_any_dtype(series):
            return f"DateTime ({series.dtype})"
        elif pd.api.types.is_numeric_dtype(series):
            return f"Numeric ({series.dtype})"
        elif series.dtype == 'object':
            if series.apply(lambda x: isinstance(x, dict)).any():
                return "JSON"
            elif series.apply(lambda x: isinstance(x, list)).any():
                return "List"
        return f"String/Other ({series.dtype})"

    def parse_column(self, column_name: str, expected_type: str) -> None:
        """
        Parses a column based on the expected type and updates the DataFrame.
        The original column is backed up as 'column_name_backup'.
        """
        original_type = self.get_data_type(column_name)
        series = self.processed_df[column_name]

        # Back up the original column if not already backed up
        backup_column_name = column_name + '_backup'
        if backup_column_name not in self.processed_df.columns:
            self.processed_df[backup_column_name] = series.copy()

        if expected_type == "Numeric":
            parsed_series = pd.to_numeric(series, errors='coerce')
        elif expected_type == "DateTime":
            parsed_series = pd.to_datetime(series, errors='coerce')
        elif expected_type == "JSON":
            def safe_json_loads(val):
                try:
                    return json.loads(val) if isinstance(val, str) else val
                except json.JSONDecodeError:
                    return None  # Handle invalid JSON by returning None
            parsed_series = series.apply(safe_json_loads)
        elif expected_type == "List":
            def safe_eval(val):
                try:
                    return eval(val) if isinstance(val, str) else val
                except:
                    return None  # Handle invalid eval by returning None
            parsed_series = series.apply(safe_eval)
        else:
            parsed_series = series  # Keep as is for "String/Other"

        # Update the column with parsed data
        self.processed_df[column_name] = parsed_series

        new_type = self.get_data_type(column_name)

        if original_type != new_type:
            print(f"Column '{column_name}' has been parsed from {original_type} to {new_type}")
        else:
            print(f"No change in data type. Column '{column_name}' remains as {new_type}")


    def generate_column_summary(self, df):
        summary = []
        for col in df.columns:
            col_type = df[col].dtype
            null_count = df[col].isnull().sum()
            null_percentage = (null_count / len(df)) * 100
            
            try:
                unique_count = df[col].nunique()
            except:
                unique_count = "N/A (unhashable type)"
            
            sample_value = df[col].iloc[0] if not df[col].empty else None
            sample_type = type(sample_value).__name__
            
            if pd.api.types.is_numeric_dtype(df[col]):
                summary_dict = {
                    'Column': col,
                    'Type': f"{col_type} ({sample_type})",
                    'Unique Values': unique_count,
                    'Null Count': null_count,
                    'Null %': f"{null_percentage:.2f}%",
                    'Min': df[col].min(),
                    'Max': df[col].max(),
                    'Mean': df[col].mean(),
                    'Median': df[col].median()
                }
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                summary_dict = {
                    'Column': col,
                    'Type': f"{col_type} ({sample_type})",
                    'Unique Values': unique_count,
                    'Null Count': null_count,
                    'Null %': f"{null_percentage:.2f}%",
                    'Min Date': df[col].min(),
                    'Max Date': df[col].max()
                }
            else:
                summary_dict = {
                    'Column': col,
                    'Type': f"{col_type} ({sample_type})",
                    'Unique Values': unique_count,
                    'Null Count': null_count,
                    'Null %': f"{null_percentage:.2f}%",
                    'Sample Value': str(sample_value)[:50] + ('...' if len(str(sample_value)) > 50 else '')
                }
            
            summary.append(summary_dict)
        
        return pd.DataFrame(summary)


    def reset_to_original(self):
        """
        Resets the processed DataFrame to the original state.
        """
        self.processed_df = self.user_df.copy()
        if self.use_streamlit:
            self.st.session_state['processed_df'] = self.processed_df
            self.update_columns_list()

    


class AppLogger() :
    """
    Handles most of I/O operations for storing the activities
    """
    def __init__(self, user_id: str, tool_config=None, use_streamlit=True):
        """Initializes the logger with a session ID and default values."""
        self.user_id = user_id
        self.session_id = datetime.now().strftime('%y%m%d') +"_"+ str(uuid.uuid4())[:6]
        self.file_name = "default"
        if use_streamlit:
            self.st = import_streamlit()
            self.tool_config = self.st.session_state.get('tool_config', {})
            logs_root_folder = os.getenv('LOGS_ROOT_FOLDER') or self.st.session_state.get('tool_config', {}).get('logs_root_folder') or "logs"
        else: 
            logs_root_folder = os.getenv('LOGS_ROOT_FOLDER') or tool_config.get('logs_root_folder') or "logs"
            self.tool_config = tool_config
        self.logs_folder = f"{logs_root_folder}/{user_id}"
        self.log_types = ["files", "requests", "openai_threads"]
        
        for log_type in self.log_types:
            folder = os.path.join(self.logs_folder, log_type)
            os.makedirs(folder, exist_ok=True)
            setattr(self, f"{log_type}_folder", folder)

    def reinitialize_session_id(self):
        self.session_id = datetime.now().strftime('%y%m%d') +"_"+ str(uuid.uuid4())[:6]

    def set_file_name(self, file_name: str):
        self.file_name = file_name 

    def session_files_folder(self):
        folder = os.path.join(self.files_folder, self.session_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    
    def save_original_file(self, uploaded_file, force_save=False):
        folder = self.session_files_folder()
        os.makedirs(folder, exist_ok=True)

        def save_single_file(file, index=None):
            if index is not None:
                file_name = f"original_{index+1}_{file.name}"
            else:
                file_name = f"original_{file.name}"
            file_path = os.path.join(folder, file_name)
            
            if force_save or not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    FileLockManager(file_path).secure_write(file.getbuffer())
                print(f"Saved original file {file_path}")
            else:
                #print(f"File {file_path} already exists. Skipping.")
                pass

        if isinstance(uploaded_file, list):
            for index, file in enumerate(uploaded_file):
                save_single_file(file, index)
            self.file_name = "multiple_files"
        else:
            save_single_file(uploaded_file)
            self.file_name = uploaded_file.name

    
    def list_subfolders(self, folder_path: str) -> List[str]:
        return [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    def zip_directory(self, folder_path: str) -> BytesIO:
        byte_io = BytesIO()
        with zipfile.ZipFile(byte_io, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))
        byte_io.seek(0)
        return byte_io
    
    def purge_logs_folder(self):
        folder = self.logs_folder
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

    def to_excel(self, df: pd.DataFrame) -> bytes:
 
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        # No need to call writer.save()
        processed_data = output.getvalue()
        return processed_data


    def log_excel(self, df: pd.DataFrame, version: str = "processed", return_path = False, return_filename=False) -> None:
        """
        Saves a Pandas DataFrame to an Excel file within the session's logs folder.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            version (str, optional): A string to include in the filename, indicating
                the data version (e.g., "raw", "processed"). Defaults to "processed".
            return_path: if True the function returns the path where the file was saved
        """
        folder = self.session_files_folder()
        os.makedirs(folder, exist_ok=True)
        filename = f"{version}_{self.file_name.split('.', 1)[0]}.xlsx"
        path=f"{folder}/{filename}"
        with open(path, "wb") as f:
            #f.write(self.to_excel(df))
            FileLockManager(path).secure_write(self.to_excel(df))
        print (f"saved excel log at {path}")
        if return_path:
            return path
        if return_filename:
            return filename

    def save_request_log(self, response: dict, service: str, calltype: str) -> None:
        """
        Saves the response from an API request in either JSON or TXT format.

        Args:
            response (dict): The response data from the API call.
            service (str): The name of the service or API being called.
            calltype (str): A descriptor for the type of API call (e.g., "query", "update").
        """
        folder = os.path.join(self.requests_folder, service)
        os.makedirs(folder, exist_ok=True)

        timestamp = datetime.now().strftime('%m%d%H%M%S')

        max_length = 50 
        filename = f"{calltype}_{timestamp}"
        cleaned_filename = filename[:max_length]
        cleaned_filename = ''.join(c for c in cleaned_filename if c.isalnum() or c in ('_', '-'))

        if cleaned_filename != filename:
            print(f"Filename cleaned from '{filename}' to '{cleaned_filename}'")

        try:
            # Attempt to save as JSON
            with open(f"{folder}/{cleaned_filename}.json", "w") as f:
                json.dump(response, f, indent=4)
            print (f"saved json log at {folder}/{cleaned_filename}.json")
        except TypeError:
            # If JSON serialization fails, save as text
            with open(f"{folder}/{cleaned_filename}.txt", "w") as f:
                f.write(str(response))

    def log_openai_thread(self, thread_id: str, thread_name: str, thread_file_ids: List[str], thread_history: List[dict], user_setup_msg: str, assistant_setup_msg:str):
        thread_file = {
            "thread_id":thread_id,
            "thread_name":thread_name,
            "file_ids":thread_file_ids,
            "user_setup_msg":user_setup_msg,
            "assistant_setup_msg":assistant_setup_msg,
            "thread_history":thread_history,
        }
        
        with open(f"{self.openai_threads_folder}/{thread_id}.json", "w") as f:
            json.dump(thread_file, f, indent=4)


    def list_openai_threads(self):
        """
        Lists all OpenAI threads stored in the openai_threads_folder.

        Returns:
            List[Tuple[str, str]]: A list of tuples, each containing (thread_id, thread_name).
        """
        thread_list = []
        for filename in os.listdir(self.openai_threads_folder):
            file_path = os.path.join(self.openai_threads_folder, filename)
            if os.path.isfile(file_path) and filename.endswith('.json'):
                thread_name=""
                thread_id=""
                try:
                    with open(file_path, 'r') as f:
                        thread_data = json.load(f)
                        thread_name = thread_data.get('thread_name', thread_name)
                        thread_id = thread_data.get('thread_id', thread_id)
                    if id: 
                        thread_list.append((thread_id, thread_name))
                except json.JSONDecodeError:
                    print(f"Error reading JSON file: {filename}")
                except Exception as e:
                    print(f"Error processing file {filename}: {str(e)}")
 
        return thread_list
    
    def get_thread_info(self, thread_id):
        file_path = os.path.join(self.openai_threads_folder, thread_id + ".json")
        try:
            with open(file_path, 'r') as f:
                thread_data = json.load(f)
                return thread_data
        except:
            if self.use_streamlit:  
                self.st.error(f"thread log for thread id: {thread_id} does not exist")
            else:
                print(f"thread log for thread id: {thread_id} does not exist")

    def get_thread_history(self, thread_id):
        thread_file = self.get_thread_info(thread_id)
        return thread_file.get("thread_history", [])
    
    def update_thread_history(self,thread_history,thread_id):
        thread_file = self.get_thread_info(thread_id)
        thread_file["thread_history"] = thread_history
        with open(f"{self.openai_threads_folder}/{thread_id}.json", "w") as f:
            json.dump(thread_file, f, indent=4)

    def erase_thread_data(self, thread_id):
        file_path = os.path.join(self.openai_threads_folder, thread_id + ".json")
        try:
            os.remove(file_path)
            if self.use_streamlit:  
                self.st.success(f"Thread data for thread id: {thread_id} has been deleted.")
            else:
                print(f"Thread data for thread id: {thread_id} has been deleted.")
        except FileNotFoundError:
            print(f"Thread log for thread id: {thread_id} does not exist.")
        except Exception as e:
            print(f"Error deleting thread log: {e}")

    def batches_list(self):
        folder = self.tool_config.get('shared_folder', 'shared')
        batches_folder = os.path.join(folder, 'batches')
        return os.listdir(batches_folder)
        
    def post_scheduler_request(self,df, function, query_column, response_column, kwargs):
        
        batches_list = self.batches_list()
        batch_id = "batch_" + str(len(batches_list) + 1).zfill(3)

        input_file = self.log_excel( df, "batch", return_path = False, return_filename=True) 
        function_name = function.__name__ if callable(function) else function
        processed_kwargs = {}
        for key, value in kwargs.items():
            if callable(value):
                processed_kwargs[key] = value.__name__  # Store function name as string
            elif isinstance(value, (int, float, str, bool, type(None))):
                processed_kwargs[key] = value  # These types are JSON serializable
            else:
                processed_kwargs[key] = str(value)  # Convert other types to string
        payload={
            "batch_id": batch_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "function": function_name,
            "batch_size": 10,
            "input_file": input_file,
            "query_column": query_column,
            "response_column": response_column,
            "kwargs": processed_kwargs
                }
        folder= self.tool_config.get('shared_folder', 'shared')
        with open(f"{folder}/batches/{self.user_id}_{batch_id}_PENDING.json", "w") as f:
            json.dump(payload, f, indent=4)
            print(f"saved batch request at {folder}/batches/{batch_id}_PENDING.json")



    def load_batches_summary(self, csv_path):
        """Load the batches summary CSV file using FileLockManager."""
        lock_manager = FileLockManager(csv_path)
        try:
            csv_content = lock_manager.secure_read()
            return pd.read_csv(StringIO(csv_content))
        except FileNotFoundError:
            print(f"CSV file not found at {csv_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            return pd.DataFrame()

    def streamlit_batches_status(self):
        self.st.header("Batch Processing Status")
        
        csv_path = os.path.join(self.tool_config['shared_folder'], 'batches_summary.csv')
        
        # Load the CSV file
        batches_df = self.load_batches_summary(csv_path)
        if  self.st.sidebar.button("update batch list", use_container_width=True):
            batches_df = self.load_batches_summary(csv_path)
        
        if batches_df.empty:
            self.st.warning("No batch data available.")
            return
        
        batches_df = batches_df[batches_df['user_id'] == self.user_id]
        # Display recap of batches in the pipeline
        self.st.subheader("Batches Recap")
        total_batches = len(batches_df)
        completed_batches = len(batches_df[batches_df['status'] == 'COMPLETED'])
        pending_batches = len(batches_df[batches_df['status'] == 'PENDING'])
        
        col1, col2, col3 = self.st.columns(3)
        col1.metric("Total Batches", total_batches)
        col2.metric("Completed Batches", completed_batches)
        col3.metric("Pending Batches", pending_batches)

        # Display the DataFrame
        self.st.subheader("Batches Data")
        
        columns_to_display = ['filename', 'status', 'batch_id', 'session_id', 'function']
        
        self.st.dataframe(batches_df[columns_to_display])

        # Add a select box for completed batches
        completed_batches = batches_df[batches_df['status'] == 'COMPLETED']
        if not completed_batches.empty:
            selected_filename = self.st.selectbox(
                "Select a completed batch to download:",
                options=completed_batches['filename'].tolist(),
                #format_func=lambda x: f"{x} - {completed_batches[completed_batches['filename'] == x]['batch_id'].values[0]}"
            )

            if selected_filename:
                selected_row = completed_batches[completed_batches['filename'] == selected_filename].iloc[0]
                
                # Construct the path to the output file
                filename = f"processed_{selected_row['input_file']}"
                
                output_file_path = os.path.join(self.tool_config['logs_root_folder'], f"{self.user_id}","files", f"{selected_row['session_id']}", filename)


                # Check if the output file exists
                if os.path.isfile(output_file_path):
                    with open(output_file_path, "rb") as file:
                        self.st.download_button(
                            label="Download Output File",
                            data=file,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    self.st.warning("Output file not found for the selected batch.")
        else:
            self.st.info("No completed batches available for download.")

class DataLoader:
    """
    Handles the loading of file into the application
    """
    def __init__(self, config_key: str, app_logger:AppLogger, use_streamlit=True):

        self.configurations = {
            "dataframe": {'extensions': ['csv', 'xls', 'xlsx'], 
                          'handler': self._dataframe_handler, 
                          'default':pd.DataFrame()},
            "doc": {'extensions': ['pdf', 'docx', 'txt'], 
                     'handler': self._doc_handler, 
                     'default': "",
                     'accept_multiple_file':True},
            "audio": {'extensions': ['mp3','mp4', 'mpeg','mpga','m4a','wav','webm'], 
                     'handler': None, 
                     'default':""}
        }
        if use_streamlit:
            self.st = import_streamlit()
        if config_key not in self.configurations:
            raise ValueError("Invalid configuration key.")
        self.app_logger = app_logger
        self.file_types = self.configurations[config_key]['extensions']
        self.handler = self.configurations[config_key]['handler']
        self.default = self.configurations[config_key]['default']
        self.accept_multiple_files =  self.configurations[config_key].get('accept_multiple_file', False)
        if use_streamlit:
            uploaded_file = self.st.sidebar.file_uploader("Choose a file", type=self.file_types, accept_multiple_files=self.accept_multiple_files)
        if uploaded_file:
            self.user_file = self.load_user_file(uploaded_file)
            self.app_logger.save_original_file(uploaded_file)
            
        else:
            self.user_file = None
            self.content = self.default
            if use_streamlit:   
                self.st.write ("## Upload a file using the widget in the sidebar to start")
            
    def _dataframe_handler(self, uploaded_file) -> pd.DataFrame:
        try:
            if uploaded_file.name.endswith('.csv'):
                # Load CSV file
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                # Load Excel file with additional options for better datatype handling
                return pd.read_excel(uploaded_file, engine='openpyxl', 
                                     dtype=str)         # Load everything as strings initially
        except Exception as e:
            if self.use_streamlit:  
                self.st.sidebar.error(f"Failed to load data: {e}")
            else:
                print(f"Failed to load data: {e}")
            return self.default
        
    def _doc_handler(self, uploaded_files) -> str:

        concatenated_text = ""
        for uploaded_file in uploaded_files:
            print (uploaded_file)
            try:
                # Check the file extension to determine handling method
                if uploaded_file.name.endswith('.txt'):
                    # If it's a text file, read directly
                    text = uploaded_file.read().decode('utf-8')
                else:
                    # For other types, use textract
                    temp_file_path = '/tmp/' + uploaded_file.name
                    with open(temp_file_path, "wb") as f:
                        FileLockManager(temp_file_path).secure_write(uploaded_file.getbuffer())

                    # Extract text from the saved file using textract
                    text = textract.process(temp_file_path).decode('utf-8')
                
                concatenated_text += text
            except Exception as e:
                if self.use_streamlit:
                    self.st.sidebar.error(f"Failed to load document {uploaded_file.name}: {e}")
                else:
                    print(f"Failed to load document {uploaded_file.name}: {e}")
                concatenated_text += self.default
        
        return concatenated_text



    def load_user_file(self, uploaded_file):
        if self.handler:
            return self.handler(uploaded_file)
        else:
            return uploaded_file