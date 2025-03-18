import logging
logger = logging.getLogger(__name__)
from io import BytesIO, StringIO
from src.dataformats import BatchSummaryPayload, BatchRequestPayload
import pandas as pd
import json
import os
from datetime import datetime
import uuid
import zipfile
import time
import re
from typing import Tuple, Any, Optional, List, Union
import shutil
from contextlib import contextmanager


class FileLockManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.lock_file = file_path + ".lock"

    def _acquire_lock(self, timeout=10, check_interval=1):
        start_time = time.time()
        while True:
            try:
                fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                logger.debug(f"Lock acquired on {self.lock_file}")
                return True
            except FileExistsError:
                if (time.time() - start_time) >= timeout:
                    logger.error(f"Timeout while waiting to acquire lock on {self.lock_file}")
                    raise TimeoutError(f"Could not acquire lock on {self.lock_file} within {timeout} seconds.")
                logger.debug(f"Lock exists on {self.lock_file}. Retrying in {check_interval} seconds...")
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"Unexpected error while acquiring lock: {e}")
                raise

    def _release_lock(self):
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
                logger.debug(f"Lock released on {self.lock_file}")
        except Exception as e:
            logger.error(f"Error releasing lock {self.lock_file}: {e}")

    @contextmanager
    def locked(self):
        """Context manager to handle lock acquisition and release."""
        self._acquire_lock()
        try:
            yield
        finally:
            self._release_lock()

    def secure_write(self, data):
        """
        Securely write binary data to the file.

        Args:
            data (bytes): Data to write to the file.
        """
        with self.locked():
            with open(self.file_path, "wb") as file:
                file.write(data)
            logger.debug(f"Finished writing to {self.file_path}")

    def secure_read(self):
        """
        Securely read the file as raw text.

        Returns:
            str: Contents of the file.
        """
        with self.locked():
            with open(self.file_path, "r", encoding='utf-8') as file:
                data = file.read()
            logger.debug(f"Finished reading from {self.file_path}")
            return data

    def secure_rename(self, new_path):
        """
        Securely rename the file to a new path.

        Args:
            new_path (str): The new file path.
        """
        with self.locked():
            try:
                # Rename the main file
                os.rename(self.file_path, new_path)
                logger.info(f"Renamed {self.file_path} to {new_path}")

                # Rename the lock file to match the new file
                new_lock_file = new_path + ".lock"
                os.rename(self.lock_file, new_lock_file)
                logger.debug(f"Renamed lock file from {self.lock_file} to {new_lock_file}")

                # Update the internal state after successful renaming
                self.file_path = new_path
                self.lock_file = new_lock_file
            except Exception as e:
                logger.error(f"Error renaming file: {e}")
                raise

    def secure_remove(self):
        """
        Securely remove the file.
        """
        with self.locked():
            try:
                if os.path.exists(self.file_path):
                    os.remove(self.file_path)
                    logger.info(f"Removed {self.file_path}")
                else:
                    logger.warning(f"File {self.file_path} does not exist")
            except Exception as e:
                logger.error(f"Error removing file: {str(e)}")
                raise

    def secure_move(self, new_path):
        with self.locked():
            try:
                shutil.move(self.file_path, new_path)
                logger.info(f"Moved {self.file_path} to {new_path}")

                # Rename the lock file to match the new file
                new_lock_file = new_path + ".lock"
                if os.path.exists(self.lock_file):
                    shutil.move(self.lock_file, new_lock_file)
                    logger.debug(f"Renamed lock file from {self.lock_file} to {new_lock_file}")

                # Update internal state
                self.file_path = new_path
                self.lock_file = new_lock_file
            except Exception as e:
                logger.error(f"Error moving file: {e}")
                raise

    def secure_csv_update(self, update_data, update_function, **kwargs):
        """
        Securely update a CSV file using a provided update function.

        Args:
            update_data (dict or pd.DataFrame): Data to update or add.
            update_function (callable): A function that takes a DataFrame and update_data as input
                                        and returns an updated DataFrame.
            **kwargs: Additional arguments to pass to the update_function.
        """
        with self.locked():
            try:
                try:
                    with open(self.file_path, "r", encoding='utf-8') as file:
                        df = pd.read_csv(file)
                    logger.debug(f"Read {len(df)} rows from existing CSV.")
                except FileNotFoundError:
                    df = pd.DataFrame()
                    logger.debug("CSV file not found. Created a new empty DataFrame.")

                # Apply the update function
                updated_df = update_function(df, update_data, **kwargs)
                logger.debug(f"DataFrame updated. Now has {len(updated_df)} rows.")

                # Write the updated DataFrame back to CSV
                with open(self.file_path, "w", encoding='utf-8', newline='') as file:
                    updated_df.to_csv(file, index=False)

                logger.info(f"Successfully updated CSV file: {self.file_path}")
            except Exception as e:
                logger.error(f"Error updating CSV file {self.file_path}: {str(e)}")
                raise

class DataFrameProcessor:
    def __init__(self, df: pd.DataFrame,):
        self.user_df = df.copy()
        self.processed_df = df.copy()

            
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
            logger.info(f"Column '{column_name}' has been parsed from {original_type} to {new_type}")
        else:
            logger.info(f"No change in data type. Column '{column_name}' remains as {new_type}")


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

    
class FolderSetupMixin:
    def setup_shared_folders(self, tool_config):

        #shared folders
        self.shared_folder = os.getenv('SHARED_FOLDER') or tool_config.get('shared_folder') or "shared"
        self.batches_folder = os.path.join(self.shared_folder, 'batches')
        self.batches_summary_folder = os.path.join(self.shared_folder,tool_config['shared_summaries_folder'])
        self.completed_dir = os.path.join(self.batches_folder, tool_config['shared_completed_folder'])
        self.wip_folder = os.path.join(self.shared_folder, tool_config['shared_wip_folder'])

        folders_to_create = [
            self.shared_folder,
            self.batches_folder,
            self.batches_summary_folder,
            self.completed_dir,
            self.wip_folder,
        ]

        for folder in folders_to_create:
            if not os.path.exists(folder):
                os.makedirs(folder)
                logger.info(f"Created directory: {folder}")

    def setup_user_folders(self, tool_config, user_id):

        logs_root_folder = os.getenv('LOGS_ROOT_FOLDER') or tool_config.get('logs_root_folder') or "logs"
        self.logs_folder = f"{logs_root_folder}/{user_id}"
        #logs folders
        self.log_types = ["files", "requests", "openai_threads"]
        
        for log_type in self.log_types:
            folder = os.path.join(self.logs_folder, log_type)
            os.makedirs(folder, exist_ok=True)
            setattr(self, f"{log_type}_folder", folder)


class SessionLogger(FolderSetupMixin) :
    """
    Handles most of I/O operations for storing the activities
    """
    def __init__(self, user_id: str, tool_config):
        """Initializes the logger with a session ID and default values."""
        self.user_id = user_id
        self.session_id = datetime.now().strftime('%y%m%d') +"_"+ str(uuid.uuid4())[:6]
        self.file_name = "default"
        self.tool_config = tool_config

        self.setup_user_folders(self.tool_config, self.user_id)
        self.setup_shared_folders(self.tool_config)


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
                logger.info(f"Saved original file {file_path}")
            else:
                #logger.error(f"File {file_path} already exists. Skipping.")
                pass

        if isinstance(uploaded_file, list):
            for index, file in enumerate(uploaded_file):
                save_single_file(file, index)
            self.file_name = "multiple_files"
        else:
            save_single_file(uploaded_file)
            self.file_name = uploaded_file.name

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

    def purge_shared_files(self):
        folder = self.shared_folder
        for root, _, files in os.walk(folder):
            for file in files:
                if file.startswith(self.user_id):
                    file_path = os.path.join(root, file)
                    file_lock = FileLockManager(file_path)
                    try:
                        file_lock.secure_remove()
                    except Exception as e:
                        logger.error(f"Error removing file {file_path}: {str(e)}")


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
        logger.info (f"saved excel log at {path}")
        if return_path:
            return path
        if return_filename:
            return filename
        
    def log_json(self, data: Union[dict, list], name: str = "", return_path: bool = False, return_filename: bool = False) -> Optional[str]:
        """
        Saves a dictionary or list as a JSON file within the session's logs folder.

        Args:
            data (Union[dict, list]): The data to save as JSON.
            version (str, optional): A string to include in the filename, indicating
                the data version (e.g., "raw", "processed"). Defaults to "processed".
            return_path (bool, optional): If True, returns the full path where the file was saved.
            return_filename (bool, optional): If True, returns just the filename.

        Returns:
            Optional[str]: The path or filename if requested, None otherwise.
        """
        folder = self.session_files_folder()
        os.makedirs(folder, exist_ok=True)
        filename = f"{name}.json"
        path = f"{folder}/{filename}"
        
        with open(path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved JSON log at {path}")
        
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
            logger.info(f"Filename cleaned from '{filename}' to '{cleaned_filename}'")

        try:
            # Attempt to save as JSON
            with open(f"{folder}/{cleaned_filename}.json", "w") as f:
                json.dump(response, f, indent=4)
            logger.info (f"saved json log at {folder}/{cleaned_filename}.json")
        except TypeError:
            # If JSON serialization fails, save as text
            with open(f"{folder}/{cleaned_filename}.txt", "w") as f:
                f.write(str(response))


class BatchRequestLogger(FolderSetupMixin):

    def __init__(self, user_id, session_id, tool_config):
        """Initializes the logger with a session ID and default values."""
        self.user_id = user_id  
        self.session_id = session_id
        self.tool_config = tool_config
        self.batch_summary_logger = BatchSummaryLogger(self.tool_config)
        self.setup_user_folders(self.tool_config, self.user_id)
        self.setup_shared_folders(self.tool_config)

    def _generate_batch_id(self):
        def get_highest_batch_number(directory):
            pattern = r".*batch_(\d{3})_(PENDING|WIP|COMPLETED|CANCELLED|STOPPED)\.json"
            highest_number = 0
            for filename in os.listdir(directory):
                match = re.match(pattern, filename)
                if match:
                    batch_number = int(match.group(1))
                    highest_number = max(highest_number, batch_number)
            return highest_number

        highest_in_batches = get_highest_batch_number(self.batches_folder)
        highest_in_completed = get_highest_batch_number(self.completed_dir)
        
        highest_overall = max(highest_in_batches, highest_in_completed)
        new_batch_number = highest_overall + 1
        
        return f"batch_{new_batch_number:03d}"
        
    def post_df_scheduler_request(self, 
                               df, 
                               session_logger, 
                               function_name, 
                               query_column, 
                               response_column, 
                               kwargs):
        batch_id = self._generate_batch_id()
        #batch_id = "batch_" + str(len(batches_list) + 1).zfill(3)

        input_filename = session_logger.log_excel(df, "batch", return_path=False, return_filename=True)
        processed_kwargs = {
            key: (value.__name__ if callable(value) else 
                  value if isinstance(value, (int, float, str, bool, type(None))) else 
                  str(value))
            for key, value in kwargs.items()
        }

        payload = BatchRequestPayload(
            batch_id=batch_id,
            user_id=self.user_id,
            session_id=self.session_id,
            function=function_name,
            type="df",
            batch_size=0,
            input_file=input_filename,
            query_column=query_column,
            response_column=response_column,
            kwargs=processed_kwargs
        )

        folder = self.tool_config.get('shared_folder', 'shared')
        payload_filename=f"{self.user_id}_{batch_id}_PENDING.json"
        with open(f"{folder}/batches/{payload_filename}", "w") as f:
            json.dump(payload.to_dict(), f, indent=4)
            logger.info(f"saved batch request at {folder}/batches/{batch_id}_PENDING.json")
        self.batch_summary_logger.update_batch_summary(payload, status="PENDING", filename=payload_filename)


    def post_list_scheduler_request(self, 
                               queries_list, 
                               session_logger, 
                               function_name, 
                               kwargs):
        batch_id = self._generate_batch_id()

        request_name = datetime.now().strftime('%y%m%d%H%M%S') + "_" + function_name
        input_filename= session_logger.log_json(queries_list, request_name, return_path=False, return_filename=True)
        processed_kwargs = {
            key: (value.__name__ if callable(value) else 
                  value if isinstance(value, (int, float, str, bool, type(None))) else 
                  str(value))
            for key, value in kwargs.items()
        }

        payload = BatchRequestPayload(
            batch_id=batch_id,
            user_id=self.user_id,
            session_id=self.session_id,
            function=function_name, #TOFIX this name generate confusion, it should be function_name also in the payload
            type="list",
            batch_size=kwargs["request_size"], #TOFIX batch size is the request size, I should change this way of naming  as for the df is set to 0 as it will always request the whole file
            input_file=input_filename,
            query_column="list processor",
            response_column="list processor",
            kwargs=processed_kwargs
        )

        folder = self.tool_config.get('shared_folder', 'shared')
        payload_filename=f"{self.user_id}_{batch_id}_PENDING.json"
        with open(f"{folder}/batches/{payload_filename}", "w") as f:
            json.dump(payload.to_dict(), f, indent=4)
            logger.info(f"saved batch request at {folder}/batches/{batch_id}_PENDING.json")
        self.batch_summary_logger.update_batch_summary(payload, status="PENDING", filename=payload_filename)

    def read_wip_progress(self, user_id, batch_id):
        for file in os.listdir(self.wip_folder):
            if file.startswith(f"{user_id}_{batch_id}_"):
                return int(file.split('_')[-1])
        return 0

    def load_batches_summary(self):
        """Load the batches summary CSV file using FileLockManager."""
        csv_path = os.path.join(
            self.tool_config['shared_folder'],
            self.tool_config['shared_summaries_folder'], 
            f'{self.user_id}_batches_summary.csv'
            )
        
        lock_manager = FileLockManager(csv_path)
        try:
            csv_content = lock_manager.secure_read()
            return pd.read_csv(StringIO(csv_content))
        except FileNotFoundError:
            logger.warning(f"CSV file not found at {csv_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return pd.DataFrame()


class BatchSummaryLogger(FolderSetupMixin):
    def __init__(self, tool_config):
        self.setup_shared_folders(tool_config)

    def update_batch_summary(self, payload: BatchRequestPayload, status="PENDING", filename=None, total_rows=None):
        summary_payload = BatchSummaryPayload.from_request_payload(payload, status)

        update_fields = ['status']
        if status == "PENDING":
            summary_payload.schedule_time = datetime.now().isoformat()
            summary_payload.filename = filename
            update_fields.extend(['schedule_time', 'filename'])
        elif status == "WIP":
            summary_payload.start_time = datetime.now().isoformat()
            summary_payload.batch_size = total_rows
            update_fields.extend(['start_time', 'batch_size'])
        elif status == "COMPLETED":
            summary_payload.end_time = datetime.now().isoformat()
            summary_payload.filename = filename
            update_fields.extend(['end_time', 'filename'])
        elif status == "CANCELLED":
            summary_payload.end_time = datetime.now().isoformat()
            update_fields.extend(['end_time', 'filename'])
        elif status == "STOPPED":
            summary_payload.end_time = datetime.now().isoformat()
            update_fields.extend(['end_time', 'filename'])

        csv_path = os.path.join(
            self.batches_summary_folder,
            f'{payload.user_id}_batches_summary.csv'
        )

        def update_function(df, update_data, update_fields):
            new_row = pd.DataFrame([update_data])
            if 'batch_id' in df.columns and df['batch_id'].isin([update_data['batch_id']]).any():
                # Update only specific fields for existing rows
                for field in update_fields:
                    df.loc[df['batch_id'] == update_data['batch_id'], field] = new_row[field].values[0]
                logger.debug(f"Updated existing batch_id {update_data['batch_id']} in summary CSV.")
            else:
                # Append new row if it doesn't exist
                df = pd.concat([df, new_row], ignore_index=True)
                logger.debug(f"Appended new batch_id {update_data['batch_id']} to summary CSV.")
            return df

        lock_manager = FileLockManager(csv_path)
        lock_manager.secure_csv_update(
            update_data=summary_payload.to_dict(),
            update_function=update_function,
            update_fields=update_fields
        )

        logger.info(f"Updated CSV summary for user {payload.user_id}")


    def update_wip_progress(self, user_id, batch_id, wip_percentage):
        """
        Creates or updates an empty file with a name that reflects the batch progress.
        """
        if not 0 <= wip_percentage <= 100:
            raise ValueError(f"Progress must be between 0 and 100. Got {wip_percentage}")
        
        for existing_file in os.listdir(self.wip_folder):
            if existing_file.startswith(f"{user_id}_{batch_id}_"):
                os.remove(os.path.join(self.wip_folder, existing_file))

        file_name = f"{user_id}_{batch_id}_{wip_percentage}"
        file_path = os.path.join(self.wip_folder, file_name)

        if not wip_percentage == 100:
            open(file_path, 'w').close()
            logger.info(f"Progress updated: {file_name}")
    
    def post_stop_request(self, user_id, batch_id):
        stop_requests = self.check_stop_requests()
        file_name = f"{user_id}_{batch_id}_STOP"
        if file_name not in stop_requests:
            file_path = os.path.join(self.batches_folder, file_name)
            open(file_path, 'w').close()
        else:
            raise ValueError(f"Stop request for batch {batch_id} already exists.")
    
    def check_stop_requests(self):
        stop_requests = []
        for file in os.listdir(self.batches_folder):
            if file.endswith("_STOP"):
                stop_requests.append(file)
        return stop_requests

    def handled_stop_request(self, user_id, batch_id):
        file_name = f"{user_id}_{batch_id}_STOP"
        file_path = os.path.join(self.batches_folder, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

    def update_payload_status(self, payload_filename, status):
        payload_filepath = os.path.join(self.batches_folder, payload_filename)
        
        if not status in ["PENDING", "WIP", "COMPLETED", "CANCELLED","STOPPED", "FAILED"]:
            """
            PENDING: waiting to be processed
            WIP: being processed
            COMPLETED: processed successfully
            CANCELLED: processing was cancelled before starting
            STOPPED: processing was stopped while running
            FAILED: processing failed
            """
            raise ValueError(f"Invalid status: '{status}'")

        payload_dir_path, payload_filename = os.path.split(payload_filepath)

        base_name = re.sub(r'_(PENDING|WIP|COMPLETED)\.json$', '', payload_filename)
        
        new_filename = f"{base_name}_{status}.json"
        new_payload_path = os.path.join(payload_dir_path, new_filename)
        file_lock = FileLockManager(payload_filepath)

        try:
            file_lock.secure_rename(new_payload_path)
            logger.info(f"Updated {payload_filename} to {new_filename}")  

            if status == "COMPLETED" or status == "CANCELLED" or status == "STOPPED":
                archived_payload_path = os.path.join(self.completed_dir,new_filename)

                FileLockManager(new_payload_path).secure_move(archived_payload_path)
            logger.info(f"Successfully updated payload status for '{new_filename}'.")
            return new_filename 
                  
        except Exception as e:
            logger.error(f"Error updating payload status: {str(e)}")
            return payload_filename
        
    def cleanup_stuck_progress_files(self, timeout_seconds):
        """
        Removes old progress files that might be stuck due to errors or interruptions.

        Args:
            user_id (str): The user ID.
            batch_id (str): The batch ID.
            timeout_seconds (int): The maximum age of a progress file in seconds before it's considered stuck.
        """
        now = time.time()
        for file in os.listdir(self.wip_folder):
 
            file_path = os.path.join(self.wip_folder, file)
            file_lock = FileLockManager(file_path) #instantiate the class
            try:
                file_age = now - os.path.getmtime(file_path)
                if file_age > timeout_seconds:
                    file_lock.secure_remove() #use the secure_remove function
                    logger.warning(f"Removed stuck progress file: {file}")

            except Exception as e:
                logger.error(f"Error cleaning up progress file {file}: {e}")

        if not os.listdir(self.wip_folder):
               logger.info(f"No stuck files found in {self.wip_folder}")

class OpenaiThreadLogger(FolderSetupMixin) :

    def __init__(self, user_id: str, tool_config):

        self.user_id = user_id
        self.tool_config = tool_config
        self.setup_user_folders(self.tool_config, self.user_id)


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
                    logger.error(f"Error reading JSON file: {filename}")
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {str(e)}")
 
        return thread_list
    
    def get_thread_info(self, thread_id):
        file_path = os.path.join(self.openai_threads_folder, thread_id + ".json")
        try:
            with open(file_path, 'r') as f:
                thread_data = json.load(f)
                return thread_data
        except:
            logger.warning(f"thread log for thread id: {thread_id} does not exist")

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
            logger.info(f"Thread data for thread id: {thread_id} has been deleted.")
        except FileNotFoundError:
            logger.warning(f"Thread log for thread id: {thread_id} does not exist.")
        except Exception as e:
            logger.error(f"Error deleting thread log: {e}")
