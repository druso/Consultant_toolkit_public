import streamlit as st
from io import BytesIO
import pandas as pd
from pandas import json_normalize
import json
import textract
import os
from datetime import datetime
import uuid
import zipfile
from typing import Tuple, Any, Optional, List






class DataLoader:
    """
    Facilitates the uploading and processing of various file types within a Streamlit application.

    The class is designed to handle different configurations of file types 
    ('dataframe', 'doc', 'audio', etc.) and their associated processing logic. 

    Attributes:
        configurations (dict): Defines valid configuration keys and their associated 
                               file extensions, handlers (for processing), and default values.
        file_types (list): A list of allowed file extensions based on the chosen configuration.
        handler (callable or None): The function to call for processing the uploaded file.
        default: The default value to use if no file is uploaded.
        uploaded_file (file-like or None): The uploaded file object from the Streamlit sidebar.
    """
    def __init__(self, config_key: str):
        """
        Initializes the DataLoader and attempts to upload a file.

        Args:
            config_key (str): The key specifying the data handling configuration 
                              (e.g., 'dataframe', 'doc', 'audio').
        
        Raises:
            ValueError: If the `config_key` is invalid (not found in configurations).
        """
        self.configurations = {
            "dataframe": {'extensions': ['csv', 'xls', 'xlsx'], 
                          'handler': self._dataframe_handler, 
                          'default':pd.DataFrame()},
            "doc": {'extensions': ['pdf', 'docx', 'txt'], 
                     'handler': self._doc_handler, 
                     'default': "",
                     'accept_multiple_file':True},
            "audio": {'extensions': ['mp3','mp4,mpeg','mpga','m4a','wav','webm'], 
                     'handler': None, 
                     'default':""}
        }
        if config_key not in self.configurations:
            raise ValueError("Invalid configuration key.")
        self.file_types = self.configurations[config_key]['extensions']
        self.handler = self.configurations[config_key]['handler']
        self.default = self.configurations[config_key]['default']
        self.accept_multiple_files =  self.configurations[config_key].get('accept_multiple_file', False)
        self.uploaded_file = st.sidebar.file_uploader("Choose a file", type=self.file_types, accept_multiple_files=self.accept_multiple_files)
        if not self.uploaded_file:
            self.uploaded_file = st.file_uploader("Choose a file", type=self.file_types, key="Main Loader")
            


    def _dataframe_handler(self, uploaded_file) -> pd.DataFrame:
        """
        Processes uploaded files and loads them into a pandas DataFrame.

        This method handles CSV, XLS, and XLSX files. If the file cannot be loaded 
        or an unsupported format is provided, the default value for dataframes 
        (an empty pandas DataFrame) is returned.

        Args:
            uploaded_file (file-like object): The file object uploaded by the user.

        Returns:
            pd.DataFrame: The loaded pandas DataFrame, or the default (empty) DataFrame 
                          if loading fails or an unsupported format is encountered.

        Raises:
            FileNotFoundError: If the specified file cannot be found.
            pd.errors.ParserError: If there's an issue parsing the CSV or Excel file.
            pd.errors.EmptyDataError: If the file is empty.
            # Other potential pandas errors might need to be listed here as well.
        """
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                return pd.read_excel(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Failed to load data: {e}")
            return self.default
        
    def _doc_handler(self, uploaded_files) -> str:

        """
        Extracts text content from uploaded document files.

        Handles both plain text (.txt) files by direct reading and other document types (e.g., .pdf, .docx)
        using the `textract` library. If extraction fails or an unsupported file format is provided, 
        the default value (an empty string) is returned.

        Args:
            uploaded_files (list of file-like objects): The file objects uploaded by the user.

        Returns:
            str: The concatenated text content from all files, or an empty string if extraction fails
                or the file format is unsupported.
        
        Raises:
            FileNotFoundError: If the specified file cannot be found.
            textract.exceptions.MissingFileError: If textract cannot locate the file.
            textract.exceptions.UnsupportedFileTypeError: If textract does not support the file type.
            # Other potential textract exceptions you might want to catch explicitly
        """
        concatenated_text = ""
        for uploaded_file in uploaded_files:
            try:
                # Check the file extension to determine handling method
                if uploaded_file.name.endswith('.txt'):
                    # If it's a text file, read directly
                    text = uploaded_file.read().decode('utf-8')
                else:
                    # For other types, use textract
                    temp_file_path = '/tmp/' + uploaded_file.name
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Extract text from the saved file using textract
                    text = textract.process(temp_file_path).decode('utf-8')
                
                concatenated_text += text
            except Exception as e:
                st.sidebar.error(f"Failed to load document {uploaded_file.name}: {e}")
                concatenated_text += self.default
        
        return concatenated_text

    def load_user_file(self) -> Tuple[Any, Optional[str]]:
        """
        Processes the uploaded file (if any) using the appropriate handler.

        If a file has been uploaded and a handler is defined for the selected configuration, 
        the handler is called to process the file content. Otherwise, the uploaded file object
        and its name are returned directly. If no file is uploaded, the default value for the 
        configuration and `None` (for filename) are returned.

        In case of documents it will support multiple files to be uploaded (but won't be able to track their name for now)

        Returns:
            Tuple[Any, Optional[str]]: A tuple containing:
                - The processed content (if a handler exists and a file was uploaded),
                the uploaded file object itself (if no handler exists),
                or the default value (if no file was uploaded).
                - The filename (if a file was uploaded) or None (if no file was uploaded).
        """
        if self.uploaded_file is not None:
            if self.handler:
                content = self.handler(self.uploaded_file)
                if isinstance(self.uploaded_file, list):
                    filename =  "multiple_files"
                elif self.uploaded_file:
                    filename = self.uploaded_file.name
                return content, filename
            else:
                self.uploaded_file, self.uploaded_file.name
        else:
            return self.default, None






class DataUtilities:
    """
    Provides utility functions for working with Pandas DataFrames.

    Includes methods for:
        - Converting DataFrames to Excel format (bytes).
        - Unrolling JSON data embedded within a DataFrame column.
    """


    def __init__(self):
        pass

    def to_excel(self, df: pd.DataFrame) -> bytes:
        """
        Converts a Pandas DataFrame to Excel format (bytes).

        Args:
            df (pd.DataFrame): The DataFrame to convert.

        Returns:
            bytes: The Excel file content as bytes.
        """
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        # No need to call writer.save()
        processed_data = output.getvalue()
        return processed_data
    
    def unroll_json_in_dataframe(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Unrolls (flattens) JSON data within a specified column of a DataFrame.
        Skips rows that are not valid JSON.

        Args:
            df (pd.DataFrame): The DataFrame containing the JSON data.
            column_name (str): The name of the column containing the JSON data.

        Returns:
            pd.DataFrame: The unrolled DataFrame or the original DataFrame if an error occurs.
        """
        def explode_and_normalize(data):
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.json_normalize(data)
            else:
                return None

        result_rows = []
        for index, row in df.iterrows():
            try:
                json_data = row[column_name]
                if isinstance(json_data, str):
                    json_data = json.loads(json_data)
                
                expanded_data = explode_and_normalize(json_data)
                if expanded_data is not None:
                    for _, expanded_row in expanded_data.iterrows():
                        new_row = row.drop(column_name).to_dict()
                        new_row.update(expanded_row)
                        result_rows.append(new_row)
                else:
                    result_rows.append(row.to_dict())
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                result_rows.append(row.to_dict())

        result_df = pd.DataFrame(result_rows)
        return result_df






class AppLogger() :
    """
    Handles logging and retrieval of data during an application session.

    This class manages the storage and organization of logs related to Excel files, 
    API requests, or any other data you choose to store. It uses a session-based folder structure
    for better organization, with each session having its own unique ID.

    Attributes:
        session_id (str): Unique ID for the current session, used for folder creation.
        file_name (str): Default filename for logged Excel files (can be changed).
        logs_folder (str): Root directory where session-specific logs are stored.
    """
    def __init__(self):
        """Initializes the logger with a session ID and default values."""
        self.session_id = datetime.now().strftime('%y%m%d') +"_"+ str(uuid.uuid4())[:6]
        self.file_name = "default"
        self.logs_folder="logs"
        pass

    def set_file_name(self, file_name: str):
        """
        Sets the filename to be used when logging Excel files.

        Args:
            file_name (str): The new filename to use.
        """
        self.file_name = file_name 

    def log_excel(self, df: pd.DataFrame, version: str = "processed") -> None:
        """
        Saves a Pandas DataFrame to an Excel file within the session's logs folder.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            version (str, optional): A string to include in the filename, indicating
                the data version (e.g., "raw", "processed"). Defaults to "processed".
        """
        folder = f"{self.logs_folder}/{self.session_id}/generative_excel"
        if not os.path.exists(folder):
            os.makedirs(folder)
        excel=DataUtilities().to_excel(df)
        with open(f"{folder}/{version}_{self.file_name}.xlsx", "wb") as f:
            f.write(excel)
        pass

    def save_request_log(self, response: dict, service: str, calltype: str) -> None:
        """
        Saves the response from an API request in either JSON or TXT format.

        Args:
            response (dict): The response data from the API call.
            service (str): The name of the service or API being called.
            calltype (str): A descriptor for the type of API call (e.g., "query", "update").
        """
        folder = f"{self.logs_folder}/{self.session_id}/requests/{service}"
        if not os.path.exists(folder):
            os.makedirs(folder)

        try:
            # Attempt to save as JSON
            with open(f"{folder}/{calltype}_{datetime.now().strftime('%m%d%H%M%S')}.json", "w") as f:
                json.dump(response, f, indent=4)
        except TypeError:
            response = str(response)
            with open(f"{folder}/{calltype}_{datetime.now().strftime('%m%d%H%M%S')}.txt", "w") as f:
                f.write(response)

    def list_logs(self, log: str = None) -> List[str]:
        """
        Lists the session directories or specific logs under a given session.

        Args:
            log (str, optional): The specific log type to list (e.g., "requests"). If None,
                                lists all session directories.

        Returns:
            List[str]: A list of session IDs or log file/folder names.
        """
        path = self.logs_folder if log is None else os.path.join(self.logs_folder, log)
        return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]    

    def zip_directory(self, folder_path: str) -> BytesIO:
        """
        Zips the contents of a directory into a byte stream.

        Args:
            folder_path (str): The path to the directory to be zipped.

        Returns:
            BytesIO: An in-memory byte stream containing the zipped directory.

        Raises:
            FileNotFoundError: If the specified folder doesn't exist.
        """
        byte_io = BytesIO()
        with zipfile.ZipFile(byte_io, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))
        byte_io.seek(0)
        return byte_io
    
    def zip_logs(self, selected_session_id: str, selected_logs_type: str) -> BytesIO:
        """
        Zips the logs of a specific type from a selected session.

        Args:
            selected_session_id (str): The ID of the session whose logs are to be zipped.
            selected_logs_type (str): The type of logs to zip (e.g., "requests", "generative_excel").

        Returns:
            BytesIO: An in-memory byte stream containing the zipped logs.

        Raises:
            FileNotFoundError: If the specified session or log type folder doesn't exist.
        """
        folder = f"{self.logs_folder}/{selected_session_id}/{selected_logs_type}"
        zipped_logs = self.zip_directory(folder)
        return zipped_logs




    """def expand_json_in_dataframe(self, df, column_name):
        # Check if the column contains stringified JSON or dictionaries
        try:
            # Attempt to load the JSON content; it assumes either valid JSON strings or Python dicts
            df[column_name] = df[column_name].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        except json.JSONDecodeError:
            print("Error decoding JSON. Ensure the column contains valid JSON or dictionaries.")

        # Prepare to collect rows that do not need expansion
        other_rows = df[df[column_name].apply(lambda x: not isinstance(x, dict))]
        
        # Prepare to expand rows with JSON-like structures
        json_rows = df[df[column_name].apply(lambda x: isinstance(x, dict))]
        
        # Normalize and expand the JSON-like data into a DataFrame
        expanded_data = json_rows.explode(column_name).reset_index(drop=True)
        expanded_df = json_normalize(expanded_data[column_name])
        
        # Merge the expanded data with the original DataFrame index and other columns
        expanded_df = pd.concat([expanded_data.drop(columns=[column_name]), expanded_df], axis=1)
        
        # Concatenate the rows with no JSON with the expanded DataFrame
        result_df = pd.concat([other_rows, expanded_df], ignore_index=True)
        
        return result_df"""