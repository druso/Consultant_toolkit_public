import streamlit as st
from io import BytesIO
import pandas as pd
import json
import textract
import os
from datetime import datetime
import uuid
import zipfile
from typing import Tuple, Any, Optional, List




def unroll_json(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Unrolls (flattens) JSON data within a specified column of a DataFrame.
    Ensures that all columns have consistent types to avoid ArrowInvalid errors.
    Applies vectorized operations where possible for better performance.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the JSON data.
        column_name (str): The name of the column containing the JSON data.

    Returns:
        pd.DataFrame: The unrolled DataFrame with consistent column types.
    """
    
    # Helper function to safely load JSON
    def safe_json_loads(val):
        try:
            return json.loads(val) if isinstance(val, str) else val
        except json.JSONDecodeError:
            return None  # Handle invalid JSON by returning None

    # Vectorized JSON loading for the specified column
    df[column_name] = df[column_name].apply(safe_json_loads)

    # Helper function to explode and normalize data
    def explode_and_normalize(data):
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.json_normalize(data)
        else:
            return None

    result_rows = []
    for index, row in df.iterrows():
        json_data = row[column_name]

        # Check if any value within the JSON data is a list or dict
        expanded_data = explode_and_normalize(json_data)
        if expanded_data is not None:
            for _, expanded_row in expanded_data.iterrows():
                new_row = row.drop(column_name).to_dict()  # Drop the original JSON column
                new_row.update(expanded_row)  # Update the row with exploded data
                result_rows.append(new_row)
        else:
            # Straight values are left as-is
            result_rows.append(row.to_dict())

    result_df = pd.DataFrame(result_rows)

    # Ensure consistent type across all columns (convert lists and non-lists to lists)
    for col in result_df.columns:
        if any(isinstance(i, list) for i in result_df[col]):
            result_df[col] = result_df[col].apply(lambda x: x if isinstance(x, list) else [x])

    return result_df


class DataLoader:
    """
    Handles the loading of file into the application
    """
    def __init__(self, config_key: str):

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
            st.write ("## Upload a file using the widget in the sidebar to start")
            #self.uploaded_file = st.file_uploader("Choose a file", type=self.file_types, accept_multiple_files=self.accept_multiple_files, key="Main Loader")
            
    def _dataframe_handler(self, uploaded_file) -> pd.DataFrame:

        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                return pd.read_excel(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Failed to load data: {e}")
            return self.default
        
    def _doc_handler(self, uploaded_files) -> str:

        concatenated_text = ""
        for uploaded_file in uploaded_files:
            print (uploaded_file)
            print ("I'mhere")
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

        if self.uploaded_file is not None:
            if self.handler:
                content = self.handler(self.uploaded_file)
                if isinstance(self.uploaded_file, list):
                    filename =  "multiple_files"
                elif self.uploaded_file:
                    filename = self.uploaded_file.name
                return content, filename
            else:
                return self.uploaded_file, self.uploaded_file.name
        else:
            return self.default, None





class AppLogger() :
    """
    Handles most of I/O operations for storing the activities
    """
    def __init__(self, user_id: str):
        """Initializes the logger with a session ID and default values."""
        self.session_id = datetime.now().strftime('%y%m%d') +"_"+ str(uuid.uuid4())[:6]
        self.file_name = "default"
        
        self.logs_folder = f"logs/{user_id}"
        self.log_types = ["files", "requests", "openai_threads"]
        
        for log_type in self.log_types:
            folder = os.path.join(self.logs_folder, log_type)
            os.makedirs(folder, exist_ok=True)
            setattr(self, f"{log_type}_folder", folder)

    def set_file_name(self, file_name: str):
        self.file_name = file_name 

    def session_files_folder(self):
        folder = os.path.join(self.files_folder, self.session_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
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

    def to_excel(self, df: pd.DataFrame) -> bytes:
 
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        # No need to call writer.save()
        processed_data = output.getvalue()
        return processed_data



    def log_excel(self, df: pd.DataFrame, version: str = "processed", return_path = False) -> None:
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
        path=f"{folder}/{version}_{self.file_name.split('.', 1)[0]}.xlsx"
        with open(path, "wb") as f:
            f.write(self.to_excel(df))
        if return_path:
            return path

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
            st.error(f"thread log for thread id: {thread_id} does not exist")

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
            st.success(f"Thread data for thread id: {thread_id} has been deleted.")
        except FileNotFoundError:
            print(f"Thread log for thread id: {thread_id} does not exist.")
        except Exception as e:
            print(f"Error deleting thread log: {e}")