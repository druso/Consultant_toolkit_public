import logging
logger = logging.getLogger(__name__)

import pandas as pd
import streamlit as st
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher
from src.setup import load_config, setup_logging, CredentialManager
from src.file_manager import AppLogger, FileLockManager
import textract

def user_login():

    if st.session_state['tool_config'].get('require_login', False):
        user_config = load_config('users.yaml')
        if not user_config:
            st.error("Users configuration file not found. Please check users.yaml file")
            st.stop()

        #Some times streamlit authenticator does not hash the config passwords automatically, making it impossible to login. 
        #these lines should address those cases
        for username, user_info in user_config['credentials']['usernames'].items():
            if not user_info['password'].startswith('$2b$'):  # Check if the password is not already hashed
                user_info['password'] = Hasher([user_info['password']]).generate()[0]  # Hash the password


        authenticator = stauth.Authenticate(
            user_config.get ('credentials', {}),
            user_config.get('cookie', {}).get('name', 'consulting_toolkit_cookie'),
            user_config.get('cookie', {}).get('key', 'signature_key'),
            user_config.get('cookie', {}).get('expiry_days', 30),
            user_config.get('pre-authorized', {})
        )
        authenticator.login()
        if st.session_state["authentication_status"]:
            authenticator.logout(location='sidebar')
            
    else:
        st.session_state["authentication_status"] = True
        st.session_state["username"] = "anonymous"



def page_setup(page_config):
    st.set_page_config(
        page_title=page_config['page_title'],
        page_icon=page_config['page_icon'],
        layout="wide",
    )
    st.write(f"# {page_config['page_icon']} - {page_config['page_title']}")

    if not st.session_state.get('tool_config'):
        st.session_state['tool_config'] = load_config('tool_configs.yaml')

    user_login()
    if 'logging_setup' not in st.session_state:
        setup_logging()  
        st.session_state['logging_setup'] = True
    

    if st.session_state.get("authentication_status"):       

        if not st.session_state.get('credential_manager'):
            st.session_state['credential_manager'] = CredentialManager(st.session_state['tool_config'])
 
        if [service for service in st.session_state['credential_manager'].services_list if service['initialized'] == False]:
            st.warning("Some API keys are not set. Please go to **🔧 Settings & Recovery** from menu and provide them to use the full potential of the toolkit")
        else:
            logger.warning("Warning: Some API keys are not set. May not be able to run all functions")


        if "app_logger" not in st.session_state:
            st.session_state["app_logger"] = AppLogger(st.session_state['username'],st.session_state['tool_config'] )
            
        st.sidebar.write(f"Your session id: {st.session_state['app_logger'].session_id}")  

def page_footer():
    st.divider()
    st.write("No other cookie or tracking is done here apart from consulting_toolkit_cookie for login and streamlit session handler")
    st.write("However uploaded files are stored on server hosting this service and when processed information are shared with OpenAI and other 3rd party when you run the functions. Please be mindful.")
    st.write("[OpenAI EU Terms of Use](https://openai.com/policies/eu-terms-of-use)")
    st.write("[Groq Terms of Use](https://wow.groq.com/terms-of-use/)")



def configure_llm_streamlit(llm_manager, LLMManager, app_logger):

    # Initialize a temporary LlmManager to get available configurations
    configurations = llm_manager.configurations
    config_keys = list(configurations.keys())

    # Collect user input from Streamlit frontend
    config_key = st.sidebar.selectbox("Select LLM Model", config_keys)
    llm_temp = st.sidebar.slider("Set Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

    return LLMManager(   
        app_logger=app_logger,
        credential_manager=st.session_state['credential_manager'],
        config_key= config_key,
        llm_temp= llm_temp,)

class DataLoader:
    """
    Handles the loading of file into the application
    """
    def __init__(self, config_key: str, app_logger: AppLogger):

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

        if config_key not in self.configurations:
            raise ValueError("Invalid configuration key.")
        self.app_logger = app_logger
        self.file_types = self.configurations[config_key]['extensions']
        self.handler = self.configurations[config_key]['handler']
        self.default = self.configurations[config_key]['default']
        self.accept_multiple_files =  self.configurations[config_key].get('accept_multiple_file', False)
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=self.file_types, accept_multiple_files=self.accept_multiple_files)
        if uploaded_file:
            self.user_file = self.load_user_file(uploaded_file)
            self.app_logger.save_original_file(uploaded_file)
            
        else:
            self.user_file = None
            self.content = self.default
            st.write ("## Upload a file using the widget in the sidebar to start")
            
    def _dataframe_handler(self, uploaded_file) -> pd.DataFrame:
        try:
            if uploaded_file.name.endswith('.csv'):
                # Load CSV file
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                return pd.read_excel(uploaded_file, engine='openpyxl', 
                                     dtype=str)   
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
                if uploaded_file.name.endswith('.txt'):
                    text = uploaded_file.read().decode('utf-8')
                else:
                    temp_file_path = '/tmp/' + uploaded_file.name
                    with open(temp_file_path, "wb") as f:
                        FileLockManager(temp_file_path).secure_write(uploaded_file.getbuffer())

                    text = textract.process(temp_file_path).decode('utf-8')
                
                concatenated_text += text
            except Exception as e:
                st.sidebar.error(f"Failed to load document {uploaded_file.name}: {e}")
                print(f"Failed to load document {uploaded_file.name}: {e}")
                concatenated_text += self.default
        
        return concatenated_text



    def load_user_file(self, uploaded_file):
        if self.handler:
            return self.handler(uploaded_file)
        else:
            return uploaded_file