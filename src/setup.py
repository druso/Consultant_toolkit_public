import os
import yaml
import streamlit as st
import streamlit_authenticator as stauth
from src.file_manager import AppLogger
from yaml.loader import SafeLoader
from functools import lru_cache


@lru_cache(maxsize=1)
def load_config(file_name):
    try:
        with open(file_name, 'r') as file:
            return yaml.load(file, Loader=SafeLoader)
    except FileNotFoundError:
        print(f"Configuration file {file_name} not found.")
        return {}


def user_login():
    if st.session_state['tool_config'].get('require_login', False):
        user_config = load_config('users.yaml')
        authenticator = stauth.Authenticate(
            user_config.get ('credentials', {}),
            user_config.get('cookie', {}).get('name', 'consulting_toolkit_cookie'),
            user_config.get('cookie', {}).get('key', 'signature_key'),
            user_config.get('cookie', {}).get('expiry_days', 30),
            user_config.get('pre-authorized', {})
        )
        authenticator.login()
    else:
        st.session_state["authentication_status"] = True


class CredentialManager:
    def __init__(self, tool_config):
        self.tool_config = tool_config
        self.services_list = self.load_services_config()

    def get_api_key(self, service_name):
        api_keys = {}
        for service in self.services_list:
            if service['service_name'] == service_name:
                key_name = service['key']
                if service.get('user_provided_key'):
                    api_keys[key_name] = service.get('user_provided_key')
                else:
                    api_keys[key_name] = os.environ.get(key_name)
        if not api_keys:
            return None
        elif len(api_keys) == 1:
            api_keys
            return next(iter(api_keys.values()))  # Return the single value if there's only one key
        else:
            return api_keys

    def initialize_services_credentials(self, service):
        if os.environ.get(service['key']):
            if not service.get('initialized'):
                service['initialized'] = 'os'
        else:
            service['initialized'] = False

    def load_services_config(self):
        services_list = []
        for service, config in self.tool_config.items():
            if isinstance(config, dict) and config.get('use', False):
                if isinstance(config['key'], list):
                    for key in config['key']:
                        services_list.append({'key': key, 'service_name': service})
                else:
                    services_list.append({'key': config['key'], 'service_name': service})
                st.session_state[f'use_{service}'] = True
        
        for service in services_list:
            self.initialize_services_credentials(service)

        return services_list

    def get_services_list(self):
        return self.services_list

    def update_user_provided_key(self, service_key, user_key):
        for service in self.services_list:
            if service['key'] == service_key:
                service['user_provided_key'] = user_key
                service['initialized'] = 'user'
                break



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

    if st.session_state.get("authentication_status"):       

        if not st.session_state.get('credential_manager'):
            st.session_state['credential_manager'] = CredentialManager(st.session_state['tool_config'])
 
        services_list = st.session_state['credential_manager'].get_services_list()
        print (services_list)
        
        if [service for service in services_list if service['initialized'] == False]:
            st.warning("Some API keys are not set. Please go to **🔧 Settings & Recovery** from menu and provide them to use the full potential of the toolkit")
        

        if "app_logger" not in st.session_state:
            st.session_state["app_logger"] = AppLogger()
            
        session_id = st.session_state["app_logger"].session_id
        st.sidebar.write(f"Your session id: {session_id}")    

def page_footer():
    st.divider()
    st.write("No other cookie or tracking is done here apart from consulting_toolkit_cookie for login and streamlit session handler")
    st.write("However uploaded files are stored on server hosting this service and when processed information are shared with OpenAI and other 3rd party when you run the functions. Please be mindful.")
    st.write("[OpenAI EU Terms of Use](https://openai.com/policies/eu-terms-of-use)")
    st.write("[Groq Terms of Use](https://wow.groq.com/terms-of-use/)")