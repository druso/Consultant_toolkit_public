import logging
logger = logging.getLogger(__name__)

import os
import yaml
from src.file_manager import FileLockManager
from functools import lru_cache
import re
import logging.config


def setup_logging():
    # Define a log format
    log_format = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    # Configure the root logger
    logging.basicConfig(level=logging.INFO,  # Set the base level to DEBUG
                        format=log_format,   # Apply the log format
                        handlers=[
                            logging.StreamHandler(),  # Output to the console
                            logging.FileHandler("app.log")  # Save logs to a file
                        ])
    

@lru_cache(maxsize=1)
def load_config(file_name, handle_env_vars=True):
    try:
        yaml_content = FileLockManager(file_name).secure_read()
 
        if handle_env_vars:
            # Combine the substitution logic here
            pattern = re.compile(r'\$([A-Za-z0-9_]+)')
            def replace(match):
                env_var = match.group(1)
                return os.environ.get(env_var, f'${env_var}')
            yaml_content = pattern.sub(replace, yaml_content)

        return yaml.safe_load(yaml_content)
    except FileNotFoundError:
        print(f"Configuration file {file_name} not found.")
        return {}
    

def scheduler_setup():
    tool_config = load_config('tool_configs.yaml')
    folder = tool_config.get('shared_folder', 'shared')  
    os.makedirs(folder, exist_ok=True)
    setup_logging()
    print ("scheduler started")
    return tool_config   




class CredentialManager:
    def __init__(self, tool_config, use_streamlit=True):
        self.tool_config = tool_config
        self.services_list = self.load_services_config()
        


    def __initialize_services_credentials(self, service):
        if os.environ.get(service['key']):
            if not service.get('initialized'):
                service['initialized'] = 'os'
        else:
            service['initialized'] = False       

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
            return next(iter(api_keys.values()))  # Return the single value if there's only one key
        else:
            return api_keys

    def load_services_config(self):
        services_list = []
        for service, config in self.tool_config.items():
            if isinstance(config, dict) and config.get('use', False):
                if isinstance(config['key'], list):
                    for key in config['key']:
                        services_list.append({'key': key, 'service_name': service})
                else:
                    services_list.append({'key': config['key'], 'service_name': service})
                
        
        for service in services_list:
            self.__initialize_services_credentials(service)

        return services_list

    def get_services_list(self):
        return self.services_list

    def update_user_provided_key(self, service_key, user_key):
        for service in self.services_list:
            if service['key'] == service_key:
                service['user_provided_key'] = user_key
                service['initialized'] = 'user'
                break