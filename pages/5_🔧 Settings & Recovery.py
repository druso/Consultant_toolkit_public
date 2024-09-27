from src.file_manager import AppLogger
from src.external_tools import openai_advanced_uses
from src.setup import page_setup, page_footer
from src.prompts import assistant_sys_prompt
import streamlit as st
import os

page_config = {'page_title':"Settings & Recovery",
          'page_icon':"💾",}

page_setup(page_config)

def handle_service_credentials(service, credential_manager):
    if service['initialized'] != "os":
        value = st.text_input(service['key'], type='password', value= service.get("user_provided_key"))
        if value:
            credential_manager.update_user_provided_key(service['key'], value)
            st.success(f"{service['key']} has been updated.")
        else:
            st.error(f"{service['key']} is not set, related services may not work properly")

if st.session_state["authentication_status"]:
    credential_manager = st.session_state['credential_manager']
    app_logger = AppLogger(st.session_state['username'])


    st.title("API Keys Settings")#########################################################################################
    services_list = credential_manager.get_services_list()
    os_uninitialized_services = [service for service in services_list if service['initialized'] != "os"]
    
    if os_uninitialized_services:
        with st.expander("User API Keys", expanded=True):
            for service in os_uninitialized_services:
                handle_service_credentials(service, credential_manager)

    else:
        st.info("All API keys are already set. Give plenty of thanks to the webmaster for setting them up")
    st.divider()




    st.title("Setup Assistant")#########################################################################################
    st.write("If you don't have any assistant available in My Assistants, generate it here")
    if st.button("Generate the openai assistant"):
        openai = openai_advanced_uses(app_logger)
        assistant_configs=st.session_state['tool_config'].get('assistant_configs')
        assistant_list = openai.list_assistants()
        if any(assistant_configs['assistant_name'] == assistant[1] for assistant in assistant_list):
            st.write(f"An assistant with the same name already exists: {assistant_configs['assistant_name']}")
        else:
            assistant_configs['assistant_sys_prompt']=assistant_sys_prompt
            assistant = openai.create_assistant(assistant_configs)
            st.write(f"Assistant created with id {assistant.id}")
    st.divider()




    st.title("Session Recovery")#########################################################################################

    st.write("""Something broke along the way? 
            Something messed up the operations? 
            Chill, don't get angry, take a breath.
            Why don't you see if anything was actually saved during the execution? 
            Select your session_id and download the processed files or requests logs. May be able to recover something""")

    # List all subfolders in the main directory
    
    logs_folder = st.selectbox('Select the type of log', [app_logger.files_folder,app_logger.requests_folder])

    if logs_folder:
        available_logs = app_logger.list_subfolders(logs_folder)
        selected_logs_folder = st.selectbox('Select a folder', available_logs)

        if selected_logs_folder:
            if st.button('Prepare the ZIP', use_container_width=True,):
                zip_file = app_logger.zip_directory(logs_folder+"/"+selected_logs_folder)
                st.download_button(
                    label="Download",
                    data=zip_file,
                    file_name=f"{selected_logs_folder}.zip",
                    mime="application/zip",
                    use_container_width=True, 
                    type='primary'
                )

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    st.sidebar.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')
page_footer()
