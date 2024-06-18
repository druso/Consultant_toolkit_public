from src.file_manager import AppLogger
from src.setup import page_setup, page_footer
import streamlit as st

page_config = {'page_title':"Session Recovery",
          'page_icon':"💾",}

page_setup(page_config)

app_logger = AppLogger()

st.write("""Something broke along the way? 
         Something messed up the operations? 
         Chill, don't get angry, take a breath.
         Why don't you see if anything was actually saved during the execution? 
         Select your session_id and download the processed files or requests logs. May be able to recover something""")

# List all subfolders in the main directory
session_ids = app_logger.list_logs()
selected_session_id = st.selectbox('Select a session_id', session_ids)

if selected_session_id:
    logs_type = app_logger.list_logs(selected_session_id)
    selected_logs_type = st.selectbox('Select a Nested Folder', logs_type)

    if selected_logs_type:
        if st.button('Prepare the ZIP', use_container_width=True,):
            zip_file = app_logger.zip_logs(selected_session_id, selected_logs_type)
            st.download_button(
                label="Download",
                data=zip_file,
                file_name=f"{selected_session_id}_{selected_logs_type}.zip",
                mime="application/zip",
                use_container_width=True, 
                type='primary'
            )


page_footer()