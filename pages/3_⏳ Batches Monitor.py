import logging
logger = logging.getLogger(__name__)

from src.file_manager import BatchRequestLogger
from src.streamlit_interface import streamlit_batches_status
from src.streamlit_setup import page_setup, page_footer
import streamlit as st

page_config = {'page_title':"Batches Monitor",
          'page_icon':"⏳",}
page_setup(page_config)

if st.session_state["authentication_status"]:
    session_logger = st.session_state["session_logger"]
    credential_manager = st.session_state['credential_manager']

    streamlit_batches_status(session_logger)
    
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    st.sidebar.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')
page_footer()