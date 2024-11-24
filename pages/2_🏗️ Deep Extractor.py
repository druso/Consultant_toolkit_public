import logging
logger = logging.getLogger(__name__)
from src.streamlit_interface import DeepExtractorInterface
from src.streamlit_setup  import page_setup, page_footer
import streamlit as st
import os

page_config = {'page_title':"Deep Extractor",
          'page_icon':"🏗️",}

page_setup(page_config)


if st.session_state["authentication_status"]:
    session_logger = st.session_state["session_logger"]
    credential_manager = st.session_state['credential_manager']

    DeepExtractorInterface(session_logger).main_interface()


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    st.sidebar.error('Username/password is incorrect')
    page_footer()
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')
    page_footer()