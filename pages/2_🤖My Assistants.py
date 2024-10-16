import logging
logger = logging.getLogger(__name__)
from src.streamlit_interface import assistant_interface
from src.external_tools import openai_advanced_uses
from src.streamlit_setup  import page_setup, page_footer
import streamlit as st
import os

page_config = {'page_title':"My Assistants",
          'page_icon':"🤖",}

page_setup(page_config)


if st.session_state["authentication_status"]:
    app_logger = st.session_state["app_logger"]
    credential_manager = st.session_state['credential_manager']

    openai_assistant = openai_advanced_uses(app_logger, credential_manager)

    assistant_interface(openai_assistant)



elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    st.sidebar.error('Username/password is incorrect')
    page_footer()
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')
    page_footer()