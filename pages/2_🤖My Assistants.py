from src.request_processor import assistant_interface
from src.external_tools import openai_advanced_uses
from src.setup import page_setup, page_footer
import streamlit as st
import os

page_config = {'page_title':"My Assistants",
          'page_icon':"🤖",}

page_setup(page_config)


if st.session_state["authentication_status"]:

    openai_advanced_uses = openai_advanced_uses(st.session_state["app_logger"])

    assistant_interface(openai_advanced_uses)



elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    st.sidebar.error('Username/password is incorrect')
    page_footer()
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')
    page_footer()