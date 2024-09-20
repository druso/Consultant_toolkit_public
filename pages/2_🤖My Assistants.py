from src.file_manager import AppLogger
from src.setup import page_setup, page_footer
import streamlit as st
import os

page_config = {'page_title':"My Assistants",
          'page_icon':"💾",}

page_setup(page_config)


if st.session_state["authentication_status"]:
    pass





elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    st.sidebar.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')
page_footer()