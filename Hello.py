from src.setup import page_setup, page_footer
import streamlit as st
import pandas as pd
import os

page_config = {'page_title':"Consulting toolkit",
          'page_icon':"🛠️",}

page_setup(page_config)

if st.session_state["authentication_status"]:
    with open("readme.md", "r", encoding="utf-8") as f:
        readme_content = f.read()

    st.write(readme_content)
    

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    st.sidebar.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')

page_footer()