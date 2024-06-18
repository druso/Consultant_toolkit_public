from src.file_manager import AppLogger
import streamlit as st
import pandas as pd
import os

secrets = [{'key':'OPENAI_API_KEY'},
           {'key':'GROQ_API_KEY'},
           {'key':'SERP_API_KEY'},
           {'key':'OXYLABS_USER'},
           {'key':'OXYLABS_PSW'},
           ]


def initialize_secret(secret):
    if os.environ.get(secret['key']):
        st.session_state[secret['key']] = os.environ.get(secret['key'])
        secret['initialized_os'] = True

def handle_secret(secret):
        if not secret.get('initialized_os'):
            st.session_state[secret['key']] =st.text_input(secret['key'], value=st.session_state.get(secret['key']), type='password')
            if not st.session_state.get(secret['key']):
                st.error(f"{secret['key']} is not set, related services may not work properly")
            else:
                secret['initialized_user'] = True


def page_setup(page_config):


    st.set_page_config(
        page_title=f"{page_config['page_title']}",
        page_icon=page_config['page_icon'],
        layout="wide",)

    if "app_logger" not in st.session_state:
        st.session_state["app_logger"] = AppLogger()
        
    session_id = st.session_state["app_logger"].session_id
    st.sidebar.write(f"Your session id: {session_id}")
    
    for secret in secrets:
        initialize_secret(secret)

    if any(not secret.get('initialized_os') for secret in secrets):
        with st.expander("User API Keys", expanded=not all(st.session_state.get(secret['key']) for secret in secrets)):
            for secret in secrets:
                handle_secret(secret)

    st.write(f"# {page_config['page_icon']} - {page_config['page_title']}")

def page_footer():
    st.divider()
    st.write("No cookie or tracking is done by Druso")
    st.write("However files are copied and stored on US servers and also shared with OpenAI and other 3rd party when you run the functions. Please be mindful.")
    st.write("[OpenAI EU Terms of Use](https://openai.com/policies/eu-terms-of-use)")
    st.write("[Groq Terms of Use](https://wow.groq.com/terms-of-use/)")