import logging
logger = logging.getLogger(__name__)
import streamlit as st
import os
from src.streamlit_interface import SingleRequestConstructor
from src.external_tools import LlmManager, AudioTranscribe
from src.streamlit_setup import page_setup, page_footer, DataLoader, configure_llm_streamlit
import io
from openai import OpenAI

page_config = {'page_title':"Audio Transcriber",
          'page_icon':"🎙️",}
page_setup(page_config)

if st.session_state["authentication_status"]:
    session_logger = st.session_state["session_logger"]
    credential_manager = st.session_state['credential_manager']
    data_loader = DataLoader("audio", session_logger)


    if data_loader.user_file:
        transcribe_button_disabled = False

    llm_manager = LlmManager(session_logger,credential_manager)
    llm_manager = configure_llm_streamlit(llm_manager, LlmManager, session_logger)
    audio_transcriber = AudioTranscribe(session_logger, st.session_state['credential_manager'])


    transcript = SingleRequestConstructor().audio_transcribe(data_loader.user_file, audio_transcriber)

    if "transcript" not in st.session_state:
        st.session_state["transcript"] = ""
    if transcript:  # Update only if we get new transcript
        st.session_state["transcript"] = transcript

    # Use session state for the text area
    provided_text = st.text_area(
        "Your Transcription",
        value=st.session_state["transcript"],
        placeholder="The transcription will appear here or add your own transcription"
    )    
    st.divider()

    summary = SingleRequestConstructor().llm_summarize(provided_text, llm_manager)

    if summary not in st.session_state:
        st.session_state["summary"] = ""
    if summary and len(summary) > 200:
        st.session_state["summary"] = summary

    st.write(st.session_state["summary"])

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    st.sidebar.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')
page_footer()