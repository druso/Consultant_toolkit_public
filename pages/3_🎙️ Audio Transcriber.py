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
    app_logger = st.session_state["app_logger"]
    credential_manager = st.session_state['credential_manager']

    data_loader = DataLoader("audio", app_logger)

    if data_loader.user_file:

        llm_manager = LlmManager(app_logger,credential_manager)
        llm_manager = configure_llm_streamlit(llm_manager, LlmManager, app_logger)
        audio_transcriber = AudioTranscribe(app_logger, st.session_state['credential_manager'])
        transcript = ""
        
        tabs= st.tabs(["Transcribe","Summarize"])
        with tabs[0]: 
            if st.button("Transcribe!", use_container_width=True, type="primary"):
                transcript = audio_transcriber.whisper_openai_transcribe(data_loader.user_file)
            st.text_area("Your Transcription",transcript,placeholder="The transcription will apper here")

            st.download_button(
                    label="Download Text File",
                    data=transcript.encode('utf-8'),  # Convert the string to bytes
                    file_name="sample_text.txt",
                    mime="text/plain"
                )
            
        with tabs[1]:
            if not transcript:
                st.write("Transcribe something first")

            else:
                SingleRequestConstructor().llm_summarize(transcript, llm_manager)
                st.download_button(
                        label="Download Text File",
                        data=transcript.encode('utf-8'),  # Convert the string to bytes
                        file_name="sample_text.txt",
                        mime="text/plain"
                    )

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    st.sidebar.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')
page_footer()