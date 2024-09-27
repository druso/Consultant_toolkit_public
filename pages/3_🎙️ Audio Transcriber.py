import streamlit as st
import os
from src.file_manager import DataLoader
from src.request_processor import SingleRequestConstructor
from src.external_tools import LlmManager, AudioTranscribe
from src.setup import page_setup, page_footer

import io
from openai import OpenAI

page_config = {'page_title':"Audio Transcriber",
          'page_icon':"🎙️",}

page_setup(page_config)

if st.session_state["authentication_status"]:

    uploaded_file, file_name = DataLoader("audio").load_user_file()
    llm_manager = LlmManager("streamlit",st.session_state["app_logger"])
    audio_transcriber = AudioTranscribe(st.session_state["app_logger"])
    transcript = ""

    if uploaded_file:
        
        tabs= st.tabs(["Transcribe","Summarize"])
        with tabs[0]: 
            if st.button("Transcribe!", use_container_width=True, type="primary"):
                transcript = audio_transcriber.whisper_openai_transcribe(uploaded_file)
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