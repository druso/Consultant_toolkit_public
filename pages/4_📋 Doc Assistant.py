from src.file_manager import DataLoader
from src.external_tools import LlmManager
from src.request_processor import DfRequestConstructor, TextEmbeddingsProcessors, SingleRequestConstructor
from src.setup import page_setup, page_footer
import streamlit as st
import pandas as pd


page_config = {'page_title':"Doc Assistant",
          'page_icon':"📋",}
page_setup(page_config)


if 'chunks_df' not in st.session_state:
    st.session_state['chunks_df'] = pd.DataFrame()
doc_status = {}
if 'doc_status' not in st.session_state:
    st.session_state['doc_status'] = {'not_chunked':True,'not_studied':True }



if st.session_state["authentication_status"]:
    app_logger = st.session_state["app_logger"]
    data_loader = DataLoader("doc", app_logger)


    if data_loader.user_file:
        user_doc = data_loader.user_file
        llm_manager = LlmManager("streamlit",app_logger)
        st.write("1. First thing we need to chunk the text into more digestible bits")
        st.session_state['chunks_df'] = SingleRequestConstructor().text_chunker(user_doc, app_logger.file_name,st.session_state['chunks_df'],)
        if not st.session_state['chunks_df'].empty:   
            app_logger.log_excel(st.session_state['chunks_df']) 
            st.session_state['doc_status']['not_chunked']=False 
        
        request_constructor=DfRequestConstructor(st.session_state['chunks_df'], app_logger)

        st.divider()
        st.write("2. Then the LLM can 'study' the contents. " + (
        "But you need to chunk it first, press the button above my love and start the magic!"
        if st.session_state['doc_status']['not_chunked']
        else "It may take a moment, depending on the lenght of the loaded document."
        ))


        if st.button("Study the content", disabled=st.session_state['doc_status']['not_chunked'], use_container_width=True):  
            st.session_state['chunks_df'] = request_constructor.llm_embed_single_column(llm_manager,config_package={"query_column":"chunk", "response_column":"embedding"})
            app_logger.log_excel(st.session_state['chunks_df'])
            st.session_state['doc_status']['not_studied']=False
        
        st.divider()
        st.write("3. Finally you can ask your question, " + (
        "But you should follow the steps above first dear friend."
        if st.session_state['doc_status']['not_chunked']
        else "Remember you can try different LLMs by choosing them on the sidebar."
        ))

        sys_msg_template="# RELEVANT CONTENT"
        usr_msg_template="""You need to assist a user providing a short, straight and synthetic response.
            You base your short answer only on the provided RELAVANT CONTENT. 
            Avoid repetition, be short.
            If the info is not available just reply 'sorry the info appears to be not available'"""

        response, combined_chunk, st.session_state['chunks_df'] = SingleRequestConstructor().llm_df_rag_request(st.session_state['chunks_df'],sys_msg_template, usr_msg_template,llm_manager)
        
        if response:
            st.text_area("response",response)

        # Display the combined chunk
        if combined_chunk:
            st.text_area("Most Relevant Chunk (with context):", combined_chunk)


        with st.expander("Full Content Table", expanded=False):
            st.write("Here you can see how your document is chunked and sorted based on your question")
            st.dataframe(st.session_state['chunks_df'], use_container_width=True, hide_index=False, )
    

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    st.sidebar.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')
page_footer()