import logging
logger = logging.getLogger(__name__)

from src.file_manager import DataFrameProcessor, BatchRequestLogger
from src.external_tools import LlmManager, SerpApiManager, WebScraper, OxyLabsManager, openai_advanced_uses
from src.streamlit_interface import DfRequestConstructor, openai_thread_setup
from src.streamlit_interface import dataframe_streamlit_handler, sync_streamlit_processed_df, streamlit_batches_status
from src.streamlit_setup import page_setup, page_footer, DataLoader, configure_llm_streamlit
import streamlit as st
import pandas as pd

page_config = {'page_title':"Generative Excel",
          'page_icon':"📈",}
page_setup(page_config)


if st.session_state["authentication_status"]:
    session_logger = st.session_state["session_logger"]
    credential_manager = st.session_state['credential_manager']
    data_loader = DataLoader("dataframe", session_logger)

    if isinstance(data_loader.user_file, pd.DataFrame) and not data_loader.user_file.empty:
        user_df= data_loader.user_file
        df_processor = DataFrameProcessor(user_df)
        df_processor = dataframe_streamlit_handler(df_processor)

        st.sidebar.divider()  
            
        llm_manager = LlmManager(session_logger,credential_manager)
        llm_manager = configure_llm_streamlit(llm_manager, LlmManager, session_logger)
        
        serpapi_manager = SerpApiManager(session_logger, credential_manager)
        web_scraper = WebScraper(session_logger)
        oxylabs_manager = OxyLabsManager(session_logger,credential_manager)
        openai_advance_manager = openai_advanced_uses(session_logger, credential_manager)
        request_constructor=DfRequestConstructor(df_processor, session_logger)
        
          
        tabs = st.tabs(["LLM", "Google", "Amazon", "Crawler", "Table Handler", "Assistant Setup","Scheduler"])
        with tabs[0]:        
            df_processor = request_constructor.llm_request_single_column(llm_manager)
            sync_streamlit_processed_df(df_processor)
        with tabs[1]:
            df_processor = request_constructor.google_request_single_column(serpapi_manager, oxylabs_manager)  
            sync_streamlit_processed_df(df_processor)  
        with tabs[2]:
            df_processor = request_constructor.oxylabs_request_single_column(oxylabs_manager)  
            sync_streamlit_processed_df(df_processor)
        with tabs[3]:
            df_processor = request_constructor.crawler_request_single_column(web_scraper, oxylabs_manager) 
            sync_streamlit_processed_df(df_processor)
        with tabs[4]:
            df_processor = request_constructor.df_handler()
        with tabs[5]:
            openai_thread_setup(openai_advance_manager).streamlit_interface(df_processor.processed_df)
        with tabs[6]:
            st.write("### Scheduler - Work in Progress")
            

        st.divider()
        # Show a preview of the processed file
        st.write("## Preview of your processed file")
        st.dataframe(df_processor.processed_df.head(100), use_container_width=True, hide_index=True,)

        st.session_state['processed_df'] = df_processor.processed_df.astype(str).fillna('')

        def get_excel():
            return st.session_state["session_logger"].to_excel(df_processor.processed_df)

        if st.button("prepare file to download",use_container_width=True):
            excel_file = st.session_state["session_logger"].to_excel(df_processor.processed_df)

            st.download_button(
                label="Download Excel file",
                data=excel_file,
                file_name="processed_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary",
                key="download_button_footer"
            )
    st.divider()
    batch_request_logger = BatchRequestLogger(session_logger.user_id, session_logger.session_id, session_logger.tool_config)
    streamlit_batches_status(batch_request_logger)
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    st.sidebar.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')
page_footer()