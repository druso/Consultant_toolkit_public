from src.file_manager import DataLoader, DataFrameProcessor
from src.external_tools import LlmManager, SerpApiManager, WebScraper, OxyLabsManager, openai_advanced_uses
from src.request_processor import DfRequestConstructor, openai_thread_setup
from src.setup import page_setup, page_footer
import streamlit as st
import pandas as pd

page_config = {'page_title':"Generative Excel",
          'page_icon':"📈",}
page_setup(page_config)


if st.session_state["authentication_status"]:
    app_logger = st.session_state["app_logger"]

    data_loader = DataLoader("dataframe", app_logger)

    if isinstance(data_loader.user_file, pd.DataFrame) and not data_loader.user_file.empty:
        user_df= data_loader.user_file
        df_processor = DataFrameProcessor(user_df)

        st.sidebar.divider()  
            
        llm_manager = LlmManager("streamlit",app_logger)
        serpapi_manager = SerpApiManager(app_logger)
        web_scraper = WebScraper(app_logger)
        oxylabs_manager = OxyLabsManager(app_logger)
        openai_advance_manager = openai_advanced_uses(app_logger)
        request_constructor=DfRequestConstructor(df_processor, app_logger)
        
          
        tabs = st.tabs(["LLM", "Google", "Amazon", "Crawler", "Table Handler", "Assistant Setup","Scheduler"])
        with tabs[0]:        
            df_processor = request_constructor.llm_request_single_column(llm_manager)
        with tabs[1]:
            df_processor = request_constructor.google_request_single_column(serpapi_manager, oxylabs_manager)    
        with tabs[2]:
            df_processor = request_constructor.oxylabs_request_single_column(oxylabs_manager)  
        with tabs[3]:
            df_processor = request_constructor.crawler_request_single_column(web_scraper, oxylabs_manager) 
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
            return st.session_state["app_logger"].to_excel(df_processor.processed_df)

        if st.button("prepare file to download",use_container_width=True):
            excel_file = st.session_state["app_logger"].to_excel(df_processor.processed_df)

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
    app_logger.streamlit_batches_status()
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    st.sidebar.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')
page_footer()