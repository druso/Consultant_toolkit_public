from src.file_manager import DataLoader, DataUtilities
from src.external_tools import LlmManager, SerpApiManager, WebScraper, OxyLabsManager
from src.request_processor import DfRequestConstructor
from src.setup import page_setup, page_footer
import streamlit as st
import pandas as pd

page_config = {'page_title':"Generative Excel",
          'page_icon':"📈",}

if 'processed_df' not in st.session_state:
    st.session_state['processed_df'] = pd.DataFrame()


page_setup(page_config)

if st.session_state["authentication_status"]:

    user_df, file_name = DataLoader("dataframe").load_user_file()
    def reset_process():
        st.session_state['processed_df'] = user_df.copy()
        request_constructor=DfRequestConstructor(st.session_state['processed_df'], st.session_state["app_logger"])
        return request_constructor

    if user_df.empty:
        st.write("## Load an excel or csv to start")
    else:
        st.session_state["app_logger"].set_file_name(file_name)
        st.session_state["app_logger"].log_excel(user_df,"original")
        
        # load the available column from the df,  generate a copy for processing and setup the llm_df_manager

        if st.session_state['processed_df'].empty:
            st.session_state['processed_df'] = user_df.copy()


        request_constructor=DfRequestConstructor(st.session_state['processed_df'], st.session_state["app_logger"])
        if st.sidebar.button("Refresh column list", use_container_width=True):
            request_constructor._column_refresh() 

            

        llm_manager = LlmManager("streamlit",st.session_state["app_logger"])
        serpapi_manager = SerpApiManager(st.session_state["app_logger"])
        web_scraper = WebScraper(st.session_state["app_logger"])
        oxylabs_manager = OxyLabsManager(st.session_state["app_logger"])
        utilities = DataUtilities()
        st.sidebar.divider()
        if st.sidebar.button('Reset Processing', use_container_width=True, help="Will restore the file to the originally uploaded file"):
            request_constructor = reset_process()

        

        tabs = st.tabs(["LLM", "SerpAPI", "Crawler", "Amazon", "Table Handler"])
        with tabs[0]:        
            st.session_state['processed_df'] = request_constructor.llm_request_single_column(llm_manager)
        with tabs[1]:
            st.session_state['processed_df'] = request_constructor.serpapi_request_single_column(serpapi_manager)    
        with tabs[2]:
            st.session_state['processed_df'] = request_constructor.crawler_request_single_column(web_scraper)    
        with tabs[3]:
            st.session_state['processed_df'] = request_constructor.oxylabs_request_single_column(oxylabs_manager)
        with tabs[4]:
            st.session_state['processed_df'] = request_constructor.df_handler()


        
        st.divider()
        # Show a preview of the processed file
        st.write("## Preview of your processed file")
        
        # Button to download the DataFrame as an Excel file
        with st.expander("Column Unroller *BETA*"):
            st.write("If you have structured in a column cells you can unroll it with this function. It will generate new lines for lists, new columns for dictionaries. Works for SerpAPI results, LLM structured response...")
            expander_msg_column = st.selectbox("Select column to expand", 
                                            options=st.session_state['processed_df'].columns.tolist(),
                                            help="The column where you have structured content that needs to be unrolled")
            if st.button("expand column objects"):
                st.session_state['processed_df'] = utilities.unroll_json_in_dataframe(st.session_state['processed_df'],expander_msg_column)
        
        st.dataframe(st.session_state['processed_df'], use_container_width=True, hide_index=True,)
        st.session_state["app_logger"].log_excel(user_df)
        #Reload the constructor with the latest version of the dataframe
        request_constructor=DfRequestConstructor(st.session_state['processed_df'], st.session_state["app_logger"])

        
        st.session_state["app_logger"].log_excel(st.session_state['processed_df'])
        df_excel = utilities.to_excel(st.session_state['processed_df'])
        st.download_button(
            label="Download Excel file",
            data=df_excel,
            file_name="processed_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary",
            key="download_button_footer"
        )
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    st.sidebar.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')
page_footer()