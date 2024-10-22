from src.prompts import sysmsg_keyword_categorizer,sysmsg_review_analyst_template, sysmsg_summarizer_template


from src.external_tools import LlmManager, StopProcessingError, RetryableError, SkippableError, openai_advanced_uses
from streamlit_autorefresh import st_autorefresh
from src.file_manager import DataFrameProcessor, SessionLogger, OpenaiThreadLogger, BatchRequestLogger, BatchSummaryLogger
from src.batch_handler import DfBatchesConstructor
import streamlit as st
import pandas as pd
import tiktoken
import math
import time
import base64
import os
from sklearn.metrics.pairwise import cosine_similarity


from openai.types.beta.assistant_stream_event import (
    ThreadRunStepCreated,
    ThreadRunStepDelta,
    ThreadRunStepCompleted,
    ThreadMessageCreated,
    ThreadMessageDelta
    )

from openai.types.beta.threads.text_delta_block import TextDeltaBlock 
from openai.types.beta.threads.runs.tool_calls_step_details import ToolCallsStepDetails
from openai.types.beta.threads.runs.code_interpreter_tool_call import (
    CodeInterpreterOutputImage,
    CodeInterpreterOutputLogs
    )


class DfRequestConstructor():
    def __init__(self, df_processor:DataFrameProcessor, session_logger:SessionLogger=None):
        """
        Initialize the streamlit df request constructor
        """

        self.df_processor = df_processor
        self.session_logger = session_logger
        self.batch_request_logger = BatchRequestLogger(session_logger.user_id, session_logger.session_id, session_logger.tool_config)
        self.batches_constructor = DfBatchesConstructor(df_processor, session_logger)

        self.bulk_sys_templates = {
            "Custom Prompt": "",
            "Keyword categorization": sysmsg_keyword_categorizer,
            "Review Analysis": sysmsg_review_analyst_template,
            "Text Summarizer": sysmsg_summarizer_template,
        }
            
    
    def _base_batch_streamlit(self,function,function_ready=True, query_name="query_name", response_name="response_name", function_name="function_name", config_package=None,**kwargs):
        
        if config_package:
    
                self.batches_constructor.df_batches_handler(function, 
                                    batch_size=config_package.get('batch_size', 0), 
                                    response_column=config_package.get('response_column', 'response'), 
                                    query_column=config_package.get('query_column'), 
                                    **kwargs
                                    )

        else:
            default_query_value='Select a column...'
            query_column = st.selectbox(f"Select columns use as {query_name}", 
                                        options=[default_query_value] + self.df_processor.processed_df.columns.tolist(), help="This is the input colum. Each row will be processed through the function")
            batch_size = st.number_input(f"{function_name} requests test batch size",min_value=1,max_value=10,step=1,value=3, help="Here you set how many rows to process with a click.")
            response_column = st.text_input(f"{response_name} Column Name",response_name, help="The response from the function will be stored in a column with the name you provide here")
            if query_column == response_column:
                st.warning("*You should select a different column for the response than what you're using as the input or you'll get into a paradox*")
                waiting_input=True
            elif query_column == default_query_value:
                st.warning("*No column selected. Please choose a column.*")
                waiting_input=True
            elif not function_ready:
                st.warning("*You still need to configure something above*")
                waiting_input=True 
            else:
                waiting_input=False
            
            #############TEST BATCH RUN################
            if st.button(f"Run a {function_name} test batch", use_container_width=True, disabled=waiting_input):
                progress_bar = st.progress(0, text="Preparing the operation")
                df=self.df_processor.processed_df
                total_to_process = batch_size if batch_size > 0 else len(df[df[response_column].isna()].index)
                try:
                    for progress in self.batches_constructor.df_batches_handler(
                        func=function, 
                        response_column=response_column, 
                        query_column=query_column, 
                        batch_size=batch_size, 
                        **kwargs
                    ):
                        if progress["df"] is None:
                            processed_count =progress["processed_count"]
                            if processed_count > batch_size:
                                processed_count=batch_size
                            progress_bar.progress(int(processed_count / batch_size * 100), text=f"Processing... {processed_count}/{total_to_process}")
                        else:
                            progress_bar.empty()
                            st.toast("Processing complete. Check the results below")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

            #############FULL FILE RUN################      
            if st.button(f"Run {function_name} on the whole file", use_container_width=True, disabled=waiting_input):
                if query_column == default_query_value:
                    st.warning("*No column selected. Please choose a column.*")
                else:
                    self.batch_request_logger.post_scheduler_request(
                                                    df=self.df_processor.processed_df,
                                                    session_logger=self.session_logger,
                                                    function=function, 
                                                    query_column=query_column, 
                                                    response_column=response_column, 
                                                    kwargs=kwargs
                                                    )
                    st.toast("Request sent to the scheduler", icon='🚀')
    
    def llm_request_single_column(self, llm_manager, config_package=None):
        if config_package:
            self._base_batch_streamlit(function=llm_manager.llm_request, config_package=config_package, sys_msg=config_package['sys_msg'])
        else:
            st.write("### Batch Requests to LLM")
            st.write("Here you can take a column and massively process it through an LLM. Provide instructions as System Message")

            selected_sysmsg_template = st.selectbox("System Prompt Template",options=list(self.bulk_sys_templates.keys()),)
            sysmsg_template = self.bulk_sys_templates.get(selected_sysmsg_template, "No option selected") 
            sys_msg = st.text_area("System Message",
                        value= sysmsg_template,
                        height=300,
                        placeholder="Write here your instructions or select and customize one of the templates above",
                        help="System message is where you provide instructions that will be used to process each content of the column you select"
                        )
            if not sys_msg:
                function_ready = False
            else:
                function_ready = True

            self._base_batch_streamlit(
                function=llm_manager.llm_request,
                function_ready = function_ready,
                query_name="Content to LLM", 
                response_name="LLM Response",
                function_name="LLM",
                sys_msg=sys_msg,
                llm_temp=llm_manager.llm_temp,
                llm_model=llm_manager.model
                )
        return self.df_processor


    def llm_embed_single_column(self, llm_manager, config_package=None):
        if config_package:
            self._base_batch_streamlit(function=llm_manager.embed_query, config_package=config_package, force_string=False)
        else:
            st.write("### Batch Embedding Requests")
            st.write("Here you can take a column and massively generate embeddings for each value")
            
            self._base_batch_streamlit(function=llm_manager.embed_query, 
                                    query_name="Content to embed", 
                                    response_name="Embeddings",
                                    function_name="LLM",
                                    force_string=False)
        return self.df_processor.processed_df


    def google_request_single_column(self, serpapi_manager, oxylabs_manager, config_package=None):
        if config_package:
            self._base_batch_streamlit(function=serpapi_manager.extract_organic_results, config_package=config_package, force_string=False)
        else:
            st.write("### Batch Requests to Crawl Google")
            st.write("Here you can take a column and search each item on Google")

            crawler_type = st.selectbox("Select the crawler", options=["Oxylabs", "SerpAPI"], help="Select the crawler you want to use")

            if crawler_type == "SerpAPI":
                country=st.selectbox("country",['it','us','uk','fr','de','es']) 
                num_results = st.number_input("Number of results per query",min_value=1,max_value=10,step=1,value=3, help="The number of search results to save in the response")

                last_years = st.number_input("N of Past Years",min_value=0,max_value=10,step=1,value=0, help="Limit results to certain past years. Set to 0 for no limit")

                self._base_batch_streamlit(function=serpapi_manager.extract_organic_results, 
                                        query_name="Content to search", 
                                        response_name="Google Results",
                                        function_name="Google Search",
                                        num_results=num_results,
                                        country=country,
                                        last_years=last_years,)
                
            if crawler_type == "Oxylabs":

                domain=st.selectbox("country",['it','us','uk','fr','de','es']) 
                num_results = st.number_input("Number of results per query",min_value=1,max_value=1000,step=1,value=3, help="The number of search results to save in the response")

                last_years = st.number_input("N of Past Years",min_value=0,max_value=10,step=1,value=0, help="Limit results to certain past years. Set to 0 for no limit")

                self._base_batch_streamlit(function=oxylabs_manager.serp_paginator, 
                                        query_name="Content to search", 
                                        response_name="Google Results",
                                        function_name="Google Search",
                                        num_results=num_results,
                                        domain=domain,
                                        last_years=last_years,
                                        #status_bar=True,
                                        )


        return self.df_processor


    def crawler_request_single_column(self, web_scraper,oxylabs_manager, config_package=None):
        if config_package:
            self._base_batch_streamlit(function=web_scraper.url_simple_extract, config_package=config_package, force_string=False)
        else:
            st.write("### Batch Crawling Requests")
            st.write("Here you can take links stored in a column and crawl them massively")
            crawler_type = st.selectbox("Select the crawler", options=["Oxylabs", "Generic"], help="Select the crawler you want to use")

            if crawler_type == "Oxylabs":
                self._base_batch_streamlit(function=oxylabs_manager.web_crawler, 
                                        query_name="Link to crawl", 
                                        response_name="Crawled content",
                                        function_name="Crawl",)
            
            else:
                self._base_batch_streamlit(function=web_scraper.url_simple_extract, 
                                        query_name="Link to crawl", 
                                        response_name="Crawled content",
                                        function_name="Crawl",)
        return self.df_processor


    def oxylabs_request_single_column(self, oxylabs_manager, config_package=None):
        if config_package:
            self._base_batch_streamlit(function=oxylabs_manager.extract_amazon_product_info, config_package=config_package, force_string=False)
        else:
            st.write("### Batch Amazon Requests")
            st.write("Here you can take asins code from a column and crawl them massively")
        
            amazon_domain = st.selectbox("Select the amazon website country", 
                                        options=['it', 'de', 'es', 'fr', 'com', 'co.uk', 'co.jp', 'ae', 'ca', 'cn', 'com.au', 'com.be', 'com.br', 'com.mx', 'com.tr', 'eg', 'in', 'nl', 'pl', 'sa', 'se'],
                                        help="The column that will be passed as query to the oxylabs call") 

            function = st.selectbox("Info to extract",['Product Info','Reviews'])
            if function == 'Product Info':
                self._base_batch_streamlit(function=oxylabs_manager.get_amazon_product_info, 
                                        query_name="ASIN to retrieve", 
                                        response_name="Amazon info",
                                        function_name="Amazon search",
                                        amazon_domain=amazon_domain)

            elif function == 'Reviews':
                review_pages = st.number_input("Review Pages per product",min_value=1,max_value=5,step=1,value=1, help="The number of search results to save in the response")
                
                self._base_batch_streamlit(function=oxylabs_manager.get_amazon_review, 
                                        query_name="ASIN to retrieve", 
                                        response_name="Amazon info",
                                        function_name="Amazon search",
                                        amazon_domain=amazon_domain,
                                        review_pages=review_pages)
        return self.df_processor


    
    def df_handler(self):
        st.write("### Table Handler")
        available_columns = st.session_state['available_columns']
        tabs = st.tabs(["Unroll", "Drop", "Merge", "Add", "Rename", "Parse"])
        #column unroller
        with tabs[0]:
            st.write ("### Column Unroller")
            st.write("If you have structured in a column cells you can unroll it with this function. It will generate new lines for lists, new columns for dictionaries. Works for SerpAPI results, LLM structured response...")
            expander_msg_column = st.selectbox("Select column to expand", 
                                            options=available_columns,
                                            help="The column where you have structured content that needs to be unrolled")
            if st.button("expand column objects"):
                self.session_logger.log_excel(self.df_processor.processed_df,version="before_unrolling")
                self.df_processor.unroll_json(expander_msg_column)


        #column drop
        with tabs[1]:
            st.write ("### Column Dropper")
            columns_to_drop = st.multiselect("Columns to drop",available_columns)
            st.write (f"You do know that by pressing this you will erase the columns {columns_to_drop} forever, right?")
            if st.button("Yes, Drop Them!", type="primary"):
                self.df_processor.processed_df=self.df_processor.processed_df.drop(columns=columns_to_drop, errors='ignore')

        #column merge
        with tabs[2]:
            st.write ("### Column Merger")
            columns_to_merge = st.multiselect("Columns to merge",available_columns)
            separator = st.text_input("Separator")
            new_column_name = st.text_input("New Column Name")
            if st.button("Merge columns!", type="primary"):
                self.df_processor.processed_df[new_column_name] = self.df_processor.processed_df[columns_to_merge].apply(
                    lambda x: separator.join(x.astype(str)), axis=1
                )


        #column add
        with tabs[3]:
            st.write ("### Column Adder")
            new_column_content = st.text_input("New Column Content")
            new_column_name = st.text_input("New Column_Name")
            if st.button("Generate new column!", type="primary"):
                self.df_processor.processed_df[new_column_name] = new_column_content


        #column rename
        with tabs[4]:
            st.write("### Column Renamer")
            column_to_rename = st.selectbox("Select column to rename", available_columns)
            new_column_name = st.text_input("New name for the selected column")
            if st.button("Rename column!", type="primary"):
                if column_to_rename and new_column_name:
                    self.df_processor.processed_df = self.df_processor.processed_df.rename(columns={column_to_rename: new_column_name})
                    st.success(f"Column '{column_to_rename}' has been renamed to '{new_column_name}'")

                else:
                    st.warning("Please select a column and provide a new name")

        #Column parser
        with tabs[5]:
            st.write("### Column Parser")
            column_to_parse = st.selectbox("Select column to parse", available_columns)
            expected_type = st.selectbox("Select expected data type", 
                                         ["Numeric", "DateTime", "JSON", "List", "String/Other"])
            if st.button("Parse Column", type="primary"):
                original_type = self.df_processor.get_data_type(column_to_parse)
                self.df_processor.parse_column(column_to_parse, expected_type)
                new_type = self.df_processor.get_data_type(column_to_parse)

                
                if original_type != new_type:
                    st.success(f"Column '{column_to_parse}' has been parsed from {original_type} to {new_type}")
                    
                else:
                    st.info(f"No change in data type. Column '{column_to_parse}' remains as {new_type}")
                
                
        with st.expander("Table Recap", expanded=True):
            df = self.df_processor.processed_df
            total_rows, total_cols = df.shape
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # in MB
            
            try:
                duplicated_rows = df.duplicated().sum()
            except:
                duplicated_rows = "cannot compute"

            st.write(f"**Shape:** {total_rows} rows - {total_cols} columns | **Memory Usage:** {memory_usage:.2f} MB | **Total Null Values:** {df.isnull().sum().sum()} | **Duplicate Rows:** {duplicated_rows}")
            
            st.write("**Column Summary:**")
            column_summary = self.df_processor.generate_column_summary(df)
            st.dataframe(column_summary, use_container_width=True)
        

        return self.df_processor

    
class SingleRequestConstructor():
    def __init__(self):
        """
        ...
        """
        pass

    def llm_summarize(self,text, llm_manager:LlmManager):
        
        st.write("In doing the summarization the full transcript gets chopped into chunks:")
        chunk_length = st.slider("Chunk lenght (in tokens)", min_value=0,max_value=1600,value=800,step=1)
        chunk_overlap = st.slider("Chunk overlap (in tokens)", min_value=0,max_value=160, value=80 ,step=1)
        with st.expander:
            sys_msg = st.text_area("Summarizer Prompt", value=st.session_state['app_configs']['system_message'], height=200)
        
        if st.button("Summarize!", use_container_width=True, type="primary"):
            with st.spinner(' 🔄 Summarizing...'):
                chunks_list = TextEmbeddingsProcessors().text_token_chunker(text, chunk_length, chunk_overlap)
                st.toast(f"ready to summarize {len(chunks_list)} chunks", icon='✨')
                total_tokens = 0
                chunk_num = 1
                summarized_transcript = ""
                for chunk_num, chunk in enumerate(chunks_list, start=1):
                    assistant_response, used_tokens = llm_manager.llm_request(chunk, sys_msg)
                    summarized_transcript += f"\n\n\n\n# Summary of Chunk {chunk_num}\n\n{assistant_response}"
                    st.toast(f"Chunk number {chunk_num} summarized using {used_tokens} tokens", icon='⬆️')
                    total_tokens += used_tokens
                st.toast(body=f"All chunks summarized using a total of {total_tokens} tokens", icon='🚀')
                return summarized_transcript 
            
    def text_chunker(self,user_doc,file_name,df=pd.DataFrame()):  
        with st.expander("Chunker options"):
            chunk_length = st.slider("Chunk lenght (in tokens)", min_value=50,max_value=1000,value=100,step=1)
            chunk_overlap = st.slider("Chunk overlap (in tokens)", min_value=0,max_value=100, value=0 ,step=1)    
        if st.button("Chunk the content", use_container_width=True):
            chunks_list = TextEmbeddingsProcessors().text_token_chunker(user_doc, chunk_length, chunk_overlap)
            df = pd.DataFrame({
                "file_name": [file_name] * len(chunks_list),  
                "chunk": chunks_list
            })
            df['original_index'] = df.index.copy()
        return df
    
    def llm_df_rag_request(self,df,sys_msg_template, usr_msg_template,llm_manager:LlmManager):

        query = st.text_input("Your question", placeholder="What is the answer to life, the universe and everything?")
        with st.expander("Nerdy options"):
            num_context=st.slider(label="Context Lenght",min_value=0,max_value=5,value=1,step=1, help="The tool will retrieve the relevant chunk, but also stitch the neighbour chung to it. How many you ask? You set it right at this slider my dear.")
            num_chunk=st.slider(label="Number of relevant chunk",min_value=0,max_value=3,value=2,step=1, help="The tool will order chunks based on how relevant they are to your question, you may want to get more than just the most relevant. And that's something you choose right here my friend.")
        
        if st.button("Answer me", use_container_width=True, disabled=st.session_state['doc_status']['not_studied'], type="primary"):
            query_embedding = llm_manager.embed_query(query)
            
            combined_chunk, df = TextEmbeddingsProcessors().get_combined_chunk(df, 
                                                                        query_embedding, 
                                                                        num_context=num_context, 
                                                                        num_chunk=num_chunk)
            
            sys_msg=f"{sys_msg_template}\n{combined_chunk}"
            user_msg=f"{usr_msg_template}{query}"
            response = llm_manager.llm_request(user_msg=user_msg, 
                                                    sys_msg=sys_msg)

            return response, combined_chunk, df
        return None, None, df
        
class TextEmbeddingsProcessors():
    def __init__(self):
        """
        ...
        """
        pass

    def text_token_chunker(self, text, chunk_length, chunk_overlap):

        if chunk_length <= chunk_overlap:
            raise ValueError("Chunk length must be greater than chunk overlap")
        
        enc = tiktoken.encoding_for_model('gpt-3.5-turbo')
        encoding = tiktoken.get_encoding(enc.name)
        tokenslist = encoding.encode(text)

        total_effective_length = len(tokenslist) + chunk_overlap - chunk_length
        total_chunks = max(math.ceil(total_effective_length / (chunk_length - chunk_overlap)), 1)
        max_length = math.ceil(len(tokenslist) / total_chunks)

        chunk_list_token = [tokenslist[i:i + max_length] for i in range(0, len(tokenslist), max_length - chunk_overlap)]

        chunk_list = []
        for chunk in chunk_list_token:
            chunk_text = encoding.decode(chunk)
            chunk_list.append(chunk_text)

        return chunk_list
    
    def calculate_cosine_similarity(self,embedding1, embedding2):
        """Calculate cosine similarity between two embeddings."""
        return cosine_similarity([embedding1], [embedding2])[0][0]


    def get_combined_chunk(self,df, query_embedding, num_context=1, num_chunk=2):

        # Calculate similarities and add them as a new column
        df['similarity'] = df['embedding'].apply(lambda emb: self.calculate_cosine_similarity(query_embedding, emb))
        # Sort the DataFrame by similarity in descending order and reset the index
        df = df.sort_values(by='similarity', ascending=False).reset_index(drop=True)
        top_indices = df.index[:num_chunk]

        """similarity_series = df['embedding'].apply(lambda emb: self.calculate_cosine_similarity(query_embedding, emb))

        # Get indices of top 'num_chunk' most similar chunks
        top_indices = similarity_series.nlargest(num_chunk).index"""

        # Determine the range of chunks to include based on context window for each relevant chunk
        selected_indices = set()  # Use a set to avoid duplicates
        for top_index in top_indices:
            min_index = max(0, top_index - num_context)
            max_index = min(len(df) - 1, top_index + num_context)
            selected_indices.update(range(min_index, max_index + 1)) 

        # Combine the selected chunks
        combined_chunk = ""
        for i in sorted(selected_indices): # Sort the indices before combining
            if i - 1 not in selected_indices and i > 0:  # Check for gap
                combined_chunk += "[...]\n\n"
            combined_chunk += st.session_state['chunks_df'].loc[i, 'chunk']

        return combined_chunk, df

class openai_thread_setup():
    def __init__(self, openai_advanced_uses:openai_advanced_uses):
        self.session_logger = st.session_state["session_logger"]
        self.openai_advanced_uses = openai_advanced_uses

    def __assistant_setup_msg_generator(self, df: pd.DataFrame) -> str:
        shape = f"{df.shape[0]} rows, {df.shape[1]} cols"
        
        # Column information with unique values for object columns
        col_info = ",".join([f"{col}({df[col].dtype},unique={df[col].nunique()})" if df[col].dtype == 'object' 
                            else f"{col}({df[col].dtype})" for col in df.columns])
        
        return f"understood\nAnalysis of attached xlsx file:\nShape:{shape}\nColumns:{col_info}."


    def streamlit_interface(self, df):
                
        st.write("### Assistant Setup")
        thread_name = st.text_input("Thread name",self.session_logger.file_name,max_chars=30,help="This is how you can later select the thread in **🤖My Assistants**" )
        
        goals_example="The objective is to uncover how your target audience talks about competing products and brands, revealing their pain points, desires, and preferences. This insight will highlight opportunities for enhancing our product development and refining your communication strategy to better resonate with potential customers."
        goals=st.text_area("Goal of the analysis", goals_example, help="This will help the assistant understanding the broad scope of the analysis", height=200)

        file_description_example="""Each row of the file is a review available for key products from Augustings Bader, La Mer and Bluelagoon Skincare, from a selection of websites. It contains the rating, the source of the review, timestamp as well as keywords, emotions expressed and an evaluation of the experience expressed in the review along few categories."""
        file_description=st.text_area("Description of the file", file_description_example,help="This will help the assistant understand the content of the file you're providing", height=200)
        
        user_setup_msg = f"#GOALS\n{goals}\n#FILE DESCRIPTION\n{file_description}"

        if st.button("Create Thread"):
            #Save and upload the file
            with st.spinner("Uploading the file..."):
                file_path = self.session_logger.log_excel(df,"assistant", True)
                uploaded_file = self.openai_advanced_uses.openai_upload_file(file_path, "assistants")
                file_id = uploaded_file.id
                assistant_setup_msg = self.__assistant_setup_msg_generator(df)

            #Generate the thread
            with st.spinner("Requesting the setup of the thread..."):
                thread = self.openai_advanced_uses.setup_thread(thread_name, file_id, user_setup_msg, assistant_setup_msg)
                
            st.write(f"Thread {thread_name} was set up correctly, you can now use it in the **🤖My Assistants** tab")

class assistant_interface():
    def __init__(self, openai_advanced_uses:openai_advanced_uses):
        self.session_logger = st.session_state["session_logger"]
        self.openai_thread_logger = OpenaiThreadLogger(self.session_logger.user_id, self.session_logger.tool_config)
        self.openai_advanced_uses = openai_advanced_uses
        self.unavailable = False

        self.setup_assistant()
        self.setup_thread()
        self.thread_buttons()

        self.assistant_chat_interface()

    def setup_assistant(self):
            assistant_name_filter = st.session_state['tool_config']['assistant_configs']['assistant_name']
            assistant_ids = [aid for aid in self.openai_advanced_uses.list_assistants() if aid[1] == assistant_name_filter]
            
            if assistant_ids:
                selected_assistant = st.sidebar.selectbox("Select assistant", [name for _, name in assistant_ids])
                self.assistant_id = next(id for id, name in assistant_ids if name == selected_assistant)
                self.unavailable = False
            else:
                st.sidebar.write("No assistants available")
                self.unavailable = True
                self.starting_message = "You need to create an assistant in **🔧 Settings & Recovery** before you can use this tool"


    def setup_thread(self):
        thread_ids = self.openai_thread_logger.list_openai_threads()
        if thread_ids:
            selected_thread = st.sidebar.selectbox("Select thread", [name for _, name in thread_ids])
            self.thread_id = next(id for id, name in thread_ids if name == selected_thread)
            st.session_state['messages'] = self.openai_thread_logger.get_thread_history(self.thread_id)
            self.starting_message = f"Ask anything to the thread {selected_thread}"
            self.unavailable = False
        else:
            st.sidebar.write("No threads available")
            st.session_state['messages'] = []
            message = "You need to create a thread using **📈 Generative Excel** before you can use this tool"
            st.sidebar.warning(message)
            self.starting_message = message
            self.unavailable = True


    def thread_buttons(self):
        if not self.unavailable and st.sidebar.button("reset thread", use_container_width=True):
            self.thread_id = self.openai_advanced_uses.reset_thread(self.thread_id)
            st.session_state['messages'] = []
            self.st.success(f"New Thread id generated: {self.thread_id}")
            

        if not self.unavailable and st.sidebar.button("erase thread", use_container_width=True):
            self.thread_id = self.openai_advanced_uses.erase_thread(self.thread_id)
            st.session_state['messages'] = []
            self.st.success(f"Thread data for thread id: {self.thread_id} has been deleted.")

        
        
    def assistant_chat_interface(self):  

        for message in st.session_state['messages']:
            with st.chat_message(message["role"]):
                for item in message["items"]:
                    item_type = item["type"]
                    if item_type == "text":
                        st.markdown(item["content"])
                    elif item_type == "image":
                        for image in item["content"]:
                            st.html(image)
                    elif item_type == "code_input":
                        with st.status("Code", state="complete"):
                            st.code(item["content"])
                    elif item_type == "code_output":
                        with st.status("Results", state="complete"):
                            st.code(item["content"])

        if message_content := st.chat_input(self.starting_message, disabled=self.unavailable):

            st.session_state['messages'].append({"role": "user",
                                            "items": [
                                                {"type": "text", 
                                                "content": message_content
                                                }]})
            
            self.openai_advanced_uses.post_message(self.thread_id, message_content)


            with st.chat_message("user"):
                st.markdown(message_content)

            with st.chat_message("assistant"):
                stream = self.openai_advanced_uses.stream_response(self.thread_id, self.assistant_id)

                assistant_output = []

                for event in stream:

                    if isinstance(event, ThreadRunStepCreated):
                        if event.data.step_details.type == "tool_calls":
                            assistant_output.append({"type": "code_input", "content": ""})
                            code_input_expander = st.status("Writing code ⏳ ...", expanded=True)
                            code_input_block = code_input_expander.empty()

                    elif isinstance(event, ThreadRunStepDelta):
                        if hasattr(event.data.delta.step_details, 'tool_calls'):
                            tool_calls = event.data.delta.step_details.tool_calls
                            if tool_calls and tool_calls[0].code_interpreter:
                                code_interpreter = tool_calls[0].code_interpreter
                                code_input_delta = code_interpreter.input
                                if code_input_delta:
                                    assistant_output[-1]["content"] += code_input_delta
                                    code_input_block.empty()
                                    code_input_block.code(assistant_output[-1]["content"])

                    elif isinstance(event, ThreadRunStepCompleted):
                        if hasattr(event.data.step_details, 'tool_calls'):
                            tool_calls = event.data.step_details.tool_calls
                            if tool_calls:
                                code_interpreter = tool_calls[0].code_interpreter
                                if code_interpreter.outputs:
                                    code_input_expander.update(label="Code", state="complete", expanded=False)
                                    try:
                                        output = code_interpreter.outputs[0]

                                        if isinstance(output, CodeInterpreterOutputImage):
                                            image_html_list = []
                                            for output in code_interpreter.outputs:
                                                image_file_id = output.image.file_id
                                                image_data = self.openai_advanced_uses.openai_download_file(image_file_id)
                                                image_folder = f"{self.session_logger.openai_threads_folder}/{self.thread_id}"
                                                os.makedirs(image_folder, exist_ok=True)
                                                image_path = f"{self.session_logger.openai_threads_folder}/{self.thread_id}/{image_file_id}.png"
                                                with open(image_path, "wb") as file:
                                                    file.write(image_data)

                                                with open(image_path, "rb") as file_:
                                                    data_url = base64.b64encode(file_.read()).decode("utf-8")

                                                image_html = f'<p align="center"><img src="data:image/png;base64,{data_url}" width=600></p>'
                                                st.html(image_html)
                                                image_html_list.append(image_html)

                                            assistant_output.append({"type": "image", "content": image_html_list})

                                        elif isinstance(output, CodeInterpreterOutputLogs):
                                            code_output = output.logs
                                            assistant_output.append({"type": "code_output", "content": code_output})
                                            with st.status("Results", state="complete"):
                                                st.code(code_output)
                                    except Exception as e:
                                        pass
                                    finally:
                                        code_input_expander.update(label="Code", state="complete", expanded=False)
                                else:
                                    code_input_expander.update(label="Code", state="complete", expanded=False)

                    elif isinstance(event, ThreadMessageCreated):
                        assistant_output.append({"type": "text", "content": ""})
                        assistant_text_box = st.empty()

                    elif isinstance(event, ThreadMessageDelta):
                        if isinstance(event.data.delta.content[0], TextDeltaBlock):
                            new_text = event.data.delta.content[0].text.value or ""
                            last_output = assistant_output[-1] if assistant_output else {"type": "text", "content": ""}
                            last_output["content"] += new_text
                            if not assistant_output:
                                assistant_output.append(last_output)
                            assistant_text_box.markdown(last_output["content"])
                
            st.session_state['messages'].append({"role": "assistant", "items": assistant_output})
            self.openai_thread_logger.update_thread_history(st.session_state['messages'],self.thread_id)

def sync_streamlit_processed_df(df_processor:DataFrameProcessor):
    st.session_state['processed_df'] = df_processor.processed_df
    st.session_state['available_columns'] = df_processor.processed_df.columns.tolist()

def dataframe_streamlit_handler(df_processor:DataFrameProcessor):
        
        if 'processed_df' not in st.session_state or st.session_state['processed_df'] is None:
            st.session_state['processed_df'] = df_processor.processed_df
        if 'available_columns' not in st.session_state:
            st.session_state['available_columns'] = df_processor.processed_df.columns.tolist()
        
        df_processor.processed_df = st.session_state['processed_df']

        if st.sidebar.button("Refresh columns list", use_container_width=True, 
                                  help="Will force the refresh of the lists of available columns throughout the application"):
            st.session_state['available_columns'] = df_processor.processed_df.columns.tolist()

        if st.sidebar.button('Restore Processing', use_container_width=True, 
                            help="Will restore the file to the originally uploaded file"):
            st.session_state["session_logger"].reinitialize_session_id()
            st.session_state["session_logger"].log_excel(df_processor.user_df, "original")
            st.session_state['processed_df'] = df_processor.user_df.copy()
            st.session_state['available_columns'] = st.session_state['processed_df'].columns.tolist()
            df_processor.processed_df = st.session_state['processed_df']

        return df_processor

class streamlit_batches_status():

    def __init__(self,  session_logger:SessionLogger):
        self.session_logger = session_logger
        self.batch_request_logger = BatchRequestLogger(self.session_logger.user_id,self.session_logger.session_id, self.session_logger.tool_config)  #makes no sense to activate this with session id
        self.batch_summary_logger = BatchSummaryLogger(self.session_logger.tool_config)
        self.update_batches_df()
        if not self.batches_df.empty:
            self.main_interface()


    def update_batches_df(self):

        summary_df = self.batch_request_logger.load_batches_summary()
        self.batches_df = summary_df
        if self.batches_df.empty:    
            st.warning("No batches available")
            self.batches_df = pd.DataFrame()
        else:
            self.batches_df = summary_df[summary_df.get('user_id') == self.batch_request_logger.user_id]
            column_to_datetime = ['schedule_time', 'start_time', 'end_time']
            for column_name in column_to_datetime:
                try:
                    self.batches_df[column_name] = pd.to_datetime(self.batches_df[column_name])
                except Exception as e:
                    st.warning(f"Failed to convert {column_name} to datetime: {str(e)}")

        if not self.batches_df.empty:
            self.completed_batches = self.batches_df[self.batches_df['status'] == 'COMPLETED'] 
            self.wip_batches = self.batches_df[self.batches_df['status'] == 'WIP']
            self.pending_batches = self.batches_df[self.batches_df['status'] == 'PENDING']

    
    def main_interface(self):
        count = st_autorefresh(interval=5000, key="main_autorefresh")
        self.update_batches_df()
        
        if self.batches_df.empty:
            st.warning("No batch data available.")
            return
        
        self.display_batches_recap()
        self.display_batches_data()
        tabs = st.tabs(["Check Progress", "Download Data", "Stop Batches"])
        with tabs[1]: 
            self.download_data_interface()
        with tabs[2]:
            self.stop_batch_interface()
        with tabs[0]:
            self.display_wip_batches_progress()


    def display_batches_recap(self):
        # Display recap of batches in the pipeline
        st.write("### Batches Recap")
        col1, col2, col3 = st.columns(3)
        col1.metric("Completed Batches", len(self.completed_batches))
        col2.metric("Wip Batches", len(self.wip_batches))
        col3.metric("Pending Batches", len(self.pending_batches))

    def display_batches_data(self):
        # Display the DataFrame
        st.write("### Batches Data")
        st.write("Overview of all batches:")
        columns_to_display = ['batch_id', 'status', 'input_file', 'session_id', 'function', 'schedule_time']  
        st.dataframe(self.batches_df[columns_to_display], hide_index=True, use_container_width=True, height=160)

    def display_wip_batches_progress(self):
        st.write("### Wip Batches Progress")
        
        if self.wip_batches.empty:
            st.info("No batches in progress")
            return

        # Use a placeholder to avoid stacking progress bars on each refresh
        progress_container = st.container()

        with progress_container:
            for _, batch in self.wip_batches.iterrows():
                progress = self.batch_request_logger.read_wip_progress(
                    batch['user_id'], batch['batch_id']
                )
                st.write(f"Progress for {batch['batch_id']} on file {batch['input_file']}: {progress}%")
                st.progress(progress / 100)  
    



    def download_data_interface(self):
        st.write("### Download Completed Batches")
        st.write("Select a batch to download a result:")
        
        if not self.completed_batches.empty:
            batch_ids = self.completed_batches['batch_id'].tolist()
            default_batch_id = st.session_state.get('selected_batch_id', batch_ids[0])

            selected_batch_id = st.selectbox(
                "Select a completed batch to download:",
                options=batch_ids,
                index=batch_ids.index(default_batch_id),
                key='selected_batch_id'
            )

            if selected_batch_id:
                selected_row = self.completed_batches[self.completed_batches['batch_id'] == selected_batch_id].iloc[0]
                
                filename = f"processed_{selected_row['input_file']}"
                output_file_path = os.path.join(
                    self.batch_request_logger.tool_config['logs_root_folder'], 
                    self.batch_request_logger.user_id,
                    "files", 
                    selected_row['session_id'], 
                    filename
                )

                # Check if the output file exists
                if os.path.isfile(output_file_path):
                    with open(output_file_path, "rb") as file:
                        st.download_button(
                            label="Download Output File",
                            use_container_width=True,
                            data=file,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.warning("Output file not found for the selected batch.")
        else:
            st.info("No completed batches available for download.")




    def stop_batch_interface(self):
        st.write("### Stop Batches")
        st.write("Select a batch to post a stop request:")
        
        if not self.pending_batches.empty or not self.wip_batches.empty:
            stoppable_batches = pd.concat([self.pending_batches, self.wip_batches])
            filenames = stoppable_batches['filename'].tolist()
            
            # Safely retrieve the default filename
            default_filename = st.session_state.get('selected_filename', filenames[0] if filenames else None)
            
            # Check if the default_filename exists in filenames
            if default_filename not in filenames:
                default_filename = filenames[0] if filenames else None
                st.session_state['selected_filename'] = default_filename  # Update session state accordingly
            
            if filenames:
                try:
                    selected_filename = st.selectbox(
                        "Select a batch to stop:",
                        options=filenames,
                        index=filenames.index(default_filename) if default_filename else 0,
                        key='selected_filename'
                    )
                except ValueError as e:
                    st.error(f"Error selecting batch: {str(e)}")
                    st.error(f"Available filenames: {filenames}")
                    st.error(f"Default filename: {default_filename}")
                    selected_filename = None
                
                if selected_filename:
                    selected_row = stoppable_batches[stoppable_batches['filename'] == selected_filename].iloc[0]
                    if st.button("Stop batch", use_container_width=True):
                        try:    
                            self.batch_summary_logger.post_stop_request(selected_row['user_id'], selected_row['batch_id'])
                            st.success(f"Stop request sent for batch: {selected_filename}")
                        except ValueError as e:
                            st.error(e)
            else:
                st.info("No batches available to stop.")
        else:
            st.info("No batches to stop")
        
        st.write("### Pending Stop Requests:")
        pending_stop_requests = self.batch_summary_logger.check_stop_requests()
        
        if pending_stop_requests:  # This checks if the list is not empty
            try:
                # Convert the list to a DataFrame for better display
                pending_stop_requests_df = pd.DataFrame(pending_stop_requests)
                st.dataframe(pending_stop_requests_df, hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying pending stop requests: {str(e)}")
                st.write(pending_stop_requests)  # Fallback to plain list display
        else:
            st.info("No pending stop requests")