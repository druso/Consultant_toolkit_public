from src.file_manager import AppLogger
import openai
import groq
import streamlit as st
import json
import time
import os
import requests
import re
import tiktoken
#from transformers import AutoModel
from bs4 import BeautifulSoup


#exceptions

class RetryableError(Exception):
    pass

class StopProcessingError(Exception):
    pass

class SkippableError(Exception):
    pass


# assistant setup: 1- create assistant and get id, 2- upoload file and get file id 3- assign files to assistant
#assistant run 1- create thread, 2- create message, 3- run assistant
class openai_advanced_uses:

    def __init__(self, app_logger: AppLogger):
        self.save_request_log = app_logger.save_request_log
        self.tool_configs = st.session_state['tool_config']
        self.credential_manager = st.session_state['credential_manager']
        api_key = self.credential_manager.get_api_key('openai')
        if not api_key:
            raise StopProcessingError("Requires OpenAI api_key to work")
        
        self.openai_client = openai.OpenAI(api_key=api_key)
        pass

    def openai_upload_file(self, file_path, purpose):
        return self.openai_client.files.create(
            file=open(file_path, "rb"),
            purpose= purpose
            )
    
    def openai_download_file(self, file_id):
        return self.openai_client.files.content(file_id).read()
            
    
    def create_batch_job(self, file_id,description):

        self.openai_client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
          "description": description
        }
      )
        
    def check_batch_job(self, batch_job):
        return self.openai_client.batches.retrieve(batch_job)

    def create_assistant(self, assistant_configs):
        return self.openai_client.beta.assistants.create( # Should I get it or create it in the same function?
                                    instructions=assistant_configs['assistant_sys_prompt'],
                                    name=assistant_configs['assistant_name'],
                                    tools=assistant_configs['tools'],
                                    model=assistant_configs['model'],
                                    )

    def update_assistant_files(self, thread_id, file_ids):
        self.openai_client.beta.threads.update(
                thread_id=thread_id,
                tool_resources={"code_interpreter": {"file_ids": file_ids}}
                )

    def list_assistants(self, order="desc", limit="20"):
        return self.openai_client.beta.assistants.list(
        order=order,
        limit=limit,
    )

    def get_assistant(self, assistant_id):
        return self.openai_client.beta.assistants.retrieve(assistant_id)

    def create_thread(self):
        return self.openai_client.beta.threads.create()
    
    def create_message(self, thread_id, assistant_id, message_content, tool_choice={"type": "code_interpreter"}):
        return self.openai_client.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    tool_choice=tool_choice,
                    stream=True
                ) #
    
    def run_assistant(self, assistant_id, thread_id, instructions):
        return self.openai_client.beta.assistants.run(assistant_id, thread_id, instructions)




class LlmManager:
    """
    Manages interactions with Large Language Models (LLMs) using the OpenAI or Groq API.

    This class encapsulates the logic for selecting an LLM model (e.g., GPT-3, Llama, Mixtral), 
    configuring its parameters (temperature), and handling interactions with the chosen API.
    It also provides functionality for generating embeddings from the LLM.

    Attributes:
        configurations (dict): A dictionary of LLM configurations, each containing:
            - model (str): The name of the model.
            - client (object): The API client instance for the model (OpenAI or Groq).
            - embedding_source (str): The source for generating embeddings ("openai" or "huggingface").
            - embeddings_model (str): The specific model used for embeddings.
            - max_token (int): Maximum token limit for the model.
        model (str): The name of the currently selected LLM model.
        embedding_source (str): The source used for generating embeddings.
        embeddings_model (str): The specific model used for embeddings.
        client (object): The API client instance for the selected LLM model.
        max_token (int): Maximum token limit for the selected LLM model.
        save_request_log (callable): A method (from AppLogger) for saving request logs.
        llm_temp (float): The temperature parameter for the LLM (only if 'streamlit' mode).
    """

    def __init__(self, config_key: str, app_logger: AppLogger):
        """
        Initializes the LlmManager.

        Args:
            config_key (str): 
                - If 'streamlit', presents a Streamlit UI for model and temperature selection.
                - Otherwise, directly selects the model based on the `config_key`.
            app_logger (AppLogger): An instance of the `AppLogger` class for logging.

        Raises:
            ValueError: If an invalid configuration key is provided.
        """
        self.save_request_log = app_logger.save_request_log
        self.tool_configs = st.session_state['tool_config']
        self.credential_manager = st.session_state['credential_manager']


        # Initialize configurations for each provider
        self.configurations = {}
        self._init_openai_config()
        self._init_groq_config()

        if self.configurations == {}:
            st.error("No LLM model available. Please make sure api_keys are set properly")
            st.stop()
        

        if config_key == "streamlit":
            config_key = st.sidebar.selectbox("llm model",list(self.configurations.keys()))
            self.llm_temp = st.sidebar.slider("Temperature", min_value=0.0,max_value=2.0, value=0.3 ,step=0.1)
        
        if config_key not in self.configurations:
            raise ValueError("Invalid configuration key.")

        self.model = self.configurations[config_key]['model']
        self.embedding_source = self.configurations[config_key]['embedding_source']
        self.embeddings_model = self.configurations[config_key]['embeddings_model']
        self.client = self.configurations[config_key]['client']
        self.max_token = self.configurations[config_key]['max_token']


    def _init_openai_config(self):
        if self.tool_configs['openai']['use']:
            api_key = self.credential_manager.get_api_key('openai')
            if api_key:
                openai_client = openai.OpenAI(api_key=api_key)
                for model in self.tool_configs['openai']['available_models']:
                    self.configurations[model] = {
                        'model': model,
                        'client': openai_client,
                        'embedding_source': 'openai',
                        'embeddings_model': 'text-embedding-3-small',
                        'max_token': self.tool_configs['openai'].get('max_token', 3000),
                    }
            else:
                print("OpenAI API key not found")

    def _init_groq_config(self):
        if self.tool_configs['groq']['use']:
            api_key = self.credential_manager.get_api_key('groq')
            if api_key:
                groq_client = groq.Groq(api_key=api_key)
                for model in self.tool_configs['groq']['available_models']:
                    self.configurations[model] = {
                        'model': model,
                        'client': groq_client,
                        'embedding_source': 'huggingface',
                        'embeddings_model': 'jinaai/jina-embeddings-v2-base-en',
                        'max_token': self.tool_configs['groq'].get('max_token', 3000),
                    }
            else:
                print("Groq API key not found")

    def _token_ceiling(self, text: str, llm_model: str = 'gpt-3.5-turbo') -> str:
        """
        Enforces a token limit on input text for LLM interactions.

        Uses tiktoken to encode the text and truncate it to the specified maximum token limit 
        for the given LLM model.  If the input is not a string, an empty string is returned.

        Args:
            text (str): The input text to be tokenized and potentially truncated.
            llm_model (str, optional): The LLM model name (defaults to 'gpt-3.5-turbo'). 
                                       This is used to determine the appropriate tokenization scheme.

        Returns:
            str: The original text if it's within the token limit, or the truncated text 
                 up to the maximum token limit. If input is not a string, returns empty string.
        """
        if not isinstance(text, str):
            return ""
            #raise TypeError(f"Expected a string, but got {type(text)}: {text}")
        max_token=self.max_token
        enc = tiktoken.encoding_for_model(llm_model)
        encoding = tiktoken.get_encoding(enc.name)
        tokenslist = encoding.encode(text)
        if len(tokenslist)>max_token:
            print("token ceiling invoked")
            return encoding.decode(tokenslist[:max_token])
        else:
            return text

    def llm_request(self, user_msg, sys_msg, json_mode=False, max_retries=3,retry_delay=1, time_limit=40): 

        """Call the llm and return the response. Takes as input the string for user_msg and sys_message.
        Supports json_mode if used for openai models, max_retries, retry_delay and time_limit to handle connection issue with openai"""
        user_msg = self._token_ceiling(user_msg)
        if not user_msg:
            return ""
        final_prompt = [{"role": "user", "content": user_msg}]
        final_prompt.insert (0,{"role": "system", "content": sys_msg})
        
        try:
            call_params = {
                "model": self.model,
                "messages": final_prompt,
                "temperature": self.llm_temp,
            }
            if json_mode:
                call_params["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**call_params)
            self.save_request_log(response.model_dump_json(), "LLM", "llm_request")
            return response.choices[0].message.content
        
        except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
            raise StopProcessingError(f"Encountered an error: {e}")
        except Exception as e:
            raise RetryableError(f"Encountered an error: {e}")


    def llm_chat_request(self, chat_messages, sys_msg, history_lenght=8, stream=True): 
        """Call the llm and return the response. Takes as input a list of chat messages and the string for sys_message.
        Optionally history_length can be set (default=8) and stream can be set to false"""
        # Cut conversation history to history_length (domanda + risposta = 2)
        final_chat_messages = chat_messages[-history_lenght:]
        # Add system message to the top
        final_chat_messages.insert (0,{"role": "system", "content": sys_msg})
        try:
            call_params = {
                        "model": self.model,
                        "messages": final_chat_messages,
                        "temperature": self.llm_temp,
                        "stream":stream
                    }
            response = self.client.chat.completions.create(**call_params)
            if stream:
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
            else:
                self.save_request_log(response, "LLM", "llm_chat_request")
                return response.choices[0].message.content
            
        except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
            raise StopProcessingError(f"Encountered an error: {e}")
        except Exception as e:
            raise RetryableError(f"Encountered an error: {e}")
    
    def embed_query(self, query, force_openai=True):
        
        query = self._token_ceiling(query)
        if not query:
            return ""
        
        if force_openai:
            self.embedding_source = 'openai'
            self.embeddings_model = 'text-embedding-3-small'

        """if self.embedding_source == 'huggingface':
            embedding_model = AutoModel.from_pretrained(self.embeddings_model, trust_remote_code=True)
            try:
                query_embeddings = embedding_model.encode(query).tolist()
                return query_embeddings

            except Exception as e:
                raise RetryableError(f"Encountered an error: {e}")
        else:"""
        try:
            client = self.openai_client
            embeddings_response = client.embeddings.create(input = [query], model=self.embeddings_model)
            query_embeddings = embeddings_response.data[0].embedding
            
            self.save_request_log(embeddings_response, "LLM", "embed_query")
            return query_embeddings
        except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
            raise StopProcessingError(f"Encountered an error: {e}")
        except Exception as e:
            raise RetryableError(f"Encountered an error: {e}")



class AudioTranscribe:
    def __init__(self, app_logger):
        self.save_request_log = app_logger.save_request_log
        

    def whisper_openai_transcribe(self,audio_file, client:openai.OpenAI):
        try:
            client=client(api_key = st.session_state.get('OPENAI_API_KEY'))
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
            
            self.save_request_log(transcription, "whisper", "transcribe_audio")
            return transcription.text
 
        except Exception as e:
            raise RetryableError(f"Encountered an error: {e}")


class SerpApiManager:
    def __init__(self,app_logger):
        """Initialize the SerpAPIManager and set the url endpoints. API Key needs to be an env variable"""
        self.save_request_log = app_logger.save_request_log
        self.credential_manager = st.session_state['credential_manager']
        self.base_url = "https://serpapi.com/search"
        pass
    
    def extract_organic_results(self, search_query, num_results, country):
        """Given a search_query and a num_result to provide it will return the organic first positions of the organic search
        The response will be a json with Position, Title, Link and Source for each result scraped.
        """
        api_key = self.credential_manager.get_api_key('serp_api')
        if not api_key:
            raise StopProcessingError("SerpAPI key not found. Please set it in the **🔧 Settings & Recovery** page.")

        params = {
            'engine': 'google',  # Specifies which search engine to use, e.g., google
            'q': search_query,  # Query parameter, what you are searching for
            'api_key': api_key,  # SerpApi API key
            'gl': country,
        }
        try:
            response = requests.get(self.base_url, params=params)
            if response.status_code == 401:
                raise StopProcessingError(f"Encountered a 401 Unauthorized Error: {response.text}") 
            
            response.raise_for_status()  # Raises an HTTPError for bad responses
            response_json = response.json()
            self.save_request_log(response_json, "serpAPI", f"search_query_{search_query}")
        
        except StopProcessingError as e:
            raise
        except Exception as e:
            raise RetryableError(f"Encountered an error: {e}")

        # Ensure the number of requested results does not exceed the available results
        num_results = min(num_results, len(response_json['organic_results']))
        
        # Initialize a list to store the results
        results = []
        
        # Loop through the first num_results entries in 'organic_results'
        for i in range(num_results):
            result = response_json['organic_results'][i]
            # Extract the desired information
            extracted_data = {
                'position': result.get('position'),
                'link': result.get('link'),
                'source': result.get('source', 'No source provided'),  # Provide default if source is missing
                'title': result.get('title')
            }
            results.append(extracted_data)
        
        return json.dumps(results, indent=4)


class WebScraper:
    #need to add
    #Generic - using oxylab for improved rendering
    def __init__(self,app_logger):
        self.save_request_log = app_logger.save_request_log
    def url_simple_extract(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status() 

            soup = BeautifulSoup(response.text, 'html.parser')

            # Customize text extraction here (e.g., focus on specific elements)
            text_elements = [tag.get_text(strip=True) for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]

            # Construct the combined content (if you need it)
            content = "\n\n".join(text_elements)

            # Save to your log
            clean_url = re.sub(r'[^\w\-_]', '_', url)[:20]
            self.save_request_log({"content": text_elements}, "Crawler", f"base_crawl_{clean_url}")

            return content

        except requests.exceptions.RequestException as e:
            print(f"Error extracting content from '{url}': {e}")
            return "Crawling failed"
    
class OxyLabsManager():
    def __init__(self,app_logger):
        self.save_request_log = app_logger.save_request_log
        """Initialize the Oxylab Manager and set the url endpoints. API Key needs to be an env variable"""
        self.credential_manager = st.session_state['credential_manager']
        self.base_url = "https://realtime.oxylabs.io/v1/queries"
        pass

    def _get_credentials(self):

        oxylab_keys = self.credential_manager.get_api_key('oxylab')
        if not isinstance(oxylab_keys, dict):
            raise StopProcessingError("OxyLabs credentials are not in the expected format.")
        
        username = oxylab_keys.get('OXYLABS_USER')
        password = oxylab_keys.get('OXYLABS_PSW')

        if not username or not password:
            raise StopProcessingError("OxyLabs username or password is missing. Please set it in the **🔧 Settings & Recovery** page.")
        
        return username, password
    
    

    def _is_valid_asin(self, asin_str):
        """
        Checks if a string is a valid ASIN (Amazon Standard Identification Number).

        Args:
            asin_str: The string to check.

        Returns:
            True if the string is a valid ASIN, False otherwise.
        """

        # Regular expression pattern to match ASIN format
        pattern = r'^[A-Z0-9]{10}$'

        # Use re.match to check if the string matches the pattern
        return bool(re.match(pattern, asin_str))
    

    def extract_amazon_product_info(self,asin,amazon_domain):
        if not self._is_valid_asin(asin):
            raise RetryableError(f"Invalid ASIN, it should contain 10 characters either int or letters: {asin[:20]}") 

        username, password= self._get_credentials()

        payload = {
            "source": "amazon_product",  
            "query": asin, 
            "parse":True,
            "domain":amazon_domain,
        }

        headers = {
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(self.base_url, json=payload, auth=(username, password), headers=headers)
            if response.status_code == 401:
                raise StopProcessingError(f"Encountered a 401 Unauthorized Error: {response.text}") 
            
            response.raise_for_status()  # Raises an HTTPError for bad responses
            response_json = response.json()
            self.save_request_log(response_json, "OxyLabs", f"amazon_product_{amazon_domain}_{asin}")
        
        except StopProcessingError as e:
            raise
        except Exception as e:
            raise RetryableError(f"Encountered an error: {e}")

        data = response.json()
        extracted_data = {
            "num_variation": len(data['results'][0]['content'].get('variation',[])),
            "num_ads_on_page": len(data['results'][0]['content'].get('ads',[])),
            "asin_on_ads":[ad.get("asin") for ad in data['results'][0]['content'].get("ads", []) if ad.get("asin")],
            "sales_rank": data['results'][0]['content'].get('sales_rank'),
            "product_name": data['results'][0]['content'].get('product_name'),
            "manufacturer": data['results'][0]['content'].get('product_name'),
            "description": data['results'][0]['content'].get('description'),
            "bullet_points": data['results'][0]['content'].get('bullet_points'),
            "price": data['results'][0]['content'].get('price_upper'),
            "price_shipping":data['results'][0]['content'].get('price_shipping'),
            "is_prime_eligible": data['results'][0]['content'].get('is_prime_eligible'),
            "answered_questions_count": data['results'][0]['content'].get('answered_questions_count'),
            "reviews_count": data['results'][0]['content'].get('reviews_count'),
            }
        print(extracted_data)
        
        return json.dumps(extracted_data, indent=4)
    

    def extract_amazon_reviews(self,asin,amazon_domain, review_pages):
        if not self._is_valid_asin(asin):
            raise SkippableError(f"Invalid ASIN, it should contain 10 characters either int or letters: {asin[:20]}") 

        username, password= self._get_credentials()

        payload = {
            "source": "amazon_reviews",  
            "query": asin, 
            "parse":True,
            "domain":amazon_domain,
            "start_page":1,
            "pages": review_pages, 
        }

        headers = {
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(self.base_url, json=payload, auth=(username, password), headers=headers)
            if response.status_code == 401:
                raise StopProcessingError(f"Encountered a 401 Unauthorized Error: {response.text}") 
            
            response.raise_for_status()  # Raises an HTTPError for bad responses
            response_json = response.json()
            self.save_request_log(response_json, "OxyLabs", f"amazon_product_{amazon_domain}_{asin}")
        
        except StopProcessingError as e:
            raise
        except Exception as e:
            raise RetryableError(f"Encountered an error: {e}")

        data = response.json()
        
        results = []
        try:
            for i in data['results'][0]['content']['reviews']:
                review = {"rating":i["rating"], 
                        "title":i["title"],
                        "content":i["content"],
                        "verified":i["is_verified"],
                        "date_of_review":i["timestamp"],
                        #"timestamp" : date_parser(i["timestamp"]),
                        }
                results.append(review)
        except:
            pass
        print( json.dumps(results, indent=4))
        return json.dumps(results, indent=4)


    ### Should extend oxylab and serpapi interfaces