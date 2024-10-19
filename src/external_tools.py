import logging
logger = logging.getLogger(__name__)
from src.file_manager import SessionLogger, OpenaiThreadLogger
from src.setup import CredentialManager
import openai
import groq
import json
import time
import os
import requests
import re
import tiktoken
from time import sleep
from math import ceil
from bs4 import BeautifulSoup


#exceptions
class RetryableError(Exception):
    pass

class StopProcessingError(Exception):
    pass

class SkippableError(Exception):
    pass


class openai_advanced_uses:

    def __init__(self, session_logger: SessionLogger, credential_manager=None):
        self.session_logger = session_logger
        self.openai_thread_logger = OpenaiThreadLogger(session_logger.user_id, session_logger.tool_config)
        self.save_request_log = session_logger.save_request_log
        self.tool_configs = session_logger.tool_config
        self.credential_manager = credential_manager
        api_key = self.credential_manager.get_api_key('openai')
        if not api_key:
            raise StopProcessingError("Requires OpenAI api_key to work")
        
        self.openai_client = openai.OpenAI(api_key=api_key)
        pass
    
    #HANDLE FILES

    def openai_upload_file(self, file_path, purpose):
        return self.openai_client.files.create(
            file=open(file_path, "rb"),
            purpose= purpose
            )
    
    def openai_download_file(self, file_id):
        return self.openai_client.files.content(file_id).read()
    
            
    # HANDLE BATCHES
    """def create_batch_job(self, file_id,description):

        self.openai_client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
          "description": description
        }
      )
        
    def check_batch_job(self, batch_job):
        return self.openai_client.batches.retrieve(batch_job)"""
    
    #HANDLE ASSISTANTS
    #USED
    def create_assistant(self, assistant_configs):

        return self.openai_client.beta.assistants.create( 
                                    instructions=assistant_configs['assistant_sys_prompt'],
                                    name=assistant_configs['assistant_name'],
                                    tools=assistant_configs['tools'],
                                    model=assistant_configs['model'],
                                    )
    
    def list_assistants(self, order="desc", limit="20"):
        response = self.openai_client.beta.assistants.list(order=order,limit=limit,)
        
    
        listed_assistants = [
            (assistant.id, assistant.name)
            for assistant in response.data
        ]
    
        return listed_assistants

    def setup_thread(self,thread_name, file_id, user_setup_msg, assistant_setup_msg):
        thread = self.openai_client.beta.threads.create()
        if isinstance(file_id, list):
            attachments = file_id
        else:
            attachments = [{"file_id": file_id, "tools": [{"type": "code_interpreter"}]}]

        self.openai_client.beta.threads.messages.create(
            thread.id,
            role="user",
            content=user_setup_msg,
            attachments=attachments)

        self.openai_client.beta.threads.messages.create(
            thread.id,
            role="assistant",
            content=assistant_setup_msg,
                )
        
        self.openai_thread_logger.log_openai_thread( 
                    thread_id=thread.id, 
                    thread_name=thread_name, 
                    thread_file_ids= attachments, 
                    thread_history=[], 
                    user_setup_msg=user_setup_msg, 
                    assistant_setup_msg=assistant_setup_msg
                    )
        
        return thread
    
    def erase_thread(self,thread_id):
        self.openai_client.beta.threads.delete(thread_id)
        self.openai_thread_logger.erase_thread_data(thread_id)


    def reset_thread(self, thread_id):
        thread_data = self.openai_thread_logger.get_thread_info(thread_id)
        new_thread = self.setup_thread(thread_data['thread_name'], thread_data['file_ids'],thread_data['user_setup_msg'],thread_data['assistant_setup_msg'])
        self.openai_client.beta.threads.delete(thread_id)
        self.openai_thread_logger.erase_thread_data(thread_id)
        return new_thread.id

    #TEMPORARY USED
    def create_thread(self,thread_config=None):

        #this if not needs to be removed after testing
        if not thread_config:
            return self.openai_client.beta.threads.create()
        return self.openai_client.beta.threads.create( 
                                    instructions=thread_config['assistant_sys_prompt'],
                                    name=thread_config['assistant_name'],
                                    tools=thread_config['tools'],
                                    model=thread_config['model'],
                                    )
    #TEMPORARY USED
    def update_thread_files(self, thread_id, file_ids):
        self.openai_client.beta.threads.update(
                thread_id=thread_id,
                tool_resources={"code_interpreter": {"file_ids": file_ids}}
                )
    
    #USED
    def post_message(self, thread_id, message_content):
        return self.openai_client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=message_content
                )
    #USED
    def stream_response(self, thread_id, assistant_id, tool_choice={"type": "code_interpreter"}):
        return self.openai_client.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    tool_choice=tool_choice,
                    stream=True
                ) 
    



class LlmManager:
    """
    Manages interactions with Large Language Models (LLMs) using the OpenAI or Groq API.

    Attributes:
        configurations (dict): A dictionary of LLM configurations.
        model (str): The name of the selected LLM model.
        embedding_source (str): The source used for generating embeddings.
        embeddings_model (str): The specific model used for embeddings.
        client (object): The API client instance for the selected LLM model.
        max_token (int): Maximum token limit for the selected LLM model.
        save_request_log (callable): Method for saving request logs.
        llm_temp (float): The temperature parameter for the LLM.
    """

    def __init__(
        self,
        session_logger: SessionLogger,
        credential_manager: CredentialManager,
        config_key: str = None,
        llm_temp: float = 0.3,
    ):
        
        self.tool_configs = session_logger.tool_config
        self.llm_temp = llm_temp
        self.save_request_log = session_logger.save_request_log
        self.credential_manager = credential_manager

        self.configurations = {}
        self._init_openai_config()
        self._init_groq_config()
        
        self.setup_llm_settings(config_key)


    def setup_llm_settings(self, config_key):
        if not self.configurations:
            raise ValueError("No LLM model available. Please ensure API keys are set properly.")
        
        if config_key is None:
            config_key = self.tool_configs.get('default_llm')

        if config_key not in self.configurations:
            raise ValueError(f"Invalid configuration key: {config_key}")

        self.set_model_attributes(config_key)



    def set_model_attributes(self, config_key):
        config = self.configurations[config_key]
        self.model = config["model"]
        self.embedding_source = config["embedding_source"]
        self.embeddings_model = config["embeddings_model"]
        self.client = config["client"]
        self.max_token = config["max_token"]



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

    def llm_request(self, user_msg, sys_msg, json_mode=False, max_retries=3,retry_delay=1, time_limit=40, **kwargs): 

        """Call the llm and return the response. Takes as input the string for user_msg and sys_message.
        Supports json_mode if used for openai models, max_retries, retry_delay and time_limit to handle connection issue with openai"""
        user_msg = self._token_ceiling(user_msg)
        sys_msg = self._token_ceiling(sys_msg)
        if not user_msg or not sys_msg:
            return None
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
            return response.choices[0].message.content or None
        
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
    def __init__(self, session_logger, credential_manager):
        self.save_request_log = session_logger.save_request_log
        self.credential_manager = credential_manager
        

    def whisper_openai_transcribe(self,audio_file, client:openai.OpenAI):
        try:
            client=client(api_key = self.credential_manager.get_api_key('openai'))
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
            
            self.save_request_log(transcription, "whisper", "transcribe_audio")
            return transcription.text
 
        except Exception as e:
            raise RetryableError(f"Encountered an error: {e}")


class SerpApiManager:
    def __init__(self,session_logger, credential_manager):
        """Initialize the SerpAPIManager and set the url endpoints. API Key needs to be an env variable"""
        self.save_request_log = session_logger.save_request_log
        self.credential_manager = credential_manager
        self.base_url = "https://serpapi.com/search"
        pass
    
    def extract_organic_results(self, search_query, num_results, country, last_years=None):
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
        if last_years > 0:
            params['tbs'] = f"qdr:y{last_years}"

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
                'position': result.get('position', 'No position provided'),
                'link': result.get('link', 'No link provided'),
                'source': result.get('source', 'No source provided'),
                'title': result.get('title', 'No title provided'),
                'snippet': result.get('snippet', 'No snippet provided'),
                'date': result.get('date', 'No date provided')
            }
            results.append(extracted_data)
        
        return json.dumps(results, indent=4)


class WebScraper:

    def __init__(self,session_logger):
        self.save_request_log = session_logger.save_request_log
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
    def __init__(self,session_logger, credential_manager):
        """Initialize the Oxylab Manager and set the url endpoints. API Key needs to be an env variable"""
        self.save_request_log = session_logger.save_request_log
        self.credential_manager = credential_manager
        self.base_url = "https://realtime.oxylabs.io/v1/queries"
        self.paginator_max_pages = 10
        self.request_timeout = 90
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
    
    
    def _post_oxylab_request(self, payload):
        """
        Send a POST request to Oxylab's API with the given payload.

        Args:
            payload (dict): The request payload.

        Returns:
            requests.Response: The response object if successful.
            str: An error message if the request fails.
        """
        username, password = self._get_credentials()

        try:
            response = requests.post(
                'https://realtime.oxylabs.io/v1/queries',
                auth=(username, password),
                json=payload,
                timeout=self.request_timeout
            )
            response.raise_for_status()
            response_json = response.json()
            self.save_request_log(response_json, "OxyLabs", f"{payload.get('source')}")
            return response_json

        except requests.Timeout:
            raise RetryableError("Request timed out. Should Retry.")

        except requests.ConnectionError:
            raise RetryableError("Network connection error. Should Retry.")

        except requests.HTTPError as e:
            if e.response.status_code in [401, 403]:
                raise StopProcessingError("Authentication failed. Please check your API credentials.")
            elif e.response.status_code in [400, 422]:
                raise SkippableError(f"Bad request: {e.response.text}. Skipping this request.")
            else:
                raise StopProcessingError(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")

        except requests.RequestException as e:
            raise StopProcessingError(f"Failed to fetch content: {str(e)}")

        except Exception as e:
            raise StopProcessingError(f"An unexpected error occurred: {str(e)}")
        
    
    def _is_valid_asin(self, asin_str):
        """
        Checks if a string is a valid ASIN (Amazon Standard Identification Number).
        """
        # Regular expression pattern to match ASIN format
        pattern = r'^[A-Z0-9]{10}$'
        # Use re.match to check if the string matches the pattern
        return bool(re.match(pattern, asin_str))
    

    def get_amazon_product_info(self, asin, **kwargs):
        if not self._is_valid_asin(asin):
            raise SkippableError(f"Invalid ASIN, it should contain 10 characters either int or letters: {asin[:20]}") 
        
        payload = {
            'source': 'amazon_product',
            'parse': True,
            'start_page': kwargs.get('start_page', 1)
        }
        payload['query'] = asin
        payload['amazon_domain'] = kwargs.get('amazon_domain','it')

        data = self._post_oxylab_request(payload)

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

        return extracted_data
    
    def get_amazon_review(self,asin, **kwargs ): 
        """
        Crawl Amazon product infosusing Oxylabs API.
        Possible kwargs:
        - amazon_domain: which amazon website to crawl
        - start_page: starting page number (default 1)
        """
        if not self._is_valid_asin(asin):
            raise SkippableError(f"Invalid ASIN, it should contain 10 characters either int or letters: {asin[:20]}") 

        payload = {
            'source': 'amazon_reviews',
            'parse': True,
            'start_page': kwargs.get('start_page', 1),
            'query': asin,
        }

        payload['amazon_domain'] = kwargs.get('amazon_domain','it')
        payload['pages'] = kwargs.get('pages',1)

        data = self._post_oxylab_request(payload)
        
        results = []
        
        for result_set in data['results']:
            try:
                reviews = result_set['content']['reviews']
                for review in reviews:
                    processed_review = {
                        "rating": review["rating"],
                        "title": review["title"],
                        "content": review["content"],
                        "verified": review["is_verified"],
                        "date_of_review": review["timestamp"],
                        # "timestamp": date_parser(review["timestamp"]),
                    }
                    results.append(processed_review)
            except KeyError as e:
                print(f"KeyError: {e}. Skipping this result set.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}. Skipping this result set.")

    def web_crawler(self, url, **kwargs):

        payload = {
          'source': 'universal',
          'url': url
        }
        
        payload['render']= kwargs.get('render','html')
        payload['user_agent_type']= kwargs.get('user_agent_type','desktop_chrome')
        payload['locale']= kwargs.get('locale','it-it')

        content = self._post_oxylab_request(payload)

        soup = BeautifulSoup(content["results"][0]["content"], 'html.parser')

        text_elements = [tag.get_text(strip=True) for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
        finalized_content = "\n".join(text_elements)

        clean_content = ''.join(char for char in finalized_content if ord(char) >= 32)
        # Replace other problematic characters
        clean_content = re.sub(r'[\000-\010]|[\013-\014]|[\016-\031]', '', clean_content)
        # Replace em dash and en dash with regular dash
        clean_content = clean_content.replace('—', '-').replace('–', '-')

        return clean_content

    def serp_crawler(self, query, **kwargs):
        """
        Crawl Google SERP using Oxylabs API.
        - query: search query

        Possible kwargs:
        
        - domain: domain to search in
        - start_page: starting page number (default 1)
        - results_language: language of results
        - geo_location: geographical location for search
        """

        payload = {
            'source': 'google_search',
            'parse': True,
            'context': [{'key': 'filter', 'value': 1}],
            'query':query
        }

        for key in [ 'domain', 'geo_location', 'pages']:
            if key in kwargs:
                payload[key] = kwargs[key]

        if 'results_language' in kwargs:
            payload['context'].append({'key': 'results_language', 'value': kwargs['results_language']})
    
        last_years = kwargs.get('last_years', 0)
        if last_years > 0:
            payload['context'].append({'key': 'tbs', 'value': f"qdr:y{kwargs['last_years']}"})

        print (f"sending payload: \n{payload}")
               

        response = self._post_oxylab_request(payload)

        
        organic_results = response['results'][0]['content']['results']['organic']
        page_number = response['results'][0]['content']['page']
        total_results_number = response['results'][0]['content']['results']['search_information']['total_results_count']


        # Format the results

        formatted_results = []
        total_results_number = 0

        for result_set in response['results']:
            organic_results = result_set['content']['results']['organic']
            page_number = result_set['content']['page']
            total_results_number = result_set['content']['results']['search_information']['total_results_count']

            for result in organic_results:
                
                formatted_result = {
                    "page_position": result['pos'],
                    "page_number":page_number,
                    "link": result['url'],
                    "source": result.get('favicon_text', 'No source provided'),
                    "title": result['title'],
                    "snippet": result['desc'],
                    "total_results_number": total_results_number,
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
    

    def serp_paginator(self, query, num_results, fix_position=True, **kwargs):
        results = []
        start_page = kwargs.get('start_page', 1)  # Allowing start_page to be set via kwargs
        actual_num_results = num_results
        min_results_per_page = 6  # Initial minimum
        max_pages_per_request = kwargs.get('max_pages', self.paginator_max_pages)  # Renamed for clarity
        max_retries = 3

        # Initialize the status bar if status_bar is True
        """if status_bar:
            st = import_streamlit()
            status_text = st.empty()
            progress_bar = st.progress(0)"""

        while len(results) < actual_num_results:
            remaining_results = actual_num_results - len(results)
            pages_needed = ceil(remaining_results / min_results_per_page) if min_results_per_page > 0 else 1
            pages_to_fetch = min(pages_needed, max_pages_per_request)

            current_kwargs = kwargs.copy()
            current_kwargs['pages'] = pages_to_fetch
            current_kwargs['start_page'] = str(start_page)

            # Update status bar
            """if status_bar:
                status_text.text(f"Fetching results for '{query}' (Page {start_page} to {start_page + pages_to_fetch - 1})")
                progress = len(results) / actual_num_results
                progress_bar.progress(min(progress, 1.0))"""

            retry_count = 0
            while retry_count < max_retries:
                try:
                    page_results = self.serp_crawler(query, **current_kwargs)
                    print(f"Fetched pages {start_page} to {start_page + pages_to_fetch - 1}")
                    print(f"Received {len(page_results)} results")
                    break
                except RetryableError as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Retryable error occurred: {str(e)}. Retrying in 5 seconds... ({retry_count}/{max_retries})")
                        """if status_bar:
                            status_text.text(f"Retrying... ({retry_count}/{max_retries})")"""
                        sleep(5)
                    else:
                        print(f"Max retries reached for retryable error: {str(e)}")
                        page_results = []
                except SkippableError as e:
                    print(f"Skippable error occurred: {str(e)}. Skipping pages {start_page} to {start_page + pages_to_fetch - 1}.")
                    page_results = []
                    break
                except StopProcessingError:
                    """if status_bar:
                        status_text.text("Error: Stopped processing")
                        progress_bar.progress(1.0)"""
                    print("Processing was stopped due to a StopProcessingError.")
                    raise

            if not page_results:
                start_page += pages_to_fetch
                continue

            if not results:
                total_results_number = page_results[0].get('total_results_number', actual_num_results)
                if total_results_number < actual_num_results:
                    actual_num_results = total_results_number
                    print(f"Adjusted target to {actual_num_results} based on total_results_number")
                actual_results_per_page = len(page_results) / pages_to_fetch if pages_to_fetch > 0 else 0
                if actual_results_per_page > 0:
                    min_results_per_page = max(int(actual_results_per_page), 1)  # Ensure at least 1
                else:
                    min_results_per_page = 1  # Fallback to 1 to prevent division by zero

            results.extend(page_results)

            if len(results) >= actual_num_results:
                print(f"Terminating: Collected {len(results)} results (target: {actual_num_results})")
                break

            # Increment start_page for the next batch
            start_page += pages_to_fetch

        # Trim results to the exact number of results needed
        results = results[:actual_num_results]

        if fix_position:
            for new_position, result in enumerate(results, start=1):
                result['position'] = new_position

        # Update status bar to show completion
        """if status_bar:
            status_text.text(f"Finished fetching results for '{query}'")
            progress_bar.progress(1.0)"""

        return results