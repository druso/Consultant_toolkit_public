import logging
logger = logging.getLogger(__name__)
from src.file_manager import SessionLogger, OpenaiThreadLogger
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
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


def _exponential_backoff(func, max_retries=3, base_delay=1, exc_types=(Exception,)):
    """Execute *func* with retries using exponential backoff.

    Parameters
    ----------
    func : callable
        Function to execute.
    max_retries : int, optional
        Maximum number of attempts. Default 3.
    base_delay : int or float, optional
        Initial backoff delay in seconds. Default 1.
    exc_types : tuple, optional
        Exceptions that trigger a retry.
    """
    delay = base_delay
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except exc_types as e:
            if attempt == max_retries:
                raise
            logger.warning(
                f"Attempt {attempt} failed with {e}. Retrying in {delay}s"
            )
            time.sleep(delay)
            delay *= 2


def _handle_http_errors(response):
    """Raise appropriate exceptions based on HTTP status."""
    if response.status_code in (401, 403):
        raise StopProcessingError(f"Authentication failed: {response.text}")
    if response.status_code == 404:
        raise SkippableError(f"Resource not found: {response.text}")
    if response.status_code == 429 or response.status_code >= 500:
        raise RetryableError(f"Temporary server issue ({response.status_code}): {response.text}")
    if response.status_code >= 400:
        raise SkippableError(f"Bad request ({response.status_code}): {response.text}")


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
                logger.warning("OpenAI API key not found")

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
                logger.error("Groq API key not found")

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
            logger.info("token ceiling invoked")
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
        
        call_params = {
            "model": self.model,
            "messages": final_prompt,
            "temperature": self.llm_temp,
        }
        if json_mode:
            call_params["response_format"] = {"type": "json_object"}

        def _call():
            try:
                response = self.client.chat.completions.create(**call_params)
                self.save_request_log(response.model_dump_json(), "LLM", "llm_request")
                return response.choices[0].message.content or None
            except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
                raise StopProcessingError(f"Encountered an error: {e}")
            except Exception as e:
                raise RetryableError(f"Encountered an error: {e}")

        return _exponential_backoff(_call, max_retries=max_retries, base_delay=retry_delay, exc_types=(RetryableError,))


    def llm_chat_request(self, chat_messages, sys_msg, history_lenght=8, stream=True, max_retries=3, retry_delay=1):
        """Call the llm and return the response. Takes as input a list of chat messages and the string for sys_message.
        Optionally history_length can be set (default=8) and stream can be set to false"""
        # Cut conversation history to history_length (domanda + risposta = 2)
        final_chat_messages = chat_messages[-history_lenght:]
        # Add system message to the top
        final_chat_messages.insert (0,{"role": "system", "content": sys_msg})
        call_params = {
                        "model": self.model,
                        "messages": final_chat_messages,
                        "temperature": self.llm_temp,
                        "stream":stream
                    }

        def _call():
            try:
                response = self.client.chat.completions.create(**call_params)
                return response
            except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
                raise StopProcessingError(f"Encountered an error: {e}")
            except Exception as e:
                raise RetryableError(f"Encountered an error: {e}")

        response = _exponential_backoff(_call, max_retries=max_retries, base_delay=retry_delay, exc_types=(RetryableError,))
        if stream:
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        else:
            self.save_request_log(response, "LLM", "llm_chat_request")
            return response.choices[0].message.content
    
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
        client = self.openai_client

        def _call():
            try:
                embeddings_response = client.embeddings.create(input=[query], model=self.embeddings_model)
                query_embeddings = embeddings_response.data[0].embedding
                self.save_request_log(embeddings_response, "LLM", "embed_query")
                return query_embeddings
            except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
                raise StopProcessingError(f"Encountered an error: {e}")
            except Exception as e:
                raise RetryableError(f"Encountered an error: {e}")

        return _exponential_backoff(_call, max_retries=3, exc_types=(RetryableError,))


class AudioTranscribe:
    def __init__(self, session_logger, credential_manager):
        self.save_request_log = session_logger.save_request_log
        self.credential_manager = credential_manager

    def _transcribe_audio(self, audio_file, provider, client_class, model):
        """Internal method to handle audio transcription for different providers."""
        api_key = self.credential_manager.get_api_key(provider)
        if not api_key:
            raise StopProcessingError(f"{provider} API key not found.")
        
        def _call():
            try:
                logger.info(f"Requesting transcription of audio with {provider}")
                client = client_class(api_key=api_key)
                transcription = client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                )
                self.save_request_log(transcription, "whisper", "transcribe_audio")
                return transcription.text
            except Exception as e:
                raise RetryableError(f"Encountered an error: {e}")

        return _exponential_backoff(_call, max_retries=3, exc_types=(RetryableError,))

    def whisper_groq_transcribe(self, audio_file, client=groq.Groq):
        return self._transcribe_audio(
            audio_file=audio_file,
            provider='groq',
            client_class=client,
            model="whisper-large-v3-turbo"
        )

    def whisper_openai_transcribe(self, audio_file, client=openai.OpenAI):
        return self._transcribe_audio(
            audio_file=audio_file,
            provider='openai',
            client_class=client,
            model="whisper-1"
        )


class SerpApiManager:
    def __init__(self,session_logger, credential_manager):
        """Initialize the SerpAPIManager and set the url endpoints. API Key needs to be an env variable"""
        self.save_request_log = session_logger.save_request_log
        self.credential_manager = credential_manager
        self.base_url = "https://serpapi.com/search"
        pass

            
    def serpapi_serp_crawler(self, query, **kwargs): #the new method to use with universal paginator
        """
        Crawl Google SERP using SerpAPI.
        Args:
            query: search query
            
        Possible kwargs:
            - country: geographical location for search (gl parameter)
            - last_years: filter results by number of years
            - start_page: starting page number (default 1)
            - pages: number of pages to fetch (not used in SerpAPI but kept for compatibility)
        """
        api_key = self.credential_manager.get_api_key('serp_api')
        if not api_key:
            raise StopProcessingError("SerpAPI key not found. Please set it in the **🔧 Settings & Recovery** page.")

        params = {
            'engine': 'google',
            'q': query,
            'api_key': api_key,
            'gl': kwargs.get('country', 'us'),  # Default to US if not specified
            'start': kwargs.get('processed_results', 1),
            'nfpr': 1, #exclude auto corrected results
        }

        # Add time filter if specified
        last_years = kwargs.get('last_years', 0)
        if last_years > 0:
            params['tbs'] = f"qdr:y{last_years}"

        def _call():
            try:
                response = requests.get(self.base_url, params=params)
                _handle_http_errors(response)
                return response.json()
            except requests.RequestException as e:
                raise RetryableError(f"Network error: {e}")

        try:
            response_json = _exponential_backoff(
                _call,
                max_retries=3,
                exc_types=(RetryableError,)
            )
        except (StopProcessingError, SkippableError, RetryableError):
            raise


        formatted_results = []
        total_results_number = response_json.get('search_information', {}).get('total_results', 0)
        
        for result in response_json.get('organic_results', []):
            formatted_result = {
                "result_type": "organic",
                "position": result.get('position'),
                "link": result.get('link'),
                "source": result.get('source', 'No source provided'),
                "title": result.get('title'),
                "snippet": result.get('snippet'),
                "date": result.get('date', 'No date provided'),
                "total_results_number": total_results_number,
            }
            formatted_results.append(formatted_result)
        
        return formatted_results


            
    def serpapi_review_crawler(self, product_id, **kwargs): #the new method to use with universal paginator

        api_key = self.credential_manager.get_api_key('serp_api')
        if not api_key:
            raise StopProcessingError("SerpAPI key not found. Please set it in the **🔧 Settings & Recovery** page.")

        params = {
            'engine': 'google_product',
            'product_id': product_id,
            'reviews': "1",
            'api_key': api_key,
            'gl': kwargs.get('country', 'US'),  # Default to US if not specified
        }

        if kwargs.get('filter'):
            params['filter'] = kwargs.get('filter')

        def _call():
            try:
                response = requests.get(self.base_url, params=params)
                _handle_http_errors(response)
                return response.json()
            except requests.RequestException as e:
                raise RetryableError(f"Network error: {e}")

        try:
            response_json = _exponential_backoff(
                _call,
                max_retries=3,
                exc_types=(RetryableError,)
            )
        except (StopProcessingError, SkippableError, RetryableError):
            raise


        review_list = []
        total_results_number = response_json.get('product_results', {}).get('reviews', 0)
        next_page_filter = response_json.get('serpapi_pagination', {}).get('next_page_filter', 0)
        for result in response_json.get('reviews_results', {}).get('reviews', []):
            review = {
                "result_type": "product_review",
                "position": result.get('position'),
                "title": result.get('title'),
                "date": result.get('date'),
                "rating": result.get('rating'),
                "source": result.get('source'),
                "content": result.get('content'),
                "total_results_number": total_results_number,
            }
            review_list.append(review)
        
        return review_list, next_page_filter

    def serpapi_maps_fetch_places(self, query, latitude, longitude, **kwargs):
        """
        Fetches a list of places based on latitude, longitude, and keyword.

        Parameters:
            latitude (float): Latitude of the location.
            longitude (float): Longitude of the location.
            keyword (str): Search keyword.
            api_key (str): API key for SerpApi.

        Returns:
            dict: Dictionary containing the local results (places) or None if no results.
        """        
        api_key = self.credential_manager.get_api_key('serp_api')
        if not api_key:
            raise StopProcessingError("SerpAPI key not found. Please set it in the **🔧 Settings & Recovery** page.")


        maps_params = {
            "engine": "google_maps",
            "q": query,
            "ll": f"@{latitude},{longitude},15z",
            "type": "search",
            "api_key": api_key,
        }

        def _call():
            try:
                maps_response = requests.get(self.base_url, params=maps_params)
                _handle_http_errors(maps_response)
                return maps_response.json()
            except requests.RequestException as e:
                raise RetryableError(f"Network error: {e}")

        try:
            response_json = _exponential_backoff(
                _call,
                max_retries=3,
                exc_types=(RetryableError,)
            )
        except (StopProcessingError, SkippableError, RetryableError):
            raise

        locals_list = []
        for result in response_json.get('local_results', {}):
            local = {
                "result_type": "local_result",
                "position": result.get('position'),
                "title": result.get('title'),
                "place_id": result.get('place_id'),
                "gps_coordinates": result.get('gps_coordinates'),
                "rating": result.get('rating'),
                "reviews": result.get('reviews'),
                "types": result.get('types'),
                "address": result.get('address'),
                "thumbnail": result.get('thumbnail'),
            }
            locals_list.append(local)

        return locals_list


    def serpapi_maps_reviews_crawler(self, place_id, **kwargs):
        """
        Fetches reviews for a specific place based on its place_id.

        Parameters:
            place_id (str): The place_id of the location.
            api_key (str): API key for SerpApi.

        Returns:
            dict: Dictionary containing the reviews for the place or None if no reviews found.
        """

        api_key = self.credential_manager.get_api_key('serp_api')
        if not api_key:
            raise StopProcessingError("SerpAPI key not found. Please set it in the **🔧 Settings & Recovery** page.")

        reviews_params = {
            "engine": "google_maps_reviews",
            "place_id": place_id,
            "api_key": api_key,
        }        
        
        if kwargs.get('filter'):
            #print("passing filter")
            reviews_params['next_page_token'] = kwargs.get('filter') #I use next_pagination_filter here

        def _call():
            try:
                reviews_response = requests.get(self.base_url, params=reviews_params)
                _handle_http_errors(reviews_response)
                return reviews_response.json()
            except requests.RequestException as e:
                raise RetryableError(f"Network error: {e}")

        try:
            response_json = _exponential_backoff(
                _call,
                max_retries=3,
                exc_types=(RetryableError,)
            )
        except (StopProcessingError, SkippableError, RetryableError):
            raise

        review_list = []
        total_results_number = response_json.get('place_info', {}).get('reviews', 0)
        next_page_filter = response_json.get('serpapi_pagination', {}).get('next_page_token', 0)
        for result in response_json.get('reviews', []):
            review = {
                "result_type": "place_review",
                "rating": result.get('rating'),
                "extracted_snippet": result.get('extracted_snippet'),
                "contributor": result.get('user').get('name'),
                "contributor_id": result.get('user').get('contributor_id'),
                "source": result.get('source'),
                "iso_date": result.get('iso_date'),
                "link": result.get('link'),
                "total_results_number": total_results_number,
            }
            review_list.append(review)

        return review_list, next_page_filter

class WebScraper:

    def __init__(self,session_logger):
        self.save_request_log = session_logger.save_request_log
    def url_simple_extract(self, url):
        def _call():
            try:
                response = requests.get(url)
                _handle_http_errors(response)
                return response.text
            except requests.RequestException as e:
                raise RetryableError(f"Network error: {e}")

        try:
            html = _exponential_backoff(
                _call,
                max_retries=3,
                exc_types=(RetryableError,)
            )
        except (StopProcessingError, SkippableError, RetryableError) as e:
            logger.error(f"Error extracting content from '{url}': {e}")
            return "Crawling failed"

        soup = BeautifulSoup(html, 'html.parser')

        text_elements = [tag.get_text(strip=True) for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
        content = "\n\n".join(text_elements)
        clean_url = re.sub(r'[^\w\-_]', '_', url)[:20]
        self.save_request_log({"content": text_elements}, "Crawler", f"base_crawl_{clean_url}")

        return content
    
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

        def _call():
            try:
                response = requests.post(
                    'https://realtime.oxylabs.io/v1/queries',
                    auth=(username, password),
                    json=payload,
                    timeout=self.request_timeout
                )
                _handle_http_errors(response)
                response_json = response.json()
                self.save_request_log(response_json, "OxyLabs", f"{payload.get('source')}")
                return response_json
            except requests.Timeout:
                raise RetryableError("Request timed out")
            except requests.ConnectionError:
                raise RetryableError("Network connection error")
            except requests.RequestException as e:
                raise RetryableError(str(e))

        try:
            return _exponential_backoff(
                _call,
                max_retries=3,
                exc_types=(RetryableError,)
            )
        except (StopProcessingError, SkippableError, RetryableError):
            raise
        
    
    def _is_valid_asin(self, asin_str):
        """
        Checks if a string is a valid ASIN (Amazon Standard Identification Number).
        """
        # Regular expression pattern to match ASIN format
        pattern = r'^[A-Z0-9]{10}$'
        # Use re.match to check if the string matches the pattern
        return bool(re.match(pattern, asin_str))
    


    def get_amazon_asins(self, search_string, **kwargs):

        payload = {
            'source': 'amazon_search',
            'parse': True,
            'start_page': kwargs.get('start_page', 1)
        }
        payload['query'] = search_string
        payload['domain'] = kwargs.get('domain','it')

        data = self._post_oxylab_request(payload)
        organic_list = data['results'][0]['content']['results'].get('organic',[])
        #print (organic_list)
        extracted_data = [
            {
                'source': 'amazon', 
                'position': item.get('pos',''),
                'title': item.get('title',""),
                'brand': item.get('manufacturer',''),
                'product_id': item.get('asin',""),
                'price': item.get('price',''),
                'rating': item.get('rating',''),
                'currency': kwargs.get('domain','it'),
                'reviews_count': item.get('reviews_count',''),
                'thumbnail': item['url_image'],
            } 
            for item in organic_list
        ]

        return extracted_data
        

    def get_amazon_product_info(self, asin, **kwargs):
        if not self._is_valid_asin(asin):
            raise SkippableError(f"Invalid ASIN, it should contain 10 characters either int or letters: {asin[:20]}") 
        
        payload = {
            'source': 'amazon_product',
            'parse': True,
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
                logger.error(f"KeyError: {e}. Skipping this result set.")
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}. Skipping this result set.")

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

    def oxylabs_serp_crawler(self, query, **kwargs):
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

        for key in [ 'domain', 'geo_location', 'pages', 'start_page']:
            if key in kwargs:
                payload[key] = kwargs[key]

        if 'results_language' in kwargs:
            payload['context'].append({'key': 'results_language', 'value': kwargs['results_language']})

        last_years = kwargs.get('last_years', 0)
        if last_years > 0:
            payload['context'].append({'key': 'tbs', 'value': f"qdr:y{kwargs['last_years']}"})

        logger.info(f"sending payload: \n{payload}")

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
                    "result_type":"organic",
                    "position": result['pos_overall'],
                    "link": result['url'],
                    "source": result.get('favicon_text', 'No source provided'),
                    "title": result['title'],
                    "snippet": result['desc'],
                    "date": result.get('date', 'No date provided'),
                    "total_results_number": total_results_number,
                }
                formatted_results.append(formatted_result)

        return formatted_results
    
    def get_google_productids(self, search_string, **kwargs):

        payload = {
            'source': 'google_shopping_search',
            'parse': True,
            'start_page': kwargs.get('start_page', 1)
        }
        payload['query'] = search_string
        payload['domain'] = kwargs.get('domain','it')

        data = self._post_oxylab_request(payload)

        organic_list = data['results'][0]['content']['results'].get('organic', [])
        #print(f"organic_list: {organic_list}")
        extracted_data = [
        {
            'source': 'google',
            'position': item.get('pos_overall',''),
            'title': item.get('title',""),
            'product_id': item.get('product_id',""),
            'price': item.get('price',''),
            'rating': item.get('rating',''),
            'currency': item.get('currency',''),
            'reviews_count': item.get('reviews_count',''),
            'thumbnail': item.get('thumbnail',"")
        }
        for item in organic_list
        if item.get('product_id')  
    ]

        return extracted_data    


    def get_google_product_info(self, product_id, **kwargs):

        payload = {
            'source': 'google_shopping_product',
            'parse': True,
        }
        payload['query'] = product_id
        payload['domain'] = kwargs.get('domain','it')
        if kwargs.get('start_page',None):
            payload['start_page'] = kwargs.get('start_page', 1)

        data = self._post_oxylab_request(payload)

        product = data['results'][0]['content']
        title = product.get('title',"")
        description = product.get('description',"")
        # Convert specifications to a semicolon-separated string
        specifications = product.get('specifications',[{}])[0].get('items',[])
        specifications_text = ";".join([f"{item['title']}:{item['value']}" for item in specifications])

        # Store the extracted data in a dictionary
        extracted_data = {
            'title': title,
            'description': description,
            'specifications': specifications_text
        }

        return extracted_data    

class GoogleManager:
    def __init__(self, session_logger, credential_manager):
        """Initialize the Google Manager with credentials"""
        self.save_request_log = session_logger.save_request_log
        self.credential_manager = credential_manager
        self.yt_base_url = "https://www.googleapis.com/youtube/v3"
        
        
    def _make_youtube_request(self, endpoint, params, log_identifier):
        """
        Make a request to YouTube API with error handling.
        
        Args:
            endpoint (str): API endpoint path
            params (dict): Query parameters
            log_identifier (str): Identifier for logging
            
        Returns:
            dict: JSON response from the API
            
        Raises:
            StopProcessingError: For authentication errors
            RetryableError: For temporary API issues
        """
        params['key'] = self.credential_manager.get_api_key('google')
        url = f"{self.yt_base_url}/{endpoint}"

        def _call():
            try:
                response = requests.get(url, params=params)
                _handle_http_errors(response)
                data = response.json()
                self.save_request_log(data, "YouTube", log_identifier)
                return data
            except requests.RequestException as e:
                raise RetryableError(f"Network error: {e}")

        try:
            return _exponential_backoff(
                _call,
                max_retries=3,
                exc_types=(RetryableError,)
            )
        except (StopProcessingError, SkippableError, RetryableError):
            raise
    
    def get_youtube_channel_videos(self, channel_name, max_results=50):
        """
        Fetch video IDs from a YouTube channel.
        
        Args:
            channel_name (str): Name of the YouTube channel
            max_results (int): Maximum number of videos to return (default: 50)
            
        Returns:
            list: List of video IDs
        """
        # First, get the channel ID
        channel_params = {
            'q': channel_name,
            'type': 'channel',
            'part': 'id',
            'maxResults': 1
        }
        
        channel_data = self._make_youtube_request(
            'search',
            channel_params,
            f"channel_search_{channel_name}"
        )
        
        if not channel_data.get('items'):
            raise SkippableError(f"No channel found with name: {channel_name}")
        
        channel_id = channel_data['items'][0]['id']['channelId']
        
        # Then get the videos from the channel
        videos_params = {
            'channelId': channel_id,
            'type': 'video',
            'part': 'id',
            'order': 'date',
            'maxResults': max_results
        }
        
        videos_data = self._make_youtube_request(
            'search',
            videos_params,
            f"channel_videos_{channel_name}"
        )
        
        return [item['id']['videoId'] for item in videos_data.get('items', [])]
    
    def get_youtube_search_videos(self, search_query, max_results=50, order='relevance'):
        """
        Search for YouTube videos based on a query string.
        
        Args:
            search_query (str): Search query string
            max_results (int): Maximum number of videos to return (default: 50)
            order (str): Order of results. Options: 'date', 'rating', 'relevance', 
                        'title', 'videoCount', 'viewCount' (default: 'relevance')
            
        Returns:
            list: List of video IDs
        """
        search_params = {
            'q': search_query,
            'type': 'video',
            'part': 'id',
            'maxResults': max_results,
            'order': order
        }
        
        search_data = self._make_youtube_request(
            'search',
            search_params,
            f"search_videos_{search_query}"
        )
        
        return [item['id']['videoId'] for item in search_data.get('items', [])]
    
    def get_youtube_transcript(self, video_id):
        """
        Get the transcript/captions for a YouTube video using youtube_transcript_api.
        
        Args:
            video_id (str): YouTube video ID
            
        Returns:
            str: Combined transcript text
            
        Raises:
            SkippableError: If no captions are available
            RetryableError: For temporary API issues
        
        If needed in the future it can implement also: language, translate_to, preserve_formatting, preserve_timestamps
        """
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Combine all text entries into a single string
            transcript_text = ' '.join(entry['text'] for entry in transcript_list)
            
            # Log the operation (without the full transcript to save space)
            self.save_request_log(
                {"video_id": video_id, "transcript_length": len(transcript_text)},
                "YouTube",
                f"transcript_{video_id}"
            )
            
            return transcript_text
            
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            raise SkippableError(f"No transcript available for video {video_id}: {str(e)}")
        except Exception as e:
            raise RetryableError(f"Failed to fetch transcript: {str(e)}")


def universal_paginator(request_method, query: str, num_results=None, **kwargs):
    start_page = kwargs.get('start_page', 1)
    max_retries = 3
    pages_per_request = 5 if not 'serpapi' in request_method.__name__ else 1
    
    current_page = start_page
    total_results = None
    processed_results = 0
    last_results_count = 0
    next_page_filter = None ###############################################################################
    
    while True:
        current_kwargs = kwargs.copy()
        
        if 'serpapi' in request_method.__name__:
            current_kwargs['processed_results'] = processed_results
            if next_page_filter:
                current_kwargs['filter'] = next_page_filter
        else:
            current_kwargs['pages'] = pages_per_request
            current_kwargs['start_page'] = str(current_page)
        
        retry_count = 0
        page_results = None
        
        while retry_count < max_retries:
            result = request_method(query, **current_kwargs)
            if isinstance(result, tuple): ##################################################################
                page_results, next_page_filter = result
            else:
                page_results = result
                next_page_filter = None

            if not page_results:
                break
            
            current_processed = processed_results
            current_batch_size = len(page_results)
            
            # If we have a num_results limit, only take what we need
            if num_results and (current_processed + current_batch_size) > num_results:
                page_results = page_results[:num_results - current_processed]
                current_batch_size = len(page_results)  # Recalculate after truncating

            # Calculate absolute positions
            base_position = current_processed + 1
            for i, result in enumerate(page_results):
                result['absolute_position'] = base_position + i
            
            # Update total_results if available
            if total_results is None and page_results and page_results[0].get('total_results_number'):
                total_results = min(
                    page_results[0].get('total_results_number'),
                    num_results if num_results else float('inf')
                )
            
            # Update counters safely
            last_results_count = current_batch_size
            processed_results = current_processed + current_batch_size
            
            progress_info = {
                'current_page': current_page,
                'total_results': total_results,
                'processed_results': processed_results
            }
            
            yield page_results, progress_info
            break
                
        
        # Break conditions with explicit logging
        if not page_results:
            break
        if num_results and processed_results >= num_results:
            break
        if last_results_count == 2:
            break
        
        current_page = current_page + pages_per_request
