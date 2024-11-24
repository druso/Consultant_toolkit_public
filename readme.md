### 👋 Welcome to Consultant Toolkit: 

This project has been developed during my free time, heavily assisted by ChatGPT and Gemini.

It is a collection of python functions wrapped in a cozy Streamlit app!

The tool will ask for [Groq](https://groq.com/), [Openai](https://platform.openai.com/), [Oxylab](https://oxylabs.io/) and [SerpAPI](https://serpapi.com/) API keys if not provided within the environment.

It can be used without setting the keys, but related service won't be available and errors may arise.
Finally the repository can be deployed easily using [Railway](https://railway.app/) or run locally as a streamlit app (streamlit run Hello.py)

With that said..

**🚧 This is a work in progress, and your feedback is super valuable! 🙏**

Navigate the tools using the sidebar on the left. Upload files and configure options (like your favorite LLM ).
Don't forget to take not of your  **session_id**! It's your key 🔑 to retrieving content and logs later on using the 💾 Session Recovery if needed. The tool runs on API keys with tiny budgets 💰, so use it wisely or help me find a 💸sponsor💸! 

⚠️ Important: Data is hosted on US servers 🇺🇸, and LLM requests share info with OpenAI or Groq. Be mindful of your data!

## 📈 Generative Excel

This let's you load a csv or an excel and run bulk operations on it. Once you'll load a file you can navigate through the tabs to access different functions.

* **LLM**:
    Here you can run bulk requests to a Large Language Model. You can select a column that will be used as the message to pass to the LLM and a system message with the instruction to do with that field. Like: 
        - assign the text to a category between the following 4...
        - summarize the text that is provided...
        - tell me if this needs my attention based on...
* **Google**:
    Here you can run bulk requests to Google, at the moment only the retrieval of organic search results for a given query is present.
    But we can add more, there are a lot of information that can be retrieved [🔗check out what can be done](https://serpapi.com/search-api) just ask, we can implement new requests quite easily.
* **Amazon**:
    Here you can retrieve product information or reviews from amazon, providing ASINs code. Much more can be added, feel free to take a look at oxylabs documentation. [🔗check out the endless opportunities](https://oxylabs.io/products/scraper-api/ecommerce) 
* **Youtube**:
    Provide a list of youtube video ids and retrieve the transcript of each video.
* **Crawler**:
    Here you can retrieve the content from a provided url. If you have urls in a column it will download in the dataframe the content of each website. Work can be done here, depending on how useful this is going to be
* **Table Handler**:
    Here you can make operations on the file (like drop columns...) 
* **Assistant Setup**:
    Here you can prepare create an assistant using the file that is being processed. You will be able to use the assistant in the 🤖 My Assistants section
 

You can keep working on the file by switching to different tabs so you can cuncatenate different requests. For example you can requests organic search results from serpAPI, crawl the results with the crawler and finally run an LLM to summarize the contents of the websites.

**Pro Tip**: some requests return structured data (like serpAPI), you can unroll the structured data using the "Colum Expander" function below in the Table Handler. Select the column to unroll and it will do the magic.

## 🏗️ Deep Extractor
Generative excel is great for getting few information out of a lot of queries. But what if you need to extract a lot of information from a few queries? Deep extractor is the tool for you! Paste a list of keywords or asins, select the information you need (like google results, product reviews, etc...) and schedule a batch to get everything you need.

## ⏳ Batches Monitor
Here you can monitor the status of scheduled batches. You can monitor their status, download processed files or cancel them.

## 🤖 My Assistants
If you created an assistant using the 📈 Generative Excel you can use it here, it will help you in visualizing insights about the file you created

## 🎙️ Audio Transcriber
Here you can load an audio of speech, have it transcrived and summarized.
You can change the behaviour of the summarization by optimizing the summarization prompt

## 📋 Doc Assistant
Here you can process any text document (word, txt, pdf) in order to quickly extract relevant information. Multiple documents can be uploaded at once. You follow these steps:
* **Prepare the content**
 The file needs to be divided into smaller chunk. Default options should work for most needs, but you never know. If you want to tinker you can change the length of the chunk of text and the overlap between chunks. The longer the chunks the less precise the search activity will be.
* **Study the content**
 Each chunk will be processed through an embedding function and made ready to be searched for meanings
* **Question the content**
 Finally you can input your request, the tool will search and retrieve the content, you will be shown a response to the question and a box with the relevant content extracted for the response.
 You can further configure the behaviour of the tool with:
    * *context chunk*: the tool will stitch neighbour chunks to the most relevant one. If you created a memory of small chunks, I suggest you keep some context sorrounding the relevant bit so that it's easier for the tool to grasp the relevant information to provide you with an answer
    * *number of content*: you can have the tool return not just the one bit of content that is most relevant, but more bits of relevant content across the whole text

## 🔧 Settings & Recovery
Here you can download logs from any session. Partially executed files and entire logs of any api call is available to dowload from here in case needed.
You can also check input API keys if your webmaster have not set them up for you and few other configurations
