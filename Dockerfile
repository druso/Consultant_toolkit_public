FROM python:3.9-slim 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY Hello.py /app/Hello.py
WORKDIR /app
EXPOSE 8501
CMD ["streamlit", "run", "Hello.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false", "--client.toolbarMode", "minimal"]
