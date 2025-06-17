FROM python:3.9-slim 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app 
WORKDIR /app
EXPOSE $PORT
CMD ["/bin/sh", "-c", "echo PORT=$PORT && python3 scheduler.py & streamlit run Hello.py --server.address 0.0.0.0 --server.port $PORT --server.fileWatcherType none --browser.gatherUsageStats false --client.toolbarMode minimal"]
