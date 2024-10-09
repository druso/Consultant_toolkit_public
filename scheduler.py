import time
from datetime import datetime

def run_scheduler():
    print(f"{datetime.now()} - Cronjob: Scheduler started")
    
    while True:
        print(f"{datetime.now()} - Cronjob: I'm alive")
        time.sleep(3600)

if __name__ == "__main__":
    run_scheduler()