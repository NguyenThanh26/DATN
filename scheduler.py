import schedule
import time
import threading
import logging
import os
from main import AudioTranscriptionService

logger = logging.getLogger(__name__)

running_lock = threading.Lock()
stop_event = threading.Event()

def prevent_overlap(func):
    def wrapper(*args, **kwargs):
        if running_lock.acquire(blocking=False):
            try:
                func(*args, **kwargs)
            finally:
                running_lock.release()
        else:
            logger.warning(f"Skipping {func.__name__} as previous process is running")
    return wrapper

@prevent_overlap
def auto_subtitle_process():
    logger.info("Running batch processing...")
    service = AudioTranscriptionService()
    service.process_batch()

def run_scheduler():
    logger.info("Starting scheduler")
    schedule.every(60).seconds.do(auto_subtitle_process)
    while not stop_event.is_set():
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    try:
        while not stop_event.is_set():
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Stopping scheduler")
        stop_event.set()