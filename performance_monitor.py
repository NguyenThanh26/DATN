import os
import time
import psutil
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.measurements = {}
        self.start_times = {}
        self._process = psutil.Process(os.getpid())

    def _get_ram_usage(self):
        mem_info = self._process.memory_info()
        return mem_info.rss / 1024**2

    def start_measurement(self, name: str):
        self.start_times[name] = time.time()
        self.measurements[name] = {"ram_start": self._get_ram_usage()}

    def end_measurement(self, name: str):
        if name not in self.start_times:
            raise ValueError(f"No measurement started with name: {name}")
        elapsed_time = time.time() - self.start_times[name]
        ram_end = self._get_ram_usage()
        self.measurements[name].update({
            "elapsed_time": elapsed_time,
            "ram_end": ram_end,
            "ram_diff": ram_end - self.measurements[name]["ram_start"]
        })

    def log_measurement(self, name: str):
        m = self.measurements[name]
        logger.info(f"Performance: {name}")
        logger.info(f"Time: {m['elapsed_time']:.2f} seconds")
        logger.info(f"RAM: {m['ram_end']:.2f} MB (Change: {m['ram_diff']:.2f} MB)")

    def print_summary(self):
        logger.info("\n=== PERFORMANCE SUMMARY ===")
        for name, m in self.measurements.items():
            if "elapsed_time" in m:
                logger.info(f"{name}: {m['elapsed_time']:.2f}s, RAM: +{m['ram_diff']:.2f} MB")
        logger.info("==========================\n")