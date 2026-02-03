
import asyncio
import logging
import datetime
import pytz
from multi_agent_pipeline import MultiAgentPipeline

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("scheduler_service")

# IST Configuration
IST = pytz.timezone('Asia/Kolkata')
SCHEDULE_MINUTES = [0, 15, 30, 45]
OFFSET_SECONDS = 30 # Run 30s after the mark (e.g. 09:15:30)

class SchedulerService:
    def __init__(self, symbols=["NIFTY", "BANKNIFTY"]):
        self.pipeline = MultiAgentPipeline()
        self.symbols = symbols
        self.running = False

    async def start(self):
        self.running = True
        logger.info("Scheduler Service Started (IST aligned).")
        
        while self.running:
            now_ist = datetime.datetime.now(IST)
            
            # Simple sleep loop to next 15-min mark + 30s
            # Calculate seconds to next target
            next_target = self._get_next_run_time(now_ist)
            
            sleep_seconds = (next_target - now_ist).total_seconds()
            logger.info(f"Sleeping for {sleep_seconds:.1f}s until {next_target.time()} IST...")
            
            await asyncio.sleep(sleep_seconds)
            
            # Execute Pipeline
            logger.info(f"Waking up! Executing pipeline at {datetime.datetime.now(IST)}")
            await self._execute_cycle()

    def _get_next_run_time(self, current_time):
        """
        Calculate next 00, 15, 30, 45 minute mark + 30 seconds.
        """
        next_time = current_time.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
        
        while next_time.minute not in SCHEDULE_MINUTES:
             next_time += datetime.timedelta(minutes=1)
             
        # Add offset
        next_time = next_time + datetime.timedelta(seconds=OFFSET_SECONDS)
        
        # If we already passed this target (due to processing time overlapping), jump to next
        if next_time < current_time:
             # Find next slot
             next_time += datetime.timedelta(minutes=15)
             
        return next_time

    async def _execute_cycle(self):
        """
        Run pipeline for all symbols concurrently.
        """
        tasks = [self.pipeline.run_pipeline(sym) for sym in self.symbols]
        results = await asyncio.gather(*tasks)
        
        for res in results:
            sym = res.get("symbol")
            decision = res.get("final_report", {}).get("final_decision")
            logger.info(f"PIPELINE COMPLETE [{sym}]: {decision}")

if __name__ == "__main__":
    # Standalone run
    service = SchedulerService()
    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        logger.info("Scheduler stopped.")
