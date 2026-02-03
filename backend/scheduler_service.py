import asyncio
import logging
import datetime
import pytz
from multi_agent_pipeline import MultiAgentPipeline
from config_manager import ConfigManager

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
    def __init__(self):
        self.pipeline = MultiAgentPipeline()
        self.running = False

    async def start(self):
        self.running = True
        logger.info("Scheduler Service Started (IST aligned).")
        
        while self.running:
            now_ist = datetime.datetime.now(IST)
            
            # Check market hours (9:15 to 15:30)
            market_start = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
            market_end = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
            
            # If after market hours, sleep until next day 9:15:30
            if now_ist > market_end:
                logger.info("Market closed. Sleeping until tomorrow 09:15:30...")
                next_start = market_start + datetime.timedelta(days=1)
                next_start = next_start.replace(second=OFFSET_SECONDS)
                sleep_seconds = (next_start - now_ist).total_seconds()
                await asyncio.sleep(sleep_seconds)
                continue
                
            # If before market hours, sleep until today 9:15:30
            if now_ist < market_start:
                 logger.info("Before market open. Sleeping until 09:15:30...")
                 next_start = market_start.replace(second=OFFSET_SECONDS)
                 sleep_seconds = (next_start - now_ist).total_seconds()
                 await asyncio.sleep(sleep_seconds)
                 continue

            # Within Market Hours: Find next 15-min mark
            next_target = self._get_next_run_time(now_ist)
            
            # If next target is past market close, clamp to market close? 
            # Actually, standard is until 15:30. 15:30 is the last run if it aligns.
            if next_target > market_end.replace(second=OFFSET_SECONDS + 30): # Buffer
                 # Wait for tomorrow
                 continue
            
            sleep_seconds = (next_target - now_ist).total_seconds()
            logger.info(f"Sleeping for {sleep_seconds:.1f}s until {next_target.time()} IST...")
            
            await asyncio.sleep(sleep_seconds)
            
            # Execute Pipeline
            # Load active symbol dynamically from config
            symbol = ConfigManager.get_active_symbol()
            logger.info(f"⏰ Wake up! Executing pipeline for {symbol} at {datetime.datetime.now(IST)}")
            
            # Run for SINGLE active company as requested
            await self._execute_cycle([symbol])

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

    async def _execute_cycle(self, symbols):
        """
        Run pipeline for all symbols concurrently.
        """
        tasks = [self.pipeline.run_pipeline(sym) for sym in symbols]
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
