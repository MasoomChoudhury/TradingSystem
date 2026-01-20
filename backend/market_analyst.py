"""
Market Analyst Agent ("Trading Eyes")

- Scheduler: Runs every 15 minutes (aligned to market hours)
- Input: 15m Candle Data + Chart Image
- Analysis: Gemini 1.5 Pro (Vision)
- Output: "KEEP", "STOP", "SWITCH" recommendation to Supervisor
"""
import os
import json
import logging
import asyncio
import pandas as pd
import mplfinance as mpf
from datetime import datetime, time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from tools.openalgo_tools import get_openalgo_client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class MarketAnalystAgent:
    def __init__(self):
        """Initialize the Market Analyst Agent."""
        self.scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")
        
        # Initialize Vision Model (Gemini 1.5 Pro/Flash for images)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",  # Using Pro for better vision analysis
            google_api_key=GEMINI_API_KEY,
            temperature=0.2,
        )
        
        self.active_strategy = "ORB" # Default, should be synced with Executor
        self.active_symbol = "BANKNIFTY" # Default
        
        # Schedule: Every 15 mins at :00, :15, :30, :45 + 30 seconds
        # Indian Market Hours: 09:15 to 15:30
        self.scheduler.add_job(
            self.run_analysis_cycle,
            trigger=CronTrigger(
                day_of_week='mon-fri',
                hour='9-15',
                minute='0,15,30,45',
                second='30',
                timezone='Asia/Kolkata'
            ),
            id='market_analysis_job',
            replace_existing=True
        )
        
        logger.info("👁️ Market Analyst (Trading Eyes) initialized with scheduler")

    def start(self):
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("👁️ Market Analyst scheduler started")

    def stop(self):
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("👁️ Market Analyst scheduler stopped")

    async def fetch_data(self, symbol: str, timeframe: str = "15m"):
        """Fetch OHLVC data from OpenAlgo."""
        try:
            client = get_openalgo_client()
            
            # Calculate dates: Get last 7 days to ensure enough 15m candles
            # 15m candles in a week: ~25 candles/day * 5 days = 125 candles
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # Assumes 'NSE' for Index. If trading futures, this should be 'NFO' and symbol updated.
            # Using NSE Index data for general market analysis.
            logger.info(f"Fetching {timeframe} data for {symbol} from {start_date.date()} to {end_date.date()}...")
            
            df = client.history(
                symbol=symbol,
                exchange="NSE", 
                interval=timeframe,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            
            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
                
            # Ensure columns are correct for mplfinance (Open, High, Low, Close, Volume)
            # OpenAlgo usually returns: date, open, high, low, close, volume, etc.
            # We need to set 'date' as DatetimeIndex
            
            # Rename columns if needed to capitalized title case for mplfinance
            df.columns = [c.capitalize() for c in df.columns]
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif 'Datetime' in df.columns:
                 df['Datetime'] = pd.to_datetime(df['Datetime'])
                 df.set_index('Datetime', inplace=True)
                 
            return df

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None

    async def generate_chart(self, df: pd.DataFrame, filename: str = "temp/latest_chart.png") -> str:
        """Generate candlestick chart image using mplfinance."""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Ensure index is Datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Style configuration
            mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
            s  = mpf.make_mpf_style(marketcolors=mc)

            # Plot with moving averages
            mpf.plot(
                df,
                type='candle',
                mav=(9, 20),
                volume=True,
                style=s,
                title=f"{self.active_symbol} Analysis",
                savefig=filename,
                tight_layout=True
            )
            
            return filename
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return None

    async def analyze_with_vision(self, image_path: str, strategy_context: str) -> dict:
        """Send image and context to Gemini for analysis."""
        try:
            import base64
            
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

            prompt = f"""
            You are a veteran Market Analyst ("Trading Eyes").
            
            **STRATEGY CONTEXT:**
            {strategy_context}
            
            **YOUR TASK:**
            Analyze the attached 15-minute candlestick chart. Look for price action patterns that confirm or contradict the current strategy.
            
            **OUTPUT JSON ONLY:**
            {{
                "recommendation": "KEEP" | "STOP" | "SWITCH",
                "confidence": 0.0 to 1.0,
                "observations": ["observation 1", "observation 2"],
                "reasoning": "Detailed visual analysis...",
                "suggested_action": "Stay in trade / Close position / Switch to [Strategy Name]"
            }}
            """
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    }
                ]
            )
            
            response = await self.llm.ainvoke([message])
            
            # Parse JSON from response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            return json.loads(content.strip())

        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return {"error": str(e)}

    async def run_analysis_cycle(self):
        """Main scheduled task loop."""
        logger.info("👁️ Starting scheduled market analysis...")
        
        # 1. Fetch Data
        df = await self.fetch_data(self.active_symbol, "15m")
        
        if df is None or df.empty:
             logger.error("Failed to fetch real market data. Aborting analysis.")
             self.last_analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "recommendation": "ERROR",
                "confidence": 0.0,
                "observations": ["Failed to fetch market data", "Check OpenAlgo connection"],
                "reasoning": "Could not retrieve historical data. The backend might be disconnected or outside active hours.",
                "inputs": {
                    "strategy": f"Current Strategy: {self.active_strategy}",
                    "market_data_summary": "No Data Received"
                }
             }
             return self.last_analysis_result
        
        # 2. Generate Chart
        chart_path = await self.generate_chart(df)
        
        if chart_path:
            # 3. Vision Analysis
            logger.info(f"Chart generated at {chart_path}. Analyzing...")
            strategy_text = f"Current Strategy: {self.active_strategy}. We are looking for trend continuation."
            result = await self.analyze_with_vision(
                chart_path, 
                strategy_text
            )
            
            # Store result in memory
            self.last_analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "recommendation": result.get("recommendation"),
                "confidence": result.get("confidence"),
                "observations": result.get("observations"),
                "reasoning": result.get("reasoning"),
                "inputs": {
                    "strategy": strategy_text,
                    "market_data_summary": df.tail(5).to_dict()
                }
            }
            logger.info(f"👁️ Analysis Result: {json.dumps(result, indent=2)}")
            
            # 4. Notify Supervisor of analysis result
            from agent_comms import send_agent_message
            send_agent_message(
                from_agent="analyst",
                to_agent="supervisor",
                message_type="advisory",
                content=f"Visual Analysis: {result.get('recommendation')} ({result.get('confidence')*100:.0f}% confidence)",
                metadata=self.last_analysis_result
            )
            
            # 5. Trigger session actions AND automatic trade execution
            recommendation = result.get('recommendation', 'KEEP').upper()
            try:
                from trading_session import get_session_manager
                from auto_trade_executor import get_auto_executor, handle_analyst_signal
                
                manager = get_session_manager()
                executor = get_auto_executor()
                
                if recommendation == 'STOP':
                    # Pause session AND close any open positions
                    manager.pause_session(f"Analyst: {result.get('reasoning', 'Market unfavorable')[:100]}")
                    
                    # AUTO-EXECUTE: Close position
                    close_result = await handle_analyst_signal("STOP", reason=result.get('reasoning', '')[:100])
                    
                    send_agent_message(
                        from_agent="analyst",
                        to_agent="orchestrator",
                        message_type="advisory",
                        content=f"Trading PAUSED. Position closed: {close_result.get('status')}",
                        metadata={"recommendation": "STOP", "execution": close_result}
                    )
                    
                elif recommendation == 'SWITCH':
                    # Switch strategy AND execute the switch
                    new_strategy = manager.switch_strategy(f"Analyst: {result.get('reasoning', 'Strategy adjustment')[:100]}")
                    
                    if new_strategy:
                        # AUTO-EXECUTE: Close old position and open new one
                        switch_result = await handle_analyst_signal("SWITCH", strategy=new_strategy, reason=result.get('reasoning', '')[:100])
                        
                        send_agent_message(
                            from_agent="analyst",
                            to_agent="orchestrator",
                            message_type="advisory",
                            content=f"Strategy SWITCHED to {new_strategy.name} ({new_strategy.bias}). Trade executed.",
                            metadata={"new_strategy": new_strategy.to_dict(), "execution": switch_result}
                        )
                        
                elif recommendation == 'KEEP':
                    # Check if we need to open initial position
                    pos_status = executor.get_position_status()
                    if pos_status.get("status") == "flat" and manager.current_session:
                        # No position but session active - open initial position
                        current_strategy = manager.current_session.get_active_strategy()
                        if current_strategy:
                            open_result = await executor.open_position(current_strategy, "Initial entry on KEEP signal")
                            send_agent_message(
                                from_agent="analyst",
                                to_agent="orchestrator",
                                message_type="advisory",
                                content=f"Initial position opened: {current_strategy.bias}",
                                metadata={"execution": open_result}
                            )
                            
            except Exception as e:
                logger.warning(f"Auto-execution error: {e}")
            
            # Also log this as a completed task for the UI
            from task_tracker import start_agent_task, update_agent_task, complete_agent_task
            task_id = start_agent_task(
                name="👁️ Market Analysis",
                steps=["Fetch Data", "Generate Chart", "Vision Analysis"],
                description="Periodic 15m visual analysis"
            )
            update_agent_task(task_id, "Generating Chart", "running")
            update_agent_task(task_id, "Vision Analysis", "running")
            complete_agent_task(task_id, f"Analysis: {result.get('recommendation')} ({result.get('confidence')})")
            
            return self.last_analysis_result

        else:
            logger.error("Failed to generate chart. Skipping analysis.")
            self.last_analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "recommendation": "ERROR",
                "confidence": 0.0,
                "observations": ["Chart generation failed"],
                "reasoning": "Data was fetched but chart generation failed.",
                "inputs": {
                    "strategy": f"Current Strategy: {self.active_strategy}",
                    "market_data_summary": "Chart Generation Error"
                }
             }
            return self.last_analysis_result

# Singleton instance
market_analyst = MarketAnalystAgent()
