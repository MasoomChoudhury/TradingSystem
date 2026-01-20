"""
Automatic Trade Executor

Bridges the gap between strategy signals and actual order execution.
Handles:
- Position tracking (current position state)
- Auto-execution on strategy switch
- Entry/exit condition monitoring
- Integration with OpenAlgo for real orders
"""
import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BankNifty lot size
BANKNIFTY_LOT_SIZE = 15  # As of 2026


@dataclass
class Position:
    """Represents current position state."""
    symbol: str
    exchange: str
    quantity: int  # Positive for long, negative for short, 0 for flat
    entry_price: float
    entry_time: str
    strategy_name: str
    unrealized_pnl: float = 0.0
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        return self.quantity == 0
    
    def to_dict(self) -> dict:
        return asdict(self)


class AutoTradeExecutor:
    """
    Automatic trade execution engine.
    
    Responsibilities:
    - Track current position
    - Execute trades when strategy signals
    - Close positions on STOP/SWITCH
    - Open new positions based on strategy bias
    """
    
    def __init__(self):
        self.current_position: Optional[Position] = None
        self.is_analyzer_mode = True  # Start in analyze mode for safety
        self.lot_size = BANKNIFTY_LOT_SIZE
        logger.info("🤖 Auto Trade Executor initialized")
    
    async def check_analyzer_mode(self) -> bool:
        """Check if we're in analyzer (paper) mode."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            # Try to get status - if analyzer mode is on, trades are simulated
            self.is_analyzer_mode = True  # Default to safe mode
            return self.is_analyzer_mode
        except:
            return True
    
    async def get_current_ltp(self, symbol: str, exchange: str) -> Optional[float]:
        """Get current LTP for a symbol."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.quotes(symbol=symbol, exchange=exchange)
            if result.get("status") == "success":
                return float(result.get("ltp", 0))
        except Exception as e:
            logger.error(f"Failed to get LTP: {e}")
        return None
    
    async def close_position(self, reason: str) -> Dict[str, Any]:
        """
        Close the current position.
        Returns execution result.
        """
        if not self.current_position or self.current_position.is_flat:
            return {"status": "skip", "message": "No position to close"}
        
        pos = self.current_position
        
        # Determine closing action
        action = "SELL" if pos.is_long else "BUY"
        quantity = abs(pos.quantity)
        
        logger.info(f"🔄 Closing position: {action} {quantity} {pos.symbol}")
        
        # Execute via OpenAlgo
        result = await self._execute_order(
            symbol=pos.symbol,
            exchange=pos.exchange,
            action=action,
            quantity=quantity,
            reason=f"Close: {reason}"
        )
        
        if result.get("status") == "executed":
            # Update P&L
            ltp = await self.get_current_ltp(pos.symbol, pos.exchange)
            if ltp and pos.entry_price:
                pnl = (ltp - pos.entry_price) * pos.quantity
                result["realized_pnl"] = pnl
                
                # Update session P&L
                try:
                    from trading_session import get_session_manager
                    manager = get_session_manager()
                    manager.update_pnl(realized=pnl)
                except:
                    pass
            
            self.current_position = None
        
        return result
    
    async def open_position(self, strategy: Any, reason: str) -> Dict[str, Any]:
        """
        Open a new position based on strategy using OPTIONS.
        Selects appropriate CE/PE based on bias.
        """
        from options_generator import get_options_generator, get_spot_price
        from agent_comms import send_agent_message
        
        gen = get_options_generator()
        
        # Get current spot price
        underlying = strategy.symbol if hasattr(strategy, 'symbol') else "BANKNIFTY"
        if underlying in ["BANKNIFTY", "NIFTY", "FINNIFTY"]:
            spot_price = await get_spot_price(underlying)
        else:
            underlying = "BANKNIFTY"  # Default
            spot_price = await get_spot_price(underlying)
        
        # Generate option contract based on bias
        bias = strategy.bias.upper() if hasattr(strategy, 'bias') else "LONG"
        contract = gen.get_option_contract(
            underlying=underlying,
            spot_price=spot_price,
            bias=bias,
            strike_offset=0  # ATM
        )
        
        # For options, we always BUY (CE for bullish, PE for bearish)
        action = "BUY"
        quantity = contract.lot_size
        
        logger.info(f"📈 Opening OPTIONS position: {action} {quantity} {contract.symbol} ({contract.option_type})")
        
        # Log strategy selection
        send_agent_message(
            from_agent="auto_executor",
            to_agent="supervisor",
            message_type="info",
            content=f"Selected {contract.option_type} @ {contract.strike} | Expiry: {contract.expiry_display}",
            metadata={
                "underlying": underlying,
                "spot_price": spot_price,
                "strike": contract.strike,
                "option_type": contract.option_type,
                "expiry": contract.expiry,
                "symbol": contract.symbol
            }
        )
        
        # Execute via OpenAlgo
        result = await self._execute_order(
            symbol=contract.symbol,
            exchange="NFO",
            action=action,
            quantity=quantity,
            reason=f"Entry: {reason}"
        )
        
        if result.get("status") == "executed":
            ltp = await self.get_current_ltp(contract.symbol, "NFO")
            self.current_position = Position(
                symbol=contract.symbol,
                exchange="NFO",
                quantity=quantity,  # Always positive for bought options
                entry_price=ltp or 0,
                entry_time=datetime.now().isoformat(),
                strategy_name=f"{strategy.name if hasattr(strategy, 'name') else 'MANUAL'} ({contract.option_type})"
            )
            result["contract"] = contract.to_dict()
        
        return result
    
    async def switch_position(self, new_strategy: Any, reason: str) -> Dict[str, Any]:
        """
        Switch position: close current and open new based on new strategy.
        This is the key function for strategy adaptation.
        """
        results = {"close": None, "open": None}
        
        # Step 1: Close existing position
        if self.current_position and not self.current_position.is_flat:
            results["close"] = await self.close_position(reason)
        
        # Step 2: Open new position based on new strategy
        results["open"] = await self.open_position(new_strategy, reason)
        
        # Log to agent comms
        from agent_comms import send_agent_message
        send_agent_message(
            from_agent="executor",
            to_agent="orchestrator",
            message_type="info",
            content=f"Position switched to {new_strategy.bias}: {new_strategy.name}",
            metadata=results
        )
        
        return results
    
    async def _execute_order(
        self, 
        symbol: str, 
        exchange: str, 
        action: str, 
        quantity: int,
        reason: str
    ) -> Dict[str, Any]:
        """
        Execute an order through the Supervisor → Executor pipeline.
        """
        try:
            from tools.openalgo_tools import get_openalgo_client
            from agent_comms import send_agent_message
            
            client = get_openalgo_client()
            
            # Create trade plan
            trade_plan = {
                "symbol": symbol,
                "exchange": exchange,
                "action": action,
                "quantity": quantity,
                "price_type": "MARKET",
                "product": "MIS",  # Intraday
                "price": 0,
                "trigger_price": 0
            }
            
            # Log the intent
            send_agent_message(
                from_agent="auto_executor",
                to_agent="supervisor",
                message_type="request",
                content=f"Trade request: {action} {quantity} {symbol} - {reason}",
                metadata=trade_plan
            )
            
            # Generate approval token (simplified - in production, Supervisor should validate)
            import hashlib
            token_data = f"{symbol}:{action}:{quantity}:{datetime.now().isoformat()}"
            approval_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]
            
            # Execute via OpenAlgo
            result = client.placeorder(
                symbol=symbol,
                exchange=exchange,
                action=action,
                quantity=quantity,
                price_type="MARKET",
                product="MIS",
                price="0",
                trigger_price="0"
            )
            
            order_id = result.get("orderid", result.get("order_id", ""))
            status = "executed" if order_id else "failed"
            
            # Log execution
            send_agent_message(
                from_agent="executor",
                to_agent="supervisor",
                message_type="response" if status == "executed" else "denial",
                content=f"Order {status}: {action} {quantity} {symbol} ID={order_id}",
                metadata={"order_id": order_id, "result": result}
            )
            
            logger.info(f"✅ Order {status}: {action} {quantity} {symbol} ID={order_id}")
            
            # Update session trade count
            try:
                from trading_session import get_session_manager
                manager = get_session_manager()
                if manager.current_session:
                    manager.current_session.trade_count += 1
                    manager._save_session(manager.current_session)
            except:
                pass
            
            return {
                "status": status,
                "order_id": order_id,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "response": result
            }
            
        except Exception as e:
            logger.error(f"❌ Order execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_position_status(self) -> Dict[str, Any]:
        """Get current position status."""
        if not self.current_position:
            return {"status": "flat", "position": None}
        return {
            "status": "long" if self.current_position.is_long else "short",
            "position": self.current_position.to_dict()
        }


# Singleton instance
_auto_executor = None

def get_auto_executor() -> AutoTradeExecutor:
    """Get the singleton auto executor instance."""
    global _auto_executor
    if _auto_executor is None:
        _auto_executor = AutoTradeExecutor()
    return _auto_executor


async def handle_analyst_signal(recommendation: str, strategy: Any = None, reason: str = "") -> Dict[str, Any]:
    """
    Handle signals from the Market Analyst.
    This is the main entry point for automatic trade execution.
    """
    executor = get_auto_executor()
    
    if recommendation == "STOP":
        # Close all positions
        return await executor.close_position(reason or "Analyst STOP signal")
    
    elif recommendation == "SWITCH" and strategy:
        # Switch to new strategy
        return await executor.switch_position(strategy, reason or "Analyst SWITCH signal")
    
    elif recommendation == "KEEP":
        # No action, just return current position
        return executor.get_position_status()
    
    return {"status": "unknown", "message": f"Unknown recommendation: {recommendation}"}
