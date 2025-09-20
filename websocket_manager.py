import asyncio
import websockets
import json
import logging
import threading
from datetime import datetime
from typing import Callable, Dict, Any

logger = logging.getLogger('tradebot.websocket')

class WebSocketManager:
    def __init__(self, on_message_callback: Callable[[Dict[str, Any]], None] = None, on_connect_callback: Callable[[], None] = None):
        self.ws_url = "wss://stream.toobit.com/quote/ws/v1"
        self.on_message_callback = on_message_callback
        self.on_connect_callback = on_connect_callback
        self.websocket = None
        self.running = False
        self.loop = None
        self.thread = None
        self.subscriptions = {}
        self.connected = False
        self.kline_callback = None
        
    def set_kline_callback(self, callback):
        self.kline_callback = callback
        
    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.ws_url)
            logger.info("WebSocket connected successfully")
            self.running = True
            self.connected = True
            
            if self.on_connect_callback:
                self.on_connect_callback()
                
            await self.listen()
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.running = False
            self.connected = False
            
    async def listen(self):
        try:
            while self.running:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data.get('ping'):
                    await self.send_pong(data['ping'])
                    continue
                
                self.handle_websocket_message(data)
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket listen error: {e}")
        finally:
            self.running = False
            self.connected = False
            
    def handle_websocket_message(self, data):
        try:
            topic = data.get('topic')
            
            if topic and topic.startswith('kline'):
                if self.kline_callback and 'data' in data:
                    for kline_dict in data['data']:
                        self.kline_callback(kline_dict)
            elif topic == 'trade':
                if self.on_message_callback:
                    self.on_message_callback(data)
            elif data.get('pong'):
                pass
            elif data.get('f') and data.get('sendTime'):
                pass
            else:
                if data.get('code') and data.get('desc'):
                    logger.warning(f"WebSocket error: {data.get('desc')}")
                  
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            
    def subscribe_kline(self, symbol: str, interval: str):
        if not self.connected:
            logger.warning("WebSocket not connected, cannot subscribe to kline")
            return False

        message = {
            "symbol": symbol,
            "topic": f"kline_{interval}",
            "event": "sub",
            "params": {
                "binary": False
            }
        }

        try:
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(json.dumps(message)), 
                self.loop
            )
            logger.info(f"Subscribed to kline_{interval} for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error subscribing to kline_{interval} for {symbol}: {e}")
            return False
            
    async def send_pong(self, ping_value):
        if self.websocket:
            pong_message = {"pong": ping_value}
            await self.websocket.send(json.dumps(pong_message))
            
    def subscribe(self, symbol: str, topics: list):
        if not self.connected:
            return
            
        for topic in topics:
            message = {
                "symbol": symbol,
                "topic": topic,
                "event": "sub",
                "params": {
                    "binary": False,
                    "limit": 100
                }
            }
            
            try:
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps(message)), 
                    self.loop
                )
                logger.info(f"Subscribed to {topic} for {symbol}")
            except Exception as e:
                logger.error(f"Error subscribing to {topic}: {e}")
                return False
            
        return True
        
    def unsubscribe(self, symbol: str, topics: list):
        if not self.connected:
            return
            
        for topic in topics:
            message = {
                "symbol": symbol,
                "topic": topic,
                "event": "cancel",
                "params": {
                    "binary": False
                }
            }
            
            try:
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps(message)), 
                    self.loop
                )
                logger.info(f"Unsubscribed from {topic} for {symbol}")
            except Exception as e:
                logger.error(f"Error unsubscribing from {topic}: {e}")
            
    def start(self):
        if self.thread and self.thread.is_alive():
            return
            
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.connect())
            
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        logger.info("WebSocket thread started")
        
    def stop(self):
        self.running = False
        if self.websocket:
            asyncio.run_coroutine_threadsafe(
                self.websocket.close(), 
                self.loop
            )
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("WebSocket stopped")
        
    def send_ping(self):
        if self.websocket and self.running:
            ping_message = {"ping": int(datetime.now().timestamp() * 1000)}
            try:
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps(ping_message)), 
                    self.loop
                )
                logger.debug("Ping sent to WebSocket")
            except Exception as e:
                logger.error(f"Error sending ping: {e}")