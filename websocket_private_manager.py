import asyncio
import websockets
import json
import time
import hmac
import hashlib
import urllib.parse
from collections import OrderedDict
import logging
import threading
import requests
from typing import Dict, Callable, Any, List, Optional
import schedule
import datetime

logger = logging.getLogger('tradebot.websocket_private')

class WebSocketPrivateManager:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.toobit.com"
        self.ws_url = None
        self.websocket = None
        self.loop = None
        self.thread = None
        self.running = False
        self.pending_requests = {}
        self.authenticated = False
        self.listen_key = None
        self.last_listen_key_update = 0
        self.connection_start_time = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        # Callbacks for different event types
        self.order_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.position_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.trade_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.balance_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # Order management
        self.pending_orders: Dict[str, Callable] = {}  # orderId -> callback
        self.order_status: Dict[str, str] = {}    # orderId -> status
        
        # Event buffer for ordering
        self.event_buffer: List[Dict[str, Any]] = []
        self.last_event_time = 0
        
        # Start keepalive timer
        self.start_keepalive_timer()
        
    def set_order_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for order events"""
        self.order_callback = callback
        
    def set_position_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for position events"""
        self.position_callback = callback
        
    def set_trade_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for trade events"""
        self.trade_callback = callback
        
    def set_balance_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for balance events"""
        self.balance_callback = callback
    
    def _generate_signature(self, params: dict) -> str:
        sorted_params = OrderedDict(sorted(params.items()))
        query_string = urllib.parse.urlencode(sorted_params)
        return hmac.new(self.secret_key.encode('utf-8'),
                         query_string.encode('utf-8'),
                         hashlib.sha256).hexdigest()
    
    def _get_listen_key(self, force_new: bool = False) -> bool:
        """دریافت listenKey از طریق REST API"""
        if not force_new and self.listen_key and self._is_listen_key_valid():
            logger.info("Using existing valid listenKey")
            return True
            
        params = {
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = self._generate_signature(params)
        
        headers = {"X-BB-APIKEY": self.api_key}
        
        try:
            # Try to get existing listenKey first
            if not force_new:
                response = requests.post(
                    f"{self.base_url}/api/v1/listenKey",
                    params=params,
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"ListenKey response: {data}")
                
                if 'listenKey' in data:
                    self.listen_key = data['listenKey']
                    self.last_listen_key_update = time.time()
                    self.ws_url = f"wss://stream.toobit.com/api/v1/ws/{self.listen_key}"
                    logger.info(f"Existing listenKey obtained: {self.listen_key}")
                    return True
                elif data.get('code') == 200 and 'data' in data and 'listenKey' in data['data']:
                    self.listen_key = data['data']['listenKey']
                    self.last_listen_key_update = time.time()
                    self.ws_url = f"wss://stream.toobit.com/api/v1/ws/{self.listen_key}"
                    logger.info(f"Existing listenKey obtained: {self.listen_key}")
                    return True
            
            # If no existing key or force_new, create new one
            logger.info("Creating new listenKey")
            response = requests.post(
                f"{self.base_url}/api/v1/listenKey",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            if 'listenKey' in data:
                self.listen_key = data['listenKey']
                self.last_listen_key_update = time.time()
                self.ws_url = f"wss://stream.toobit.com/api/v1/ws/{self.listen_key}"
                logger.info(f"New listenKey obtained: {self.listen_key}")
                return True
            elif data.get('code') == 200 and 'data' in data and 'listenKey' in data['data']:
                self.listen_key = data['data']['listenKey']
                self.last_listen_key_update = time.time()
                self.ws_url = f"wss://stream.toobit.com/api/v1/ws/{self.listen_key}"
                logger.info(f"New listenKey obtained: {self.listen_key}")
                return True
            else:
                logger.error(f"Failed to get listenKey: {data}")
                return False
        except Exception as e:
            logger.error(f"Error getting listenKey: {e}")
            return False
    
    def _is_listen_key_valid(self) -> bool:
        """Check if current listenKey is still valid (less than 55 minutes old)"""
        if not self.last_listen_key_update:
            return False
        return (time.time() - self.last_listen_key_update) < 3300  # 55 minutes
    
    def _should_reconnect(self) -> bool:
        """Check if we should reconnect (24 hours limit)"""
        if not self.connection_start_time:
            return False
        return (time.time() - self.connection_start_time) >= 86400  # 24 hours
    
    def start_keepalive_timer(self):
        """Start timer for automatic listenKey keepalive"""
        def keepalive_job():
            if self.running and self.listen_key:
                self.keepalive_listen_key()
        
        # Schedule keepalive every 50 minutes
        schedule.every(50).minutes.do(keepalive_job)
        
        def run_schedule():
            while self.running:
                schedule.run_pending()
                time.sleep(1)
        
        timer_thread = threading.Thread(target=run_schedule, daemon=True)
        timer_thread.start()
        logger.info("Keepalive timer started")
    
    async def _connect(self):
        try:
            # Check if we need to reconnect due to 24h limit
            if self._should_reconnect():
                logger.info("24h connection limit reached, reconnecting with new listenKey")
                self.close_listen_key()
                self._get_listen_key(force_new=True)
                self.connection_start_time = time.time()
                self.reconnect_attempts = 0
            else:
                # Get listenKey (existing or new)
                if not self._get_listen_key():
                    self.running = False
                    return
                
                if not self.connection_start_time:
                    self.connection_start_time = time.time()
            
            logger.info(f"Connecting to: {self.ws_url}")
            self.websocket = await websockets.connect(self.ws_url)
            logger.info("WebSocket private connection established")
            self.running = True
            self.authenticated = True
            self.reconnect_attempts = 0
            
            await self._listen()
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.running = False
            self._handle_reconnect()
    
    def _handle_reconnect(self):
        """Handle reconnection logic"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logger.info(f"Attempting to reconnect ({self.reconnect_attempts}/{self.max_reconnect_attempts})...")
            time.sleep(min(self.reconnect_attempts * 5, 30))  # Exponential backoff
            
            # Restart connection in new thread
            def restart_connection():
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.loop.run_until_complete(self._connect())
            
            restart_thread = threading.Thread(target=restart_connection, daemon=True)
            restart_thread.start()
        else:
            logger.error("Max reconnection attempts reached. Stopping WebSocket manager.")
            self.running = False
    
    async def _listen(self):
        try:
            while self.running:
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=30)
                    data = json.loads(message)
                    
                    logger.debug(f"Received message: {data}")
                    
                    # Handle ping/pong
                    if 'ping' in data:
                        await self.websocket.send(json.dumps({'pong': data['ping']}))
                        continue
                    
                    # Handle user data stream updates
                    if isinstance(data, list) and len(data) > 0:
                        event = data[0]
                        event_type = event.get('e')
                        
                        # Add to buffer for ordering
                        self._add_event_to_buffer(event)
                        
                        # Process events in order
                        self._process_buffered_events()
                    
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await self.websocket.send(json.dumps({'ping': int(time.time() * 1000)}))
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    self.running = False
                    self._handle_reconnect()
                    break
                except Exception as e:
                    logger.error(f"Error in WebSocket listener: {e}")
                    self.running = False
                    self._handle_reconnect()
                    break
        except Exception as e:
            logger.error(f"WebSocket listener error: {e}")
            self.running = False
            self._handle_reconnect()
        finally:
            self.running = False
    
    def _add_event_to_buffer(self, event: Dict[str, Any]):
        """Add event to buffer and maintain order"""
        event_time = event.get('E', 0)
        
        # Add event to buffer
        self.event_buffer.append(event)
        
        # Sort buffer by event time
        self.event_buffer.sort(key=lambda x: x.get('E', 0))
        
        # Keep only last 100 events to prevent memory issues
        if len(self.event_buffer) > 100:
            self.event_buffer = self.event_buffer[-100:]
    
    def _process_buffered_events(self):
        """Process buffered events in chronological order"""
        if not self.event_buffer:
            return
            
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Process events that are older than 1 second (to ensure we have all events for that timestamp)
        events_to_process = [
            event for event in self.event_buffer 
            if current_time - event.get('E', 0) > 1000
        ]
        
        for event in events_to_process:
            self._process_single_event(event)
            self.event_buffer.remove(event)
    
    def _process_single_event(self, event: Dict[str, Any]):
        """Process a single event based on its type"""
        event_type = event.get('e')
        
        if event_type == 'outboundContractAccountInfo':
            logger.debug("Balance update received")
            if self.balance_callback:
                self.balance_callback(event)
                
        elif event_type == 'outboundContractPositionInfo':
            logger.debug("Position update received")
            if self.position_callback:
                self.position_callback(event)
                
        elif event_type == 'contractExecutionReport':
            logger.debug("Order update received")
            self._handle_order_update(event)
            if self.order_callback:
                self.order_callback(event)
                
        elif event_type == 'ticketInfo':
            logger.debug("Trade update received")
            if self.trade_callback:
                self.trade_callback(event)
    
    def _handle_order_update(self, order_data):
        """Handle order update events"""
        order_id = order_data.get('i')
        status = order_data.get('X')
        
        logger.info(f"Order update received: ID={order_id}, Status={status}")
        
        # Update order status
        if order_id:
            self.order_status[order_id] = status
            
        # Check if there's a pending callback for this order
        if order_id in self.pending_orders:
            callback = self.pending_orders.pop(order_id)
            if callback:
                callback(order_data)
    
    async def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "MARKET",
                          price: Optional[float] = None, client_order_id: Optional[str] = None, 
                          callback: Optional[Callable] = None):
        """Place an order via WebSocket"""
        if not self.running or not self.websocket:
            logger.error("WebSocket not connected")
            return None
            
        order_params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity
        }
        
        if price:
            order_params['price'] = price
        if client_order_id:
            order_params['newClientOrderId'] = client_order_id
            
        message = {
            'id': f"order_{int(time.time() * 1000)}",
            'method': 'POST',
            'path': '/api/v1/futures/order',
            'params': order_params
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            logger.info(f"Order placed via WebSocket: {message}")
            
            # Store callback for order update
            if callback and client_order_id:
                self.pending_orders[client_order_id] = callback
                
            return {'status': 'sent', 'clientOrderId': client_order_id}
        except Exception as e:
            logger.error(f"Error placing order via WebSocket: {e}")
            return None
    
    async def cancel_order(self, symbol: str, order_id: str, callback: Optional[Callable] = None):
        """Cancel an order via WebSocket"""
        if not self.running or not self.websocket:
            logger.error("WebSocket not connected")
            return None
            
        message = {
            'id': f"cancel_{int(time.time() * 1000)}",
            'method': 'DELETE',
            'path': '/api/v1/futures/order',
            'params': {
                'symbol': symbol,
                'orderId': order_id
            }
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            logger.info(f"Cancel order sent via WebSocket: {message}")
            
            # Store callback for order update
            if callback:
                self.pending_orders[order_id] = callback
                
            return {'status': 'sent', 'orderId': order_id}
        except Exception as e:
            logger.error(f"Error cancelling order via WebSocket: {e}")
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None):
        """Get open orders via WebSocket"""
        if not self.running or not self.websocket:
            logger.error("WebSocket not connected")
            return None
            
        params = {}
        if symbol:
            params['symbol'] = symbol
            
        message = {
            'id': f"open_orders_{int(time.time() * 1000)}",
            'method': 'GET',
            'path': '/api/v1/futures/openOrders',
            'params': params
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            logger.info(f"Get open orders sent via WebSocket: {message}")
            return {'status': 'sent'}
        except Exception as e:
            logger.error(f"Error getting open orders via WebSocket: {e}")
            return None
    
    def _send_request(self, method: str, params: dict) -> dict:
        # در حال حاضر، WebSocket خصوصی فقط برای دریافت داده‌ها استفاده می‌شود
        # عملیات خصوصی از طریق REST API انجام می‌شود
        logger.error("WebSocket private is read-only. Use REST API for operations.")
        return None
    
    def start(self):
        if self.thread and self.thread.is_alive():
            return
            
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._connect())
            
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        logger.info("WebSocket private manager started")
        
    def stop(self):
        self.running = False
        if self.websocket:
            asyncio.run_coroutine_threadsafe(
                self.websocket.close(), 
                self.loop
            )
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("WebSocket private manager stopped")
    
    def keepalive_listen_key(self):
        """تمدید اعتبار listenKey"""
        if not self.listen_key:
            return False
            
        params = {
            'listenKey': self.listen_key,
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = self._generate_signature(params)
        
        headers = {"X-BB-APIKEY": self.api_key}
        
        try:
            response = requests.put(
                f"{self.base_url}/api/v1/listenKey",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('code') == 200 or data == {}:
                self.last_listen_key_update = time.time()
                logger.info("ListenKey keepalive successful")
                return True
            else:
                logger.error(f"Failed to keepalive listenKey: {data}")
                return False
        except Exception as e:
            logger.error(f"Error keeping alive listenKey: {e}")
            return False
    
    def close_listen_key(self):
        """بستن stream و invalidate کردن listenKey"""
        if not self.listen_key:
            return False
            
        params = {
            'listenKey': self.listen_key,
            'timestamp': int(time.time() * 1000)
        }
        params['signature'] = self._generate_signature(params)
        
        headers = {"X-BB-APIKEY": self.api_key}
        
        try:
            response = requests.delete(
                f"{self.base_url}/api/v1/listenKey",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('code') == 200 or data == {}:
                logger.info("ListenKey closed successfully")
                self.listen_key = None
                self.last_listen_key_update = 0
                return True
            else:
                logger.error(f"Failed to close listenKey: {data}")
                return False
        except Exception as e:
            logger.error(f"Error closing listenKey: {e}")
            return False 