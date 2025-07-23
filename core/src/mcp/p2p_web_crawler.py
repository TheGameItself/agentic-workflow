#!/usr/bin/env python3
"""
P2P Web Crawler - Distributed Web Crawling Coordinator

This module implements distributed web crawling capabilities using the P2P network
to coordinate crawling tasks across multiple nodes. It enables collaborative
internet exploration without relying on centralized APIs.

Features:
- Distributed crawl task distribution
- Collaborative content sharing
- Load balancing across P2P network
- Duplicate detection and prevention
- Crawl result aggregation
- Network-wide crawl coordination
"""

import asyncio
import logging
import hashlib
import json
import time
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pickle
import gzip
import base64

from .p2p_network_integration import P2PNetworkIntegration, UserStatus, ServerCapability
from .web_social_engine import WebSocialEngine, CrawlTask, WebPage, CrawlStatus, CrawlPriority


class CrawlTaskType(Enum):
    """Types of crawl tasks"""
    SINGLE_URL = "single_url"
    WEBSITE_CRAWL = "website_crawl"
    DOMAIN_EXPLORATION = "domain_exploration"
    CONTENT_SEARCH = "content_search"
    RESEARCH_GATHERING = "research_gathering"


class P2PCrawlTask:
    """P2P network crawl task"""
    
    def __init__(self, 
                 task_id: str,
                 task_type: CrawlTaskType,
                 urls: List[str],
                 priority: CrawlPriority,
                 max_depth: int,
                 content_filters: Optional[List[str]] = None,
                 domain_restrictions: Optional[List[str]] = None,
                 created_by: str = None):
        
        self.task_id = task_id
        self.task_type = task_type
        self.urls = urls
        self.priority = priority
        self.max_depth = max_depth
        self.content_filters = content_filters or []
        self.domain_restrictions = domain_restrictions or []
        self.created_by = created_by
        self.created_at = datetime.now()
        self.assigned_to: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.status = CrawlStatus.PENDING
        self.results: List[WebPage] = []
        self.error_message: Optional[str] = None
        
        # P2P specific fields
        self.distributed_nodes: Set[str] = set()
        self.partial_results: Dict[str, List[WebPage]] = {}
        self.coordination_data: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for P2P transmission"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type.value,
            'urls': self.urls,
            'priority': self.priority.value,
            'max_depth': self.max_depth,
            'content_filters': self.content_filters,
            'domain_restrictions': self.domain_restrictions,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'assigned_to': self.assigned_to,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status.value,
            'error_message': self.error_message,
            'distributed_nodes': list(self.distributed_nodes),
            'coordination_data': self.coordination_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'P2PCrawlTask':
        """Create from dictionary"""
        task = cls(
            task_id=data['task_id'],
            task_type=CrawlTaskType(data['task_type']),
            urls=data['urls'],
            priority=CrawlPriority(data['priority']),
            max_depth=data['max_depth'],
            content_filters=data.get('content_filters', []),
            domain_restrictions=data.get('domain_restrictions', []),
            created_by=data.get('created_by')
        )
        
        task.assigned_to = data.get('assigned_to')
        task.started_at = datetime.fromisoformat(data['started_at']) if data.get('started_at') else None
        task.completed_at = datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None
        task.status = CrawlStatus(data['status'])
        task.error_message = data.get('error_message')
        task.distributed_nodes = set(data.get('distributed_nodes', []))
        task.coordination_data = data.get('coordination_data', {})
        
        return task


class P2PWebCrawler:
    """
    P2P Web Crawler for distributed internet exploration
    
    Features:
    - Distributed crawl task distribution
    - Collaborative content sharing
    - Load balancing across P2P network
    - Duplicate detection and prevention
    - Crawl result aggregation
    - Network-wide crawl coordination
    """
    
    def __init__(self, 
                 p2p_network: P2PNetworkIntegration,
                 web_engine: WebSocialEngine,
                 node_id: str,
                 max_concurrent_tasks: int = 5):
        
        self.p2p_network = p2p_network
        self.web_engine = web_engine
        self.node_id = node_id
        self.max_concurrent_tasks = max_concurrent_tasks
        
        self.logger = logging.getLogger("P2PWebCrawler")
        
        # Task management
        self.local_tasks: Dict[str, P2PCrawlTask] = {}
        self.distributed_tasks: Dict[str, P2PCrawlTask] = {}
        self.completed_tasks: Dict[str, P2PCrawlTask] = {}
        
        # Coordination state
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.coordination_queue = asyncio.Queue()
        
        # Network state
        self.connected_nodes: Set[str] = set()
        self.node_capabilities: Dict[str, Dict[str, Any]] = {}
        self.task_distribution_history: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.crawl_performance: Dict[str, Dict[str, Any]] = {}
        self.network_utilization: Dict[str, float] = {}
        
        # Background tasks
        self.running = False
        self.task_worker = None
        self.coordination_worker = None
        self.network_monitor = None
        
        self.logger.info(f"P2P Web Crawler initialized for node {node_id}")
    
    async def start(self):
        """Start the P2P web crawler"""
        if self.running:
            return
        
        self.running = True
        
        # Start background workers
        self.task_worker = asyncio.create_task(self._task_worker())
        self.coordination_worker = asyncio.create_task(self._coordination_worker())
        self.network_monitor = asyncio.create_task(self._network_monitor())
        
        # Register with P2P network
        await self._register_with_network()
        
        self.logger.info("P2P Web Crawler started")
    
    async def stop(self):
        """Stop the P2P web crawler"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        if self.task_worker:
            self.task_worker.cancel()
        if self.coordination_worker:
            self.coordination_worker.cancel()
        if self.network_monitor:
            self.network_monitor.cancel()
        
        # Unregister from network
        await self._unregister_from_network()
        
        self.logger.info("P2P Web Crawler stopped")
    
    async def create_distributed_crawl(self, 
                                     urls: List[str],
                                     task_type: CrawlTaskType = CrawlTaskType.WEBSITE_CRAWL,
                                     priority: CrawlPriority = CrawlPriority.MEDIUM,
                                     max_depth: int = 3,
                                     content_filters: Optional[List[str]] = None,
                                     domain_restrictions: Optional[List[str]] = None) -> str:
        """Create a distributed crawl task across the P2P network"""
        
        task_id = f"crawl_{self.node_id}_{int(time.time())}"
        
        task = P2PCrawlTask(
            task_id=task_id,
            task_type=task_type,
            urls=urls,
            priority=priority,
            max_depth=max_depth,
            content_filters=content_filters,
            domain_restrictions=domain_restrictions,
            created_by=self.node_id
        )
        
        # Store locally
        self.local_tasks[task_id] = task
        
        # Distribute to network
        await self._distribute_task(task)
        
        self.logger.info(f"Created distributed crawl task {task_id} with {len(urls)} URLs")
        return task_id
    
    async def _distribute_task(self, task: P2PCrawlTask):
        """Distribute a task across the P2P network"""
        
        # Get available nodes
        available_nodes = await self._get_available_nodes()
        
        if not available_nodes:
            self.logger.warning("No available nodes for task distribution")
            return
        
        # Split URLs among nodes
        url_chunks = self._split_urls_among_nodes(task.urls, available_nodes)
        
        # Create sub-tasks for each node
        for node_id, urls in url_chunks.items():
            sub_task = P2PCrawlTask(
                task_id=f"{task.task_id}_sub_{node_id}",
                task_type=task.task_type,
                urls=urls,
                priority=task.priority,
                max_depth=task.max_depth,
                content_filters=task.content_filters,
                domain_restrictions=task.domain_restrictions,
                created_by=self.node_id
            )
            
            sub_task.assigned_to = node_id
            sub_task.distributed_nodes.add(node_id)
            
            # Send to node
            await self._send_task_to_node(node_id, sub_task)
            
            # Track distribution
            if task.task_id not in self.task_distribution_history:
                self.task_distribution_history[task.task_id] = []
            self.task_distribution_history[task.task_id].append(node_id)
    
    async def _get_available_nodes(self) -> List[str]:
        """Get available nodes for task distribution"""
        available_nodes = []
        
        # Get network users
        users = self.p2p_network.get_available_users()
        
        for user in users:
            if (user.user_id != self.node_id and 
                user.is_available_for_queries() and
                user.capability in [ServerCapability.ADVANCED, ServerCapability.EXPERT, ServerCapability.MASTER]):
                available_nodes.append(user.user_id)
        
        return available_nodes
    
    def _split_urls_among_nodes(self, urls: List[str], nodes: List[str]) -> Dict[str, List[str]]:
        """Split URLs among available nodes"""
        if not nodes:
            return {}
        
        # Simple round-robin distribution
        url_chunks = {node_id: [] for node_id in nodes}
        
        for i, url in enumerate(urls):
            node_id = nodes[i % len(nodes)]
            url_chunks[node_id].append(url)
        
        return url_chunks
    
    async def _send_task_to_node(self, node_id: str, task: P2PCrawlTask):
        """Send a task to a specific node"""
        try:
            # Serialize task
            task_data = task.to_dict()
            task_bytes = pickle.dumps(task_data)
            compressed_data = gzip.compress(task_bytes)
            encoded_data = base64.b64encode(compressed_data).decode()
            
            # Create message
            message = {
                'type': 'crawl_task',
                'task_data': encoded_data,
                'sender': self.node_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # Send via P2P network
            # This would integrate with the P2P network's messaging system
            await self._send_p2p_message(node_id, message)
            
            self.logger.info(f"Sent task {task.task_id} to node {node_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending task to node {node_id}: {e}")
    
    async def _send_p2p_message(self, node_id: str, message: Dict[str, Any]):
        """Send message via P2P network"""
        # This would integrate with the actual P2P network messaging
        # For now, we'll simulate the message sending
        pass
    
    async def _task_worker(self):
        """Background task worker"""
        while self.running:
            try:
                # Process local tasks
                await self._process_local_tasks()
                
                # Process distributed tasks
                await self._process_distributed_tasks()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in task worker: {e}")
    
    async def _process_local_tasks(self):
        """Process local crawl tasks"""
        for task_id, task in list(self.local_tasks.items()):
            if task.status == CrawlStatus.PENDING:
                # Start local crawling
                await self._execute_local_crawl(task)
            elif task.status == CrawlStatus.COMPLETED:
                # Move to completed
                self.completed_tasks[task_id] = task
                del self.local_tasks[task_id]
    
    async def _execute_local_crawl(self, task: P2PCrawlTask):
        """Execute a local crawl task"""
        try:
            task.status = CrawlStatus.IN_PROGRESS
            task.started_at = datetime.now()
            
            # Execute crawling
            for url in task.urls:
                web_page = await self.web_engine.crawl_url(
                    url=url,
                    priority=task.priority,
                    depth=0
                )
                
                if web_page:
                    # Apply content filters
                    if self._passes_content_filters(web_page, task.content_filters):
                        task.results.append(web_page)
                
                # Check for completion
                if len(task.results) >= 100:  # Limit results
                    break
            
            task.status = CrawlStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Share results with network
            await self._share_results_with_network(task)
            
            self.logger.info(f"Completed local crawl task {task.task_id} with {len(task.results)} results")
            
        except Exception as e:
            task.status = CrawlStatus.FAILED
            task.error_message = str(e)
            self.logger.error(f"Error executing local crawl task {task.task_id}: {e}")
    
    async def _process_distributed_tasks(self):
        """Process distributed tasks from other nodes"""
        # This would handle incoming tasks from other nodes
        # For now, we'll simulate processing
        pass
    
    def _passes_content_filters(self, web_page: WebPage, filters: List[str]) -> bool:
        """Check if web page passes content filters"""
        if not filters:
            return True
        
        content_lower = web_page.content.lower()
        title_lower = web_page.title.lower()
        
        for filter_term in filters:
            if (filter_term.lower() in content_lower or 
                filter_term.lower() in title_lower):
                return True
        
        return False
    
    async def _share_results_with_network(self, task: P2PCrawlTask):
        """Share crawl results with the P2P network"""
        try:
            # Serialize results
            results_data = []
            for page in task.results:
                page_data = {
                    'url': page.url,
                    'title': page.title,
                    'content': page.content[:1000],  # Limit content size
                    'content_type': page.content_type.value,
                    'domain': page.domain,
                    'crawl_timestamp': page.crawl_timestamp.isoformat(),
                    'content_hash': page.content_hash,
                    'word_count': page.word_count,
                    'links': page.links[:10]  # Limit links
                }
                results_data.append(page_data)
            
            # Create result message
            message = {
                'type': 'crawl_results',
                'task_id': task.task_id,
                'results': results_data,
                'sender': self.node_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # Broadcast to network
            await self._broadcast_to_network(message)
            
        except Exception as e:
            self.logger.error(f"Error sharing results with network: {e}")
    
    async def _broadcast_to_network(self, message: Dict[str, Any]):
        """Broadcast message to P2P network"""
        # This would integrate with the P2P network's broadcast system
        # For now, we'll simulate broadcasting
        pass
    
    async def _coordination_worker(self):
        """Background coordination worker"""
        while self.running:
            try:
                # Handle incoming messages
                await self._handle_incoming_messages()
                
                # Update network state
                await self._update_network_state()
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in coordination worker: {e}")
    
    async def _handle_incoming_messages(self):
        """Handle incoming P2P messages"""
        # This would handle incoming crawl tasks and results
        # For now, we'll simulate message handling
        pass
    
    async def _update_network_state(self):
        """Update network state and capabilities"""
        # Update connected nodes
        users = self.p2p_network.get_available_users()
        self.connected_nodes = {user.user_id for user in users if user.user_id != self.node_id}
        
        # Update node capabilities
        for user in users:
            if user.user_id != self.node_id:
                self.node_capabilities[user.user_id] = {
                    'capability': user.capability.value,
                    'reputation': user.reputation_score,
                    'research_reputation': user.research_reputation,
                    'last_seen': user.last_seen.isoformat()
                }
    
    async def _network_monitor(self):
        """Monitor network health and performance"""
        while self.running:
            try:
                # Monitor network utilization
                await self._monitor_network_utilization()
                
                # Monitor crawl performance
                await self._monitor_crawl_performance()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in network monitor: {e}")
    
    async def _monitor_network_utilization(self):
        """Monitor network utilization"""
        # Calculate network utilization based on active tasks and connections
        active_tasks = len(self.local_tasks) + len(self.distributed_tasks)
        connected_nodes = len(self.connected_nodes)
        
        utilization = min(1.0, active_tasks / (self.max_concurrent_tasks * max(connected_nodes, 1)))
        self.network_utilization[self.node_id] = utilization
    
    async def _monitor_crawl_performance(self):
        """Monitor crawl performance"""
        for task_id, task in self.local_tasks.items():
            if task.started_at and task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                pages_per_second = len(task.results) / duration if duration > 0 else 0
                
                self.crawl_performance[task_id] = {
                    'duration': duration,
                    'pages_per_second': pages_per_second,
                    'total_pages': len(task.results),
                    'success_rate': 1.0 if task.status == CrawlStatus.COMPLETED else 0.0
                }
    
    async def _register_with_network(self):
        """Register this node with the P2P network"""
        # Register as a web crawler node
        success = self.p2p_network.register_user(
            user_id=self.node_id,
            username=f"WebCrawler_{self.node_id}",
            capability=ServerCapability.EXPERT,
            expertise_domains=['web_crawling', 'content_analysis', 'distributed_computing']
        )
        
        if success:
            self.logger.info(f"Registered as web crawler node: {self.node_id}")
        else:
            self.logger.error(f"Failed to register as web crawler node: {self.node_id}")
    
    async def _unregister_from_network(self):
        """Unregister this node from the P2P network"""
        # Unregister from P2P network
        self.p2p_network.unregister_user(self.node_id)
        self.logger.info(f"Unregistered from P2P network: {self.node_id}")
    
    # Public API methods
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a crawl task"""
        # Check local tasks
        if task_id in self.local_tasks:
            task = self.local_tasks[task_id]
            return {
                'task_id': task.task_id,
                'status': task.status.value,
                'progress': len(task.results),
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'error_message': task.error_message,
                'distributed_nodes': list(task.distributed_nodes)
            }
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return {
                'task_id': task.task_id,
                'status': task.status.value,
                'progress': len(task.results),
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'error_message': task.error_message,
                'distributed_nodes': list(task.distributed_nodes)
            }
        
        return None
    
    async def get_network_statistics(self) -> Dict[str, Any]:
        """Get P2P network statistics"""
        return {
            'connected_nodes': len(self.connected_nodes),
            'local_tasks': len(self.local_tasks),
            'distributed_tasks': len(self.distributed_tasks),
            'completed_tasks': len(self.completed_tasks),
            'network_utilization': self.network_utilization,
            'node_capabilities': self.node_capabilities
        }
    
    async def get_crawl_performance(self) -> Dict[str, Any]:
        """Get crawl performance statistics"""
        if not self.crawl_performance:
            return {}
        
        total_tasks = len(self.crawl_performance)
        successful_tasks = sum(1 for perf in self.crawl_performance.values() 
                             if perf['success_rate'] > 0.8)
        
        avg_pages_per_second = sum(perf['pages_per_second'] for perf in self.crawl_performance.values()) / total_tasks if total_tasks > 0 else 0
        
        return {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0,
            'average_pages_per_second': avg_pages_per_second,
            'task_performance': self.crawl_performance
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a crawl task"""
        if task_id in self.local_tasks:
            task = self.local_tasks[task_id]
            task.status = CrawlStatus.FAILED
            task.error_message = "Task cancelled by user"
            return True
        
        return False
    
    async def get_task_results(self, task_id: str) -> List[Dict[str, Any]]:
        """Get results from a completed task"""
        task = None
        
        if task_id in self.local_tasks:
            task = self.local_tasks[task_id]
        elif task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
        
        if not task:
            return []
        
        results = []
        for page in task.results:
            results.append({
                'url': page.url,
                'title': page.title,
                'content_type': page.content_type.value,
                'domain': page.domain,
                'crawl_timestamp': page.crawl_timestamp.isoformat(),
                'word_count': page.word_count,
                'links': page.links[:5]  # Limit links in results
            })
        
        return results 