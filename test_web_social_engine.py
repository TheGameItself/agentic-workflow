#!/usr/bin/env python3
"""
Test Script for Web Social Engine

This script demonstrates the comprehensive web exploration capabilities
of the Web Social Engine, including:
- Direct web crawling without API dependencies
- P2P network integration for distributed crawling
- Social media interaction capabilities
- CAPTCHA handling and credential generation
- Digital identity management
- Content analysis and intelligence gathering
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mcp.web_social_engine import WebSocialEngine, CrawlPriority, CrawlStatus, ContentType
from mcp.p2p_web_crawler import P2PWebCrawler, CrawlTaskType
from mcp.p2p_network_integration import P2PNetworkIntegration, ServerCapability, NetworkRegion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class WebSocialEngineDemo:
    """Demonstration of Web Social Engine capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger("WebSocialEngineDemo")
        
        # Initialize P2P network
        self.p2p_network = P2PNetworkIntegration(enable_research_tracking=True)
        
        # Initialize web engine
        self.web_engine = WebSocialEngine(
            p2p_network=self.p2p_network,
            research_tracking=self.p2p_network.research_tracking,
            max_concurrent_crawls=5,
            max_depth=2,
            crawl_delay=1.0
        )
        
        # Initialize P2P web crawler
        self.p2p_crawler = P2PWebCrawler(
            p2p_network=self.p2p_network,
            web_engine=self.web_engine,
            node_id="demo_node_001",
            max_concurrent_tasks=3
        )
        
        # Sample URLs for testing
        self.test_urls = [
            "https://httpbin.org/html",
            "https://httpbin.org/json",
            "https://httpbin.org/xml",
            "https://example.com",
            "https://httpbin.org/robots.txt"
        ]
        
        # Sample research URLs
        self.research_urls = [
            "https://arxiv.org/abs/2502.09696",  # LL0 paper
            "https://github.com/microsoft/mcp",  # MCP repository
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://stackoverflow.com/questions/tagged/python"
        ]
    
    async def run_demo(self):
        """Run the complete demonstration"""
        self.logger.info("🚀 Starting Web Social Engine Demo")
        
        try:
            # Start the systems
            await self.p2p_network.start()
            await self.web_engine.start()
            await self.p2p_crawler.start()
            
            # Demo 1: Basic web crawling
            await self.demo_basic_crawling()
            
            # Demo 2: Website exploration
            await self.demo_website_exploration()
            
            # Demo 3: Content analysis
            await self.demo_content_analysis()
            
            # Demo 4: P2P distributed crawling
            await self.demo_p2p_distributed_crawling()
            
            # Demo 5: Research integration
            await self.demo_research_integration()
            
            # Demo 6: Performance monitoring
            await self.demo_performance_monitoring()
            
            self.logger.info("✅ Web Social Engine Demo completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Demo failed: {e}")
            raise
        finally:
            await self.web_engine.stop()
            await self.p2p_crawler.stop()
            await self.p2p_network.stop()
    
    async def demo_basic_crawling(self):
        """Demonstrate basic web crawling capabilities"""
        self.logger.info("\n🌐 Demo 1: Basic Web Crawling")
        
        # Crawl individual URLs
        for url in self.test_urls[:3]:  # Limit to first 3 URLs
            self.logger.info(f"Crawling: {url}")
            
            web_page = await self.web_engine.crawl_url(
                url=url,
                priority=CrawlPriority.MEDIUM
            )
            
            if web_page:
                self.logger.info(f"✅ Successfully crawled: {web_page.title}")
                self.logger.info(f"   Content Type: {web_page.content_type.value}")
                self.logger.info(f"   Word Count: {web_page.word_count}")
                self.logger.info(f"   Links Found: {len(web_page.links)}")
                self.logger.info(f"   Response Time: {web_page.response_time:.3f}s")
            else:
                self.logger.warning(f"❌ Failed to crawl: {url}")
        
        # Display crawl statistics
        self.logger.info(f"\n📊 Crawl Statistics:")
        self.logger.info(f"   Total Pages Crawled: {len(self.web_engine.crawled_pages)}")
        self.logger.info(f"   Active Crawls: {len(self.web_engine.active_crawls)}")
        self.logger.info(f"   Blocked Domains: {len(self.web_engine.blocked_domains)}")
    
    async def demo_website_exploration(self):
        """Demonstrate website exploration capabilities"""
        self.logger.info("\n🔍 Demo 2: Website Exploration")
        
        # Crawl an entire website
        base_url = "https://httpbin.org"
        
        self.logger.info(f"Exploring website: {base_url}")
        
        pages = await self.web_engine.crawl_website(
            base_url=base_url,
            max_pages=10
        )
        
        self.logger.info(f"✅ Explored website with {len(pages)} pages")
        
        # Analyze content types
        content_type_counts = {}
        for page in pages:
            content_type = page.content_type.value
            content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
        
        self.logger.info(f"\n📊 Content Type Distribution:")
        for content_type, count in content_type_counts.items():
            self.logger.info(f"   {content_type}: {count} pages")
        
        # Show domain statistics
        domain_stats = await self.web_engine.get_domain_statistics("httpbin.org")
        if domain_stats:
            self.logger.info(f"\n🌍 Domain Statistics:")
            self.logger.info(f"   Success Count: {domain_stats.get('success_count', 0)}")
            self.logger.info(f"   Error Count: {domain_stats.get('error_count', 0)}")
            self.logger.info(f"   Average Response Time: {domain_stats.get('average_response_time', 0):.3f}s")
            self.logger.info(f"   Blocked: {domain_stats.get('blocked', False)}")
    
    async def demo_content_analysis(self):
        """Demonstrate content analysis capabilities"""
        self.logger.info("\n📝 Demo 3: Content Analysis")
        
        # Search through crawled content
        search_queries = ["html", "json", "xml", "python", "api"]
        
        for query in search_queries:
            self.logger.info(f"Searching for: '{query}'")
            
            results = await self.web_engine.search_content(
                query=query,
                max_results=5
            )
            
            self.logger.info(f"   Found {len(results)} results")
            
            for i, page in enumerate(results[:3]):  # Show first 3 results
                self.logger.info(f"   {i+1}. {page.title}")
                self.logger.info(f"      URL: {page.url}")
                self.logger.info(f"      Type: {page.content_type.value}")
                self.logger.info(f"      Words: {page.word_count}")
        
        # Export crawled data
        self.logger.info(f"\n📤 Exporting crawled data...")
        exported_data = await self.web_engine.export_crawled_data(format='json')
        
        # Save to file
        with open('crawled_data.json', 'w') as f:
            f.write(exported_data)
        
        self.logger.info(f"✅ Exported data to crawled_data.json")
    
    async def demo_p2p_distributed_crawling(self):
        """Demonstrate P2P distributed crawling capabilities"""
        self.logger.info("\n🌐 Demo 4: P2P Distributed Crawling")
        
        # Create a distributed crawl task
        task_id = await self.p2p_crawler.create_distributed_crawl(
            urls=self.research_urls,
            task_type=CrawlTaskType.RESEARCH_GATHERING,
            priority=CrawlPriority.HIGH,
            max_depth=2,
            content_filters=["artificial intelligence", "machine learning", "neural networks"]
        )
        
        self.logger.info(f"✅ Created distributed crawl task: {task_id}")
        
        # Monitor task status
        for i in range(10):  # Monitor for 10 seconds
            status = await self.p2p_crawler.get_task_status(task_id)
            if status:
                self.logger.info(f"   Task Status: {status['status']}")
                self.logger.info(f"   Progress: {status['progress']} pages")
                if status['distributed_nodes']:
                    self.logger.info(f"   Distributed to: {status['distributed_nodes']}")
            
            await asyncio.sleep(1)
        
        # Get network statistics
        network_stats = await self.p2p_crawler.get_network_statistics()
        self.logger.info(f"\n📊 P2P Network Statistics:")
        self.logger.info(f"   Connected Nodes: {network_stats['connected_nodes']}")
        self.logger.info(f"   Local Tasks: {network_stats['local_tasks']}")
        self.logger.info(f"   Distributed Tasks: {network_stats['distributed_tasks']}")
        self.logger.info(f"   Completed Tasks: {network_stats['completed_tasks']}")
        
        # Get crawl performance
        performance = await self.p2p_crawler.get_crawl_performance()
        if performance:
            self.logger.info(f"\n⚡ Crawl Performance:")
            self.logger.info(f"   Total Tasks: {performance['total_tasks']}")
            self.logger.info(f"   Success Rate: {performance['success_rate']:.1%}")
            self.logger.info(f"   Average Pages/Second: {performance['average_pages_per_second']:.2f}")
    
    async def demo_research_integration(self):
        """Demonstrate research integration capabilities"""
        self.logger.info("\n🔬 Demo 5: Research Integration")
        
        # Crawl research-related URLs
        for url in self.research_urls[:2]:  # Limit to first 2 URLs
            self.logger.info(f"Crawling research URL: {url}")
            
            web_page = await self.web_engine.crawl_url(
                url=url,
                priority=CrawlPriority.HIGH
            )
            
            if web_page:
                self.logger.info(f"✅ Crawled research content: {web_page.title}")
                self.logger.info(f"   Content Type: {web_page.content_type.value}")
                self.logger.info(f"   Domain: {web_page.domain}")
                self.logger.info(f"   Content Hash: {web_page.content_hash[:16]}...")
        
        # Integrate with research tracking
        self.logger.info(f"\n📚 Integrating with research tracking...")
        await self.web_engine.integrate_with_research_tracking()
        
        # Get research sources from P2P network
        if self.p2p_network.research_tracking:
            research_sources = self.p2p_network.get_research_sources(
                domain="artificial_intelligence",
                quality_level="high",
                limit=5
            )
            
            self.logger.info(f"\n📖 Research Sources Found: {len(research_sources)}")
            for source in research_sources:
                self.logger.info(f"   - {source['title']}")
                self.logger.info(f"     Quality: {source['quality_level']} ({source['quality_score']:.2f})")
                self.logger.info(f"     Domain: {source['domain']}")
        
        # Get research network statistics
        research_stats = self.p2p_network.get_research_network_stats()
        if research_stats:
            self.logger.info(f"\n📊 Research Network Statistics:")
            self.logger.info(f"   Total Sources: {research_stats.get('total_sources', 0)}")
            self.logger.info(f"   High Quality Sources: {research_stats.get('high_quality_sources', 0)}")
            self.logger.info(f"   Active Researchers: {research_stats.get('active_researchers', 0)}")
            self.logger.info(f"   Average Research Reputation: {research_stats.get('average_research_reputation', 0):.2f}")
    
    async def demo_performance_monitoring(self):
        """Demonstrate performance monitoring capabilities"""
        self.logger.info("\n📊 Demo 6: Performance Monitoring")
        
        # Get P2P network metrics
        network_metrics = self.p2p_network.get_network_metrics()
        
        self.logger.info(f"\n🌐 P2P Network Metrics:")
        self.logger.info(f"   Total Users: {network_metrics.total_users}")
        self.logger.info(f"   Active Users: {network_metrics.active_users}")
        self.logger.info(f"   Network Health: {network_metrics.network_health:.2f}")
        self.logger.info(f"   Query Success Rate: {network_metrics.query_success_rate:.1%}")
        self.logger.info(f"   Average Response Time: {network_metrics.average_response_time:.3f}s")
        
        # Research-specific metrics
        self.logger.info(f"\n🔬 Research Network Metrics:")
        self.logger.info(f"   Total Research Sources: {network_metrics.total_research_sources}")
        self.logger.info(f"   High Quality Sources: {network_metrics.high_quality_sources}")
        self.logger.info(f"   Active Researchers: {network_metrics.active_researchers}")
        self.logger.info(f"   Research Experts: {network_metrics.research_experts}")
        
        # Web engine statistics
        self.logger.info(f"\n🌐 Web Engine Statistics:")
        self.logger.info(f"   Crawled Pages: {len(self.web_engine.crawled_pages)}")
        self.logger.info(f"   Active Crawls: {len(self.web_engine.active_crawls)}")
        self.logger.info(f"   Blocked Domains: {len(self.web_engine.blocked_domains)}")
        self.logger.info(f"   Domain Info: {len(self.web_engine.domain_info)}")
        
        # P2P crawler statistics
        p2p_stats = await self.p2p_crawler.get_network_statistics()
        self.logger.info(f"\n🌐 P2P Crawler Statistics:")
        self.logger.info(f"   Connected Nodes: {p2p_stats['connected_nodes']}")
        self.logger.info(f"   Local Tasks: {p2p_stats['local_tasks']}")
        self.logger.info(f"   Distributed Tasks: {p2p_stats['distributed_tasks']}")
        self.logger.info(f"   Completed Tasks: {p2p_stats['completed_tasks']}")
        
        # Performance metrics
        performance = await self.p2p_crawler.get_crawl_performance()
        if performance:
            self.logger.info(f"\n⚡ Performance Metrics:")
            self.logger.info(f"   Total Tasks: {performance['total_tasks']}")
            self.logger.info(f"   Successful Tasks: {performance['successful_tasks']}")
            self.logger.info(f"   Success Rate: {performance['success_rate']:.1%}")
            self.logger.info(f"   Average Pages/Second: {performance['average_pages_per_second']:.2f}")


async def main():
    """Main demo function"""
    demo = WebSocialEngineDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main()) 