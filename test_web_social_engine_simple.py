#!/usr/bin/env python3
"""
Simplified Test Script for Web Social Engine

This script demonstrates the web social engine capabilities without requiring
external dependencies. It shows the architecture and core functionality.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock imports for demonstration
class MockAiohttpSession:
    """Mock aiohttp session for demonstration"""
    async def get(self, url, headers=None, timeout=None):
        return MockResponse(url)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class MockResponse:
    """Mock HTTP response for demonstration"""
    def __init__(self, url):
        self.url = url
        self.status = 200
    
    async def text(self):
        return f"<html><head><title>Mock Page for {self.url}</title></head><body><h1>Mock Content</h1><p>This is mock content for {self.url}</p><a href='https://example.com'>Example Link</a></body></html>"

# Mock aiohttp module
class MockAiohttp:
    @staticmethod
    def ClientSession():
        return MockAiohttpSession()

# Patch the import
import sys
sys.modules['aiohttp'] = MockAiohttp()

from mcp.web_social_engine import WebSocialEngine, CrawlPriority, CrawlStatus, ContentType
from mcp.p2p_web_crawler import P2PWebCrawler, CrawlTaskType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class WebSocialEngineSimpleDemo:
    """Simplified demonstration of Web Social Engine capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger("WebSocialEngineSimpleDemo")
        
        # Initialize web engine with mock P2P network
        self.web_engine = WebSocialEngine(
            p2p_network=None,  # Mock for this demo
            research_tracking=None,  # Mock for this demo
            max_concurrent_crawls=3,
            max_depth=2,
            crawl_delay=0.1  # Faster for demo
        )
        
        # Sample URLs for testing
        self.test_urls = [
            "https://httpbin.org/html",
            "https://httpbin.org/json",
            "https://httpbin.org/xml",
            "https://example.com",
            "https://httpbin.org/robots.txt"
        ]
    
    async def run_demo(self):
        """Run the simplified demonstration"""
        self.logger.info("🚀 Starting Web Social Engine Simple Demo")
        
        try:
            # Start the web engine
            await self.web_engine.start()
            
            # Demo 1: Basic web crawling
            await self.demo_basic_crawling()
            
            # Demo 2: Content analysis
            await self.demo_content_analysis()
            
            # Demo 3: Performance monitoring
            await self.demo_performance_monitoring()
            
            # Demo 4: Export capabilities
            await self.demo_export_capabilities()
            
            self.logger.info("✅ Web Social Engine Simple Demo completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Demo failed: {e}")
            raise
        finally:
            await self.web_engine.stop()
    
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
                self.logger.info(f"   Content Hash: {web_page.content_hash[:16]}...")
            else:
                self.logger.warning(f"❌ Failed to crawl: {url}")
        
        # Display crawl statistics
        self.logger.info(f"\n📊 Crawl Statistics:")
        self.logger.info(f"   Total Pages Crawled: {len(self.web_engine.crawled_pages)}")
        self.logger.info(f"   Active Crawls: {len(self.web_engine.active_crawls)}")
        self.logger.info(f"   Blocked Domains: {len(self.web_engine.blocked_domains)}")
    
    async def demo_content_analysis(self):
        """Demonstrate content analysis capabilities"""
        self.logger.info("\n📝 Demo 2: Content Analysis")
        
        # Analyze content types
        content_type_counts = {}
        for page in self.web_engine.crawled_pages.values():
            content_type = page.content_type.value
            content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
        
        self.logger.info(f"\n📊 Content Type Distribution:")
        for content_type, count in content_type_counts.items():
            self.logger.info(f"   {content_type}: {count} pages")
        
        # Search through crawled content
        search_queries = ["mock", "content", "example", "html"]
        
        for query in search_queries:
            self.logger.info(f"\n🔍 Searching for: '{query}'")
            
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
        
        # Show domain statistics
        domain_stats = await self.web_engine.get_domain_statistics("httpbin.org")
        if domain_stats:
            self.logger.info(f"\n🌍 Domain Statistics:")
            self.logger.info(f"   Success Count: {domain_stats.get('success_count', 0)}")
            self.logger.info(f"   Error Count: {domain_stats.get('error_count', 0)}")
            self.logger.info(f"   Average Response Time: {domain_stats.get('average_response_time', 0):.3f}s")
            self.logger.info(f"   Blocked: {domain_stats.get('blocked', False)}")
    
    async def demo_performance_monitoring(self):
        """Demonstrate performance monitoring capabilities"""
        self.logger.info("\n📊 Demo 3: Performance Monitoring")
        
        # Web engine statistics
        self.logger.info(f"\n🌐 Web Engine Statistics:")
        self.logger.info(f"   Crawled Pages: {len(self.web_engine.crawled_pages)}")
        self.logger.info(f"   Active Crawls: {len(self.web_engine.active_crawls)}")
        self.logger.info(f"   Blocked Domains: {len(self.web_engine.blocked_domains)}")
        self.logger.info(f"   Domain Info: {len(self.web_engine.domain_info)}")
        
        # Performance metrics
        total_response_time = 0
        total_pages = len(self.web_engine.crawled_pages)
        
        for page in self.web_engine.crawled_pages.values():
            total_response_time += page.response_time
        
        if total_pages > 0:
            avg_response_time = total_response_time / total_pages
            self.logger.info(f"   Average Response Time: {avg_response_time:.3f}s")
            self.logger.info(f"   Total Response Time: {total_response_time:.3f}s")
        
        # Domain performance
        self.logger.info(f"\n🌍 Domain Performance:")
        for domain, info in self.web_engine.domain_info.items():
            self.logger.info(f"   {domain}:")
            self.logger.info(f"     Success Count: {info.success_count}")
            self.logger.info(f"     Error Count: {info.error_count}")
            self.logger.info(f"     Average Response Time: {info.average_response_time:.3f}s")
            self.logger.info(f"     Blocked: {info.blocked}")
    
    async def demo_export_capabilities(self):
        """Demonstrate export capabilities"""
        self.logger.info("\n📤 Demo 4: Export Capabilities")
        
        # Export crawled data
        self.logger.info(f"📤 Exporting crawled data...")
        exported_data = await self.web_engine.export_crawled_data(format='json')
        
        # Save to file
        with open('crawled_data_simple.json', 'w') as f:
            f.write(exported_data)
        
        self.logger.info(f"✅ Exported data to crawled_data_simple.json")
        
        # Show export statistics
        import json
        data = json.loads(exported_data)
        self.logger.info(f"\n📊 Export Statistics:")
        self.logger.info(f"   Total Pages: {data['statistics']['total_pages']}")
        self.logger.info(f"   Domains: {data['statistics']['domains']}")
        self.logger.info(f"   Blocked Domains: {data['statistics']['blocked_domains']}")
        
        # Show sample exported data
        if data['pages']:
            self.logger.info(f"\n📄 Sample Exported Data:")
            sample_page = data['pages'][0]
            self.logger.info(f"   Title: {sample_page['title']}")
            self.logger.info(f"   URL: {sample_page['url']}")
            self.logger.info(f"   Content Type: {sample_page['content_type']}")
            self.logger.info(f"   Word Count: {sample_page['word_count']}")
            self.logger.info(f"   Links: {len(sample_page['links'])}")
    
    def demonstrate_architecture(self):
        """Demonstrate the architecture components"""
        self.logger.info("\n🏗️ Architecture Demonstration")
        
        self.logger.info("\n📋 Core Components:")
        self.logger.info("   1. WebSocialEngine - Main web crawling engine")
        self.logger.info("   2. CAPTCHASolver - Intelligent CAPTCHA solving")
        self.logger.info("   3. DigitalIdentityManager - Identity rotation and management")
        self.logger.info("   4. P2PWebCrawler - Distributed crawling coordinator")
        
        self.logger.info("\n🔧 Key Features:")
        self.logger.info("   • Direct web crawling without API dependencies")
        self.logger.info("   • P2P network integration for distributed crawling")
        self.logger.info("   • Content analysis and classification")
        self.logger.info("   • CAPTCHA handling and credential generation")
        self.logger.info("   • Digital identity management")
        self.logger.info("   • Privacy-preserving crawling")
        self.logger.info("   • Research integration")
        
        self.logger.info("\n🌐 Supported Content Types:")
        for content_type in ContentType:
            self.logger.info(f"   • {content_type.value}")
        
        self.logger.info("\n⚡ Crawl Priority Levels:")
        for priority in CrawlPriority:
            self.logger.info(f"   • {priority.name}: {priority.value}")
        
        self.logger.info("\n📊 Crawl Status Types:")
        for status in CrawlStatus:
            self.logger.info(f"   • {status.value}")


async def main():
    """Main demo function"""
    demo = WebSocialEngineSimpleDemo()
    
    # Demonstrate architecture first
    demo.demonstrate_architecture()
    
    # Run the demo
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main()) 