#!/usr/bin/env python3
"""
Test Script for P2P Research Tracking System

This script demonstrates the comprehensive research tracking capabilities
integrated into the P2P network, including:
- Research source tracking and metadata management
- Quality assessment with multi-dimensional scoring
- Reputation system for researchers and sources
- Collaborative research validation
- Cross-network knowledge sharing
- Research trend analysis and prediction
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mcp.p2p_network_integration import P2PNetworkIntegration, UserStatus, ServerCapability, NetworkRegion
from mcp.p2p_research_tracking import (
    P2PResearchTracking, ResearchSource, ResearcherProfile, 
    ResearchQuality, SourceType, ResearchDomain
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class P2PResearchTrackingDemo:
    """Demonstration of P2P research tracking capabilities"""
    
    def __init__(self):
        self.p2p_network = P2PNetworkIntegration(enable_research_tracking=True)
        self.research_tracking = self.p2p_network.research_tracking
        self.logger = logging.getLogger("P2PResearchTrackingDemo")
        
        # Sample research data
        self.sample_research_sources = [
            {
                'title': 'LL0: Lifelong Learning Starting From Zero',
                'authors': ['Chalmers University', 'AI Research Team'],
                'source_type': 'academic_paper',
                'domain': 'artificial_intelligence',
                'url': 'https://arxiv.org/abs/2502.09696',
                'doi': '10.1000/arxiv.2502.09696',
                'abstract': 'A deep neural-network model for lifelong learning inspired by neuroplasticity.',
                'keywords': ['lifelong learning', 'neuroplasticity', 'neural networks', 'from zero'],
                'peer_reviewed': True,
                'citations': 150,
                'methodology_score': 0.9,
                'reproducibility_score': 0.85,
                'novelty_score': 0.95
            },
            {
                'title': 'Advanced Genetic Evolution in Neural Systems',
                'authors': ['MCP Research Team', 'Genetic AI Lab'],
                'source_type': 'technical_report',
                'domain': 'machine_learning',
                'url': 'https://example.com/genetic-evolution',
                'abstract': 'Implementation of genetic trigger systems with environmental adaptation.',
                'keywords': ['genetic algorithms', 'neural systems', 'environmental adaptation'],
                'peer_reviewed': False,
                'citations': 25,
                'methodology_score': 0.8,
                'reproducibility_score': 0.7,
                'novelty_score': 0.8
            },
            {
                'title': 'P2P Network Intelligence and Distributed Computing',
                'authors': ['Network Research Group', 'Distributed Systems Lab'],
                'source_type': 'conference_paper',
                'domain': 'computer_science',
                'url': 'https://example.com/p2p-intelligence',
                'abstract': 'Distributed computing with global performance optimization.',
                'keywords': ['P2P networks', 'distributed computing', 'performance optimization'],
                'peer_reviewed': True,
                'citations': 75,
                'methodology_score': 0.85,
                'reproducibility_score': 0.8,
                'novelty_score': 0.7
            }
        ]
        
        self.sample_researchers = [
            {
                'user_id': 'researcher_001',
                'username': 'Dr. Alice Johnson',
                'expertise_domains': ['artificial_intelligence', 'machine_learning'],
                'capability': ServerCapability.EXPERT
            },
            {
                'user_id': 'researcher_002',
                'username': 'Prof. Bob Smith',
                'expertise_domains': ['computer_science', 'neuroscience'],
                'capability': ServerCapability.MASTER
            },
            {
                'user_id': 'researcher_003',
                'username': 'Dr. Carol Davis',
                'expertise_domains': ['cognitive_science', 'psychology'],
                'capability': ServerCapability.ADVANCED
            }
        ]
    
    async def run_demo(self):
        """Run the complete demonstration"""
        self.logger.info("🚀 Starting P2P Research Tracking Demo")
        
        try:
            # Start the P2P network
            await self.p2p_network.start()
            
            # Demo 1: Register researchers
            await self.demo_researcher_registration()
            
            # Demo 2: Add research sources
            await self.demo_research_sources()
            
            # Demo 3: Collaborative validation
            await self.demo_collaborative_validation()
            
            # Demo 4: Quality assessment and reputation
            await self.demo_quality_assessment()
            
            # Demo 5: Research trends and analysis
            await self.demo_research_trends()
            
            # Demo 6: Network integration
            await self.demo_network_integration()
            
            # Demo 7: Advanced features
            await self.demo_advanced_features()
            
            self.logger.info("✅ P2P Research Tracking Demo completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Demo failed: {e}")
            raise
        finally:
            await self.p2p_network.stop()
    
    async def demo_researcher_registration(self):
        """Demonstrate researcher registration and profile management"""
        self.logger.info("\n📚 Demo 1: Researcher Registration")
        
        # Register researchers
        for researcher in self.sample_researchers:
            success = self.p2p_network.register_user(
                user_id=researcher['user_id'],
                username=researcher['username'],
                capability=researcher['capability'],
                region=NetworkRegion.GLOBAL,
                expertise_domains=researcher['expertise_domains']
            )
            
            if success:
                self.logger.info(f"✅ Registered researcher: {researcher['username']}")
                
                # Update status to active
                self.p2p_network.update_user_status(
                    user_id=researcher['user_id'],
                    status=UserStatus.ACTIVE
                )
            else:
                self.logger.error(f"❌ Failed to register researcher: {researcher['username']}")
        
        # Display researcher profiles
        self.logger.info("\n👥 Researcher Profiles:")
        for user_id, user in self.p2p_network.users.items():
            self.logger.info(f"  - {user.username}: {user.capability.value}, "
                           f"Research Rep: {user.research_reputation:.2f}, "
                           f"Expertise: {', '.join(user.expertise_domains)}")
    
    async def demo_research_sources(self):
        """Demonstrate adding and managing research sources"""
        self.logger.info("\n📖 Demo 2: Research Sources Management")
        
        # Add research sources
        source_ids = []
        for i, source_data in enumerate(self.sample_research_sources):
            source_id = self.p2p_network.add_research_source(
                title=source_data['title'],
                authors=source_data['authors'],
                source_type=source_data['source_type'],
                domain=source_data['domain'],
                url=source_data['url'],
                doi=source_data.get('doi'),
                user_id=self.sample_researchers[i]['user_id'],
                abstract=source_data['abstract'],
                keywords=source_data['keywords'],
                peer_reviewed=source_data['peer_reviewed'],
                citations=source_data['citations'],
                methodology_score=source_data['methodology_score'],
                reproducibility_score=source_data['reproducibility_score'],
                novelty_score=source_data['novelty_score']
            )
            
            if source_id:
                source_ids.append(source_id)
                self.logger.info(f"✅ Added research source: {source_data['title']}")
            else:
                self.logger.error(f"❌ Failed to add research source: {source_data['title']}")
        
        # Display research sources
        self.logger.info("\n📚 Research Sources:")
        sources = self.p2p_network.get_research_sources(limit=10)
        for source in sources:
            self.logger.info(f"  - {source['title']}")
            self.logger.info(f"    Quality: {source['quality_level']} ({source['quality_score']:.2f})")
            self.logger.info(f"    Domain: {source['domain']}, Type: {source['source_type']}")
            self.logger.info(f"    Validations: {source['validation_count']}, Disputes: {source['dispute_count']}")
    
    async def demo_collaborative_validation(self):
        """Demonstrate collaborative research validation"""
        self.logger.info("\n🤝 Demo 3: Collaborative Validation")
        
        # Get research sources for validation
        sources = self.p2p_network.get_research_sources(limit=5)
        
        if not sources:
            self.logger.warning("No sources available for validation")
            return
        
        # Simulate collaborative validation
        validation_scenarios = [
            {
                'source_id': sources[0]['source_id'],
                'user_id': 'researcher_001',
                'validation_type': 'validate',
                'confidence': 0.9,
                'reasoning': 'Excellent methodology and reproducible results. Peer-reviewed paper with high citation count.'
            },
            {
                'source_id': sources[0]['source_id'],
                'user_id': 'researcher_002',
                'validation_type': 'validate',
                'confidence': 0.85,
                'reasoning': 'Strong theoretical foundation and practical applications. Well-documented implementation.'
            },
            {
                'source_id': sources[1]['source_id'],
                'user_id': 'researcher_003',
                'validation_type': 'dispute',
                'confidence': 0.7,
                'reasoning': 'Methodology needs more rigorous validation. Limited reproducibility testing.'
            }
        ]
        
        for scenario in validation_scenarios:
            success = self.p2p_network.validate_research_source(
                source_id=scenario['source_id'],
                user_id=scenario['user_id'],
                validation_type=scenario['validation_type'],
                confidence=scenario['confidence'],
                reasoning=scenario['reasoning']
            )
            
            if success:
                self.logger.info(f"✅ {scenario['validation_type'].title()}d source by {scenario['user_id']}")
            else:
                self.logger.error(f"❌ Failed to {scenario['validation_type']} source by {scenario['user_id']}")
        
        # Show updated source quality
        self.logger.info("\n📊 Updated Source Quality:")
        updated_sources = self.p2p_network.get_research_sources(limit=5)
        for source in updated_sources:
            self.logger.info(f"  - {source['title']}: {source['quality_level']} ({source['quality_score']:.2f})")
    
    async def demo_quality_assessment(self):
        """Demonstrate quality assessment and reputation tracking"""
        self.logger.info("\n⭐ Demo 4: Quality Assessment and Reputation")
        
        # Record query results with research sources
        query_scenarios = [
            {
                'user_id': 'researcher_001',
                'query_id': 'query_001',
                'success': True,
                'response_time': 2.5,
                'result_quality': 0.9,
                'research_sources': ['source_001', 'source_002']
            },
            {
                'user_id': 'researcher_002',
                'query_id': 'query_002',
                'success': True,
                'response_time': 1.8,
                'result_quality': 0.95,
                'research_sources': ['source_001']
            },
            {
                'user_id': 'researcher_003',
                'query_id': 'query_003',
                'success': False,
                'response_time': 5.2,
                'result_quality': 0.3,
                'research_sources': []
            }
        ]
        
        for scenario in query_scenarios:
            success = self.p2p_network.record_query_result(
                user_id=scenario['user_id'],
                query_id=scenario['query_id'],
                success=scenario['success'],
                response_time=scenario['response_time'],
                result_quality=scenario['result_quality'],
                research_sources=scenario['research_sources']
            )
            
            if success:
                self.logger.info(f"✅ Recorded query result for {scenario['user_id']}")
            else:
                self.logger.error(f"❌ Failed to record query result for {scenario['user_id']}")
        
        # Display updated researcher reputations
        self.logger.info("\n🏆 Updated Researcher Reputations:")
        for user_id, user in self.p2p_network.users.items():
            self.logger.info(f"  - {user.username}:")
            self.logger.info(f"    Overall Reputation: {user.reputation_score:.2f}")
            self.logger.info(f"    Research Reputation: {user.research_reputation:.2f}")
            self.logger.info(f"    Research Contributions: {user.research_contributions}")
            self.logger.info(f"    Is Research Expert: {user.is_research_expert()}")
    
    async def demo_research_trends(self):
        """Demonstrate research trends and analysis"""
        self.logger.info("\n📈 Demo 5: Research Trends and Analysis")
        
        # Get research trends
        trends = self.p2p_network.get_research_trends(days=30)
        
        self.logger.info("\n📊 Research Trends (Last 30 Days):")
        self.logger.info(f"  - Total Sources: {trends.get('total_sources', 0)}")
        self.logger.info(f"  - Average Quality: {trends.get('average_quality', 0):.2f}")
        self.logger.info(f"  - Validation Rate: {trends.get('validation_rate', 0):.1%}")
        
        if 'quality_distribution' in trends:
            self.logger.info("  - Quality Distribution:")
            for quality, count in trends['quality_distribution'].items():
                self.logger.info(f"    {quality}: {count}")
        
        if 'source_type_distribution' in trends:
            self.logger.info("  - Source Type Distribution:")
            for source_type, count in trends['source_type_distribution'].items():
                self.logger.info(f"    {source_type}: {count}")
        
        # Get top researchers
        top_researchers = self.p2p_network.get_top_researchers(limit=5)
        
        self.logger.info("\n🏅 Top Researchers:")
        for i, researcher in enumerate(top_researchers, 1):
            self.logger.info(f"  {i}. {researcher['username']}")
            self.logger.info(f"     Research Reputation: {researcher['research_reputation']:.2f}")
            self.logger.info(f"     Total Sources: {researcher['total_sources']}")
            self.logger.info(f"     Validation Accuracy: {researcher['validation_accuracy']:.2f}")
    
    async def demo_network_integration(self):
        """Demonstrate network integration and metrics"""
        self.logger.info("\n🌐 Demo 6: Network Integration")
        
        # Get network metrics
        metrics = self.p2p_network.get_network_metrics()
        
        self.logger.info("\n📊 Network Metrics:")
        self.logger.info(f"  - Total Users: {metrics.total_users}")
        self.logger.info(f"  - Active Users: {metrics.active_users}")
        self.logger.info(f"  - High Reputation Servers: {metrics.high_reputation_servers}")
        self.logger.info(f"  - Network Health: {metrics.network_health:.2f}")
        self.logger.info(f"  - Query Success Rate: {metrics.query_success_rate:.1%}")
        self.logger.info(f"  - Average Response Time: {metrics.average_response_time:.3f}s")
        
        # Research-specific metrics
        self.logger.info("\n🔬 Research Network Metrics:")
        self.logger.info(f"  - Total Research Sources: {metrics.total_research_sources}")
        self.logger.info(f"  - High Quality Sources: {metrics.high_quality_sources}")
        self.logger.info(f"  - Active Researchers: {metrics.active_researchers}")
        self.logger.info(f"  - Average Research Reputation: {metrics.average_research_reputation:.2f}")
        self.logger.info(f"  - Research Experts: {metrics.research_experts}")
        
        # Get research network stats
        research_stats = self.p2p_network.get_research_network_stats()
        
        self.logger.info("\n📚 Research Network Statistics:")
        for key, value in research_stats.items():
            if isinstance(value, float):
                self.logger.info(f"  - {key}: {value:.2f}")
            else:
                self.logger.info(f"  - {key}: {value}")
    
    async def demo_advanced_features(self):
        """Demonstrate advanced research tracking features"""
        self.logger.info("\n🚀 Demo 7: Advanced Features")
        
        # Domain-specific research sources
        ai_sources = self.p2p_network.get_research_sources(
            domain='artificial_intelligence',
            quality_level='high',
            limit=5
        )
        
        self.logger.info(f"\n🤖 High-Quality AI Research Sources: {len(ai_sources)}")
        for source in ai_sources:
            self.logger.info(f"  - {source['title']} ({source['quality_score']:.2f})")
        
        # Top researchers by domain
        ai_researchers = self.p2p_network.get_top_researchers(
            domain='artificial_intelligence',
            limit=3
        )
        
        self.logger.info(f"\n👨‍🔬 Top AI Researchers: {len(ai_researchers)}")
        for researcher in ai_researchers:
            self.logger.info(f"  - {researcher['username']} ({researcher['research_reputation']:.2f})")
        
        # Research trends by domain
        ai_trends = self.p2p_network.get_research_trends(
            domain='artificial_intelligence',
            days=30
        )
        
        self.logger.info(f"\n📈 AI Research Trends:")
        self.logger.info(f"  - Total Sources: {ai_trends.get('total_sources', 0)}")
        self.logger.info(f"  - Average Quality: {ai_trends.get('average_quality', 0):.2f}")
        
        # Status bar data with research information
        status_data = self.p2p_network.get_status_bar_data()
        
        self.logger.info(f"\n📊 Status Bar Data:")
        self.logger.info(f"  - Total Segments: {len(status_data.get('segments', []))}")
        self.logger.info(f"  - Network Health: {status_data.get('network_health', 0):.2f}")
        
        # Show research experts in the network
        research_experts = [
            user for user in self.p2p_network.users.values()
            if user.is_research_expert()
        ]
        
        self.logger.info(f"\n🏆 Research Experts in Network: {len(research_experts)}")
        for expert in research_experts:
            self.logger.info(f"  - {expert.username}: {expert.research_reputation:.2f} reputation, "
                           f"{expert.research_contributions} contributions")


async def main():
    """Main demo function"""
    demo = P2PResearchTrackingDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main()) 