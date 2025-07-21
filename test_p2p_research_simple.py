#!/usr/bin/env python3
"""
Simplified Test Script for P2P Research Tracking System

This script demonstrates the research tracking capabilities without external dependencies.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class SimpleP2PResearchDemo:
    """Simplified demonstration of P2P research tracking capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger("SimpleP2PResearchDemo")
        
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
                'capability': 'expert'
            },
            {
                'user_id': 'researcher_002',
                'username': 'Prof. Bob Smith',
                'expertise_domains': ['computer_science', 'neuroscience'],
                'capability': 'master'
            },
            {
                'user_id': 'researcher_003',
                'username': 'Dr. Carol Davis',
                'expertise_domains': ['cognitive_science', 'psychology'],
                'capability': 'advanced'
            }
        ]
    
    def run_demo(self):
        """Run the simplified demonstration"""
        self.logger.info("🚀 Starting Simplified P2P Research Tracking Demo")
        
        try:
            # Demo 1: Research source tracking
            self.demo_research_source_tracking()
            
            # Demo 2: Quality assessment
            self.demo_quality_assessment()
            
            # Demo 3: Reputation system
            self.demo_reputation_system()
            
            # Demo 4: Collaborative validation
            self.demo_collaborative_validation()
            
            # Demo 5: Research trends
            self.demo_research_trends()
            
            self.logger.info("✅ Simplified P2P Research Tracking Demo completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Demo failed: {e}")
            raise
    
    def demo_research_source_tracking(self):
        """Demonstrate research source tracking capabilities"""
        self.logger.info("\n📚 Demo 1: Research Source Tracking")
        
        # Simulate research source database
        research_sources = {}
        source_id_counter = 1
        
        for source_data in self.sample_research_sources:
            source_id = f"source_{source_id_counter:03d}"
            
            # Calculate quality score
            quality_score = self.calculate_quality_score(source_data)
            quality_level = self.get_quality_level(quality_score)
            
            research_sources[source_id] = {
                'source_id': source_id,
                'title': source_data['title'],
                'authors': source_data['authors'],
                'source_type': source_data['source_type'],
                'domain': source_data['domain'],
                'url': source_data['url'],
                'doi': source_data.get('doi'),
                'abstract': source_data['abstract'],
                'keywords': source_data['keywords'],
                'peer_reviewed': source_data['peer_reviewed'],
                'citations': source_data['citations'],
                'methodology_score': source_data['methodology_score'],
                'reproducibility_score': source_data['reproducibility_score'],
                'novelty_score': source_data['novelty_score'],
                'quality_score': quality_score,
                'quality_level': quality_level,
                'network_reputation': 0.5,
                'validation_count': 0,
                'dispute_count': 0,
                'created_by': None,
                'validated_by': [],
                'disputed_by': [],
                'last_updated': datetime.now().isoformat()
            }
            
            source_id_counter += 1
            self.logger.info(f"✅ Added research source: {source_data['title']}")
        
        # Display research sources
        self.logger.info("\n📖 Research Sources Database:")
        for source_id, source in research_sources.items():
            self.logger.info(f"  - {source['title']}")
            self.logger.info(f"    ID: {source_id}")
            self.logger.info(f"    Quality: {source['quality_level']} ({source['quality_score']:.2f})")
            self.logger.info(f"    Domain: {source['domain']}, Type: {source['source_type']}")
            self.logger.info(f"    Citations: {source['citations']}, Peer Reviewed: {source['peer_reviewed']}")
            self.logger.info(f"    Authors: {', '.join(source['authors'])}")
        
        return research_sources
    
    def demo_quality_assessment(self):
        """Demonstrate quality assessment system"""
        self.logger.info("\n⭐ Demo 2: Quality Assessment System")
        
        # Simulate quality assessment for each source
        for source_data in self.sample_research_sources:
            quality_score = self.calculate_quality_score(source_data)
            quality_level = self.get_quality_level(quality_score)
            
            self.logger.info(f"\n📊 Quality Assessment for: {source_data['title']}")
            self.logger.info(f"  - Source Type Weight: {self.get_source_type_weight(source_data['source_type']):.2f}")
            self.logger.info(f"  - Citation Impact: {self.calculate_citation_score(source_data['citations']):.2f}")
            self.logger.info(f"  - Methodology Score: {source_data['methodology_score']:.2f}")
            self.logger.info(f"  - Reproducibility Score: {source_data['reproducibility_score']:.2f}")
            self.logger.info(f"  - Novelty Score: {source_data['novelty_score']:.2f}")
            self.logger.info(f"  - Peer Review Bonus: {0.2 if source_data['peer_reviewed'] else 0.0}")
            self.logger.info(f"  - Final Quality Score: {quality_score:.2f}")
            self.logger.info(f"  - Quality Level: {quality_level}")
    
    def demo_reputation_system(self):
        """Demonstrate reputation system for researchers"""
        self.logger.info("\n🏆 Demo 3: Researcher Reputation System")
        
        # Simulate researcher profiles
        researcher_profiles = {}
        
        for researcher in self.sample_researchers:
            researcher_id = researcher['user_id']
            
            # Simulate research activity
            research_contributions = len([s for s in self.sample_research_sources 
                                        if any(domain in researcher['expertise_domains'] 
                                              for domain in [s['domain']])])
            
            # Calculate reputation based on contributions and expertise
            base_reputation = 0.5
            contribution_bonus = min(0.3, research_contributions * 0.1)
            expertise_bonus = 0.1 if researcher['capability'] in ['expert', 'master'] else 0.0
            
            research_reputation = min(1.0, base_reputation + contribution_bonus + expertise_bonus)
            
            researcher_profiles[researcher_id] = {
                'researcher_id': researcher_id,
                'username': researcher['username'],
                'user_id': researcher_id,
                'expertise_domains': researcher['expertise_domains'],
                'capability': researcher['capability'],
                'research_reputation': research_reputation,
                'research_contributions': research_contributions,
                'total_sources': research_contributions,
                'validated_sources': max(0, research_contributions - 1),
                'disputed_sources': 1 if research_contributions > 0 else 0,
                'validation_accuracy': 0.8,
                'dispute_accuracy': 0.7,
                'network_contributions': research_contributions * 2,
                'activity_streak': 7,
                'last_activity': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Created researcher profile: {researcher['username']}")
        
        # Display researcher profiles
        self.logger.info("\n👥 Researcher Profiles:")
        for researcher_id, profile in researcher_profiles.items():
            self.logger.info(f"  - {profile['username']}")
            self.logger.info(f"    Research Reputation: {profile['research_reputation']:.2f}")
            self.logger.info(f"    Expertise: {', '.join(profile['expertise_domains'])}")
            self.logger.info(f"    Capability: {profile['capability']}")
            self.logger.info(f"    Research Contributions: {profile['research_contributions']}")
            self.logger.info(f"    Validation Accuracy: {profile['validation_accuracy']:.2f}")
            self.logger.info(f"    Activity Streak: {profile['activity_streak']} days")
        
        return researcher_profiles
    
    def demo_collaborative_validation(self):
        """Demonstrate collaborative validation system"""
        self.logger.info("\n🤝 Demo 4: Collaborative Validation System")
        
        # Simulate validation scenarios
        validation_scenarios = [
            {
                'source_title': 'LL0: Lifelong Learning Starting From Zero',
                'researcher': 'Dr. Alice Johnson',
                'validation_type': 'validate',
                'confidence': 0.9,
                'reasoning': 'Excellent methodology and reproducible results. Peer-reviewed paper with high citation count.'
            },
            {
                'source_title': 'LL0: Lifelong Learning Starting From Zero',
                'researcher': 'Prof. Bob Smith',
                'validation_type': 'validate',
                'confidence': 0.85,
                'reasoning': 'Strong theoretical foundation and practical applications. Well-documented implementation.'
            },
            {
                'source_title': 'Advanced Genetic Evolution in Neural Systems',
                'researcher': 'Dr. Carol Davis',
                'validation_type': 'dispute',
                'confidence': 0.7,
                'reasoning': 'Methodology needs more rigorous validation. Limited reproducibility testing.'
            }
        ]
        
        # Track validations
        source_validations = {}
        
        for scenario in validation_scenarios:
            source_title = scenario['source_title']
            if source_title not in source_validations:
                source_validations[source_title] = {
                    'validations': 0,
                    'disputes': 0,
                    'validators': [],
                    'disputers': []
                }
            
            if scenario['validation_type'] == 'validate':
                source_validations[source_title]['validations'] += 1
                source_validations[source_title]['validators'].append(scenario['researcher'])
            else:
                source_validations[source_title]['disputes'] += 1
                source_validations[source_title]['disputers'].append(scenario['researcher'])
            
            self.logger.info(f"✅ {scenario['researcher']} {scenario['validation_type']}d: {source_title}")
            self.logger.info(f"   Confidence: {scenario['confidence']:.2f}")
            self.logger.info(f"   Reasoning: {scenario['reasoning']}")
        
        # Show validation results
        self.logger.info("\n📊 Validation Results:")
        for source_title, validation_data in source_validations.items():
            total_checks = validation_data['validations'] + validation_data['disputes']
            validation_rate = validation_data['validations'] / total_checks if total_checks > 0 else 0.0
            
            self.logger.info(f"  - {source_title}")
            self.logger.info(f"    Validations: {validation_data['validations']}")
            self.logger.info(f"    Disputes: {validation_data['disputes']}")
            self.logger.info(f"    Validation Rate: {validation_rate:.1%}")
            self.logger.info(f"    Validators: {', '.join(validation_data['validators'])}")
            if validation_data['disputers']:
                self.logger.info(f"    Disputers: {', '.join(validation_data['disputers'])}")
    
    def demo_research_trends(self):
        """Demonstrate research trends analysis"""
        self.logger.info("\n📈 Demo 5: Research Trends Analysis")
        
        # Analyze domain distribution
        domain_counts = {}
        quality_distribution = {}
        source_type_distribution = {}
        
        for source in self.sample_research_sources:
            # Domain distribution
            domain = source['domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Quality distribution
            quality_score = self.calculate_quality_score(source)
            quality_level = self.get_quality_level(quality_score)
            quality_distribution[quality_level] = quality_distribution.get(quality_level, 0) + 1
            
            # Source type distribution
            source_type = source['source_type']
            source_type_distribution[source_type] = source_type_distribution.get(source_type, 0) + 1
        
        # Calculate average quality
        quality_scores = [self.calculate_quality_score(s) for s in self.sample_research_sources]
        average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        self.logger.info("\n📊 Research Trends Analysis:")
        self.logger.info(f"  - Total Sources: {len(self.sample_research_sources)}")
        self.logger.info(f"  - Average Quality Score: {average_quality:.2f}")
        
        self.logger.info("\n🌍 Domain Distribution:")
        for domain, count in domain_counts.items():
            self.logger.info(f"  - {domain}: {count} sources")
        
        self.logger.info("\n⭐ Quality Distribution:")
        for quality, count in quality_distribution.items():
            self.logger.info(f"  - {quality}: {count} sources")
        
        self.logger.info("\n📚 Source Type Distribution:")
        for source_type, count in source_type_distribution.items():
            self.logger.info(f"  - {source_type}: {count} sources")
        
        # Top authors analysis
        author_counts = {}
        for source in self.sample_research_sources:
            for author in source['authors']:
                author_counts[author] = author_counts.get(author, 0) + 1
        
        self.logger.info("\n👨‍🔬 Top Authors:")
        sorted_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
        for author, count in sorted_authors[:5]:
            self.logger.info(f"  - {author}: {count} sources")
    
    def calculate_quality_score(self, source_data: Dict[str, Any]) -> float:
        """Calculate quality score for a research source"""
        # Source type weight
        type_weight = self.get_source_type_weight(source_data['source_type'])
        
        # Citation impact
        citation_score = self.calculate_citation_score(source_data['citations'])
        
        # Methodology and reproducibility
        methodology_score = source_data['methodology_score']
        reproducibility_score = source_data['reproducibility_score']
        novelty_score = source_data['novelty_score']
        
        # Peer review bonus
        peer_review_bonus = 0.2 if source_data['peer_reviewed'] else 0.0
        
        # Calculate weighted average
        scores = [type_weight, citation_score, methodology_score, reproducibility_score, novelty_score, peer_review_bonus]
        weights = [0.2, 0.15, 0.2, 0.15, 0.1, 0.2]
        
        final_score = sum(s * w for s, w in zip(scores, weights))
        return max(0.0, min(1.0, final_score))
    
    def get_quality_level(self, quality_score: float) -> str:
        """Get quality level based on score"""
        if quality_score >= 0.9:
            return "peer_reviewed"
        elif quality_score >= 0.8:
            return "excellent"
        elif quality_score >= 0.7:
            return "high"
        elif quality_score >= 0.5:
            return "medium"
        elif quality_score >= 0.3:
            return "low"
        else:
            return "unverified"
    
    def get_source_type_weight(self, source_type: str) -> float:
        """Get weight for source type"""
        weights = {
            'academic_paper': 0.9,
            'journal_article': 0.85,
            'conference_paper': 0.8,
            'technical_report': 0.7,
            'preprint': 0.6,
            'book': 0.75,
            'thesis': 0.7,
            'patent': 0.6,
            'dataset': 0.8,
            'code_repository': 0.7,
            'blog_post': 0.4,
            'news_article': 0.3,
            'website': 0.3,
            'video': 0.5,
            'podcast': 0.4,
            'social_media': 0.2
        }
        return weights.get(source_type, 0.5)
    
    def calculate_citation_score(self, citations: int) -> float:
        """Calculate citation impact score"""
        if citations <= 0:
            return 0.0
        # Use log scale to normalize citations
        return min(1.0, (citations ** 0.5) / 10.0)


def main():
    """Main demo function"""
    demo = SimpleP2PResearchDemo()
    demo.run_demo()


if __name__ == "__main__":
    main() 