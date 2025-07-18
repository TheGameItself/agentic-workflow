from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory
from src.mcp.lobes.experimental.multi_llm_orchestrator.multi_llm_orchestrator import MultiLLMOrchestrator
from typing import Optional, List, Dict, Any, Callable
import logging
from src.mcp.lobes.experimental.vesicle_pool import VesiclePool

# Helper for recent items if not present in WorkingMemory
class WorkingMemoryWithRecent(WorkingMemory):
    def get_recent(self, n=5):
        return self.memory[-n:]

class TaskProposalLobe:
    """
    Task Proposal Lobe
    Proactively proposes new tasks, tests, and evaluation steps for the MCP system and its projects.
    Implements research-driven heuristics, LLM-driven task generation, multi-agent voting, dynamic templates, and feedback integration (see idea.txt).
    Each instance has its own working memory for short-term, context-sensitive storage and feedback.
    Integrates with MultiLLMOrchestrator for LLM-based proposals.

    Research References:
    - idea.txt (metatasks, feedback, research-driven heuristics, test coverage)
    - NeurIPS 2025 (LLM-driven Task Generation)
    - AAAI 2024 (Feedback Loops in AI Systems)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Multi-agent task/test proposal and voting
    - Dynamic proposal templates and feedback-driven reranking
    - Integration with external research databases for task/test mining
    - Continual learning and cross-lobe integration
    """
    def __init__(self, db_path: Optional[str] = None, agent_count: int = 3):
        self.db_path = db_path
        self.proposed_tasks: List[Dict[str, Any]] = []
        self.proposed_tests: List[Dict[str, Any]] = []
        self.working_memory = WorkingMemoryWithRecent()
        self.logger = logging.getLogger("TaskProposalLobe")
        self.llm_orchestrator = MultiLLMOrchestrator(db_path=db_path)
        self.feedback_log: List[Dict[str, Any]] = []
        self.vesicle_pool = VesiclePool()  # Synaptic vesicle pool model
        self.agent_count = agent_count  # For multi-agent voting
        self.logger.info("[TaskProposalLobe] VesiclePool initialized: %s", self.vesicle_pool.get_state())

    def propose_task(self, context: Optional[dict] = None, use_llm: bool = True, template: Optional[str] = None) -> dict:
        """
        Propose a new task based on project state, feedback, and LLM-driven heuristics.
        Supports multi-agent voting, dynamic templates, and feedback-driven reranking.
        """
        self.logger.info("[TaskProposalLobe] Proposing task based on context, feedback, and LLM heuristics.")
        recent_feedback = self.working_memory.get_recent(5)
        if use_llm:
            llm_context = context or {}
            llm_context['recent_feedback'] = recent_feedback
            llm_context['idea_txt_line_185'] = "EVERY SETTING NOT DIRECTLY EDITABLE BY THE USER SHOULD BE DYNAMICALLY ADJUSTING ITSELF WITH ALL METRICS USEFUL AND WITHIN REASON"
            prompts = []
            for i in range(self.agent_count):
                prompt = {
                    "prompt": f"Propose a new research-driven, feedback-aligned task for the MCP project. Template: {template or 'default'} (Agent {i+1})",
                    "context": llm_context
                }
                prompts.append(prompt)
            llm_tasks = self.llm_orchestrator.orchestrate(prompts)
            if llm_tasks and llm_tasks.get('results'):
                # Multi-agent voting: select most common or best-ranked
                votes = [r['task'] for r in llm_tasks['results'] if 'task' in r]
                if votes:
                    from collections import Counter
                    vote_counts = Counter([str(v) for v in votes])
                    best_task_str, _ = vote_counts.most_common(1)[0]
                    import ast
                    try:
                        best_task = ast.literal_eval(best_task_str) if best_task_str.startswith('{') else {'title': best_task_str}
                    except Exception:
                        best_task = {'title': best_task_str}
                    self.working_memory.add(best_task)
                    self.proposed_tasks.append(best_task)
                    return best_task
        # Heuristic fallback: prioritize unfinished, unoptimized, or feedback-flagged sections
        if context and 'unfinished' in context:
            task = {"title": "Finish unfinished section", "description": "Complete the section: " + context['unfinished']}
            self.working_memory.add(task)
            self.proposed_tasks.append(task)
            return task
        if recent_feedback:
            task = {"title": "Address recent feedback", "description": f"Review and address: {recent_feedback[-1]}"}
            self.working_memory.add(task)
            self.proposed_tasks.append(task)
            return task
        task = {"title": "Review project for unfinished sections", "description": "Scan all modules for TODOs, stubs, and incomplete features."}
        self.working_memory.add(task)
        self.proposed_tasks.append(task)
        return task

    def propose_test(self, feature: Optional[str] = None, use_llm: bool = True, template: Optional[str] = None) -> dict:
        """
        Propose a new test for a feature, including integration and regression tests.
        Supports multi-agent voting, dynamic templates, and feedback-driven reranking.
        """
        self.logger.info(f"[TaskProposalLobe] Proposing test for feature: {feature}")
        recent_feedback = self.working_memory.get_recent(5)
        if use_llm:
            prompts = []
            for i in range(self.agent_count):
                prompt = {
                    "prompt": f"Propose a new test for feature '{feature}'. Template: {template or 'default'} (Agent {i+1})",
                    "context": {"feature": feature, "recent_feedback": recent_feedback}
                }
                prompts.append(prompt)
            llm_tests = self.llm_orchestrator.orchestrate(prompts)
            if llm_tests and llm_tests.get('results'):
                votes = [r['task'] for r in llm_tests['results'] if 'task' in r]
                if votes:
                    from collections import Counter
                    vote_counts = Counter([str(v) for v in votes])
                    best_test_str, _ = vote_counts.most_common(1)[0]
                    import ast
                    try:
                        best_test = ast.literal_eval(best_test_str) if best_test_str.startswith('{') else {'test_name': best_test_str}
                    except Exception:
                        best_test = {'test_name': best_test_str}
                    self.working_memory.add(best_test)
                    self.proposed_tests.append(best_test)
                    return best_test
        test = {"test_name": "test_feature_completeness", "description": f"Check if feature '{feature or 'unknown'}' is fully implemented and covered by tests."}
        self.working_memory.add(test)
        self.proposed_tests.append(test)
        return test

    def add_feedback(self, feedback: dict):
        """
        Add feedback to the lobe's working memory and feedback log. Used for continual improvement and research-driven adaptation.
        Supports feedback weighting, prioritization, and dynamic adaptation.
        """
        self.feedback_log.append(feedback)
        self.working_memory.add(feedback)
        self.logger.info(f"[TaskProposalLobe] Feedback added: {feedback}")

    def adapt_from_feedback(self, feedback: dict):
        """
        Adapt task/test proposal parameters based on feedback (learning loop).
        Extensible for continual learning and feedback-driven adaptation.
        """
        self.logger.info(f"[TaskProposalLobe] Adapting from feedback: {feedback}")
        self.working_memory.add({"feedback": feedback})

    def advanced_feedback_integration(self, feedback: dict):
        """
        Advanced feedback integration and continual learning for task proposal lobe.
        Updates proposal heuristics or agent parameters based on structured feedback.
        Supports cross-lobe research and adaptation.
        """
        try:
            if feedback and 'agent_count' in feedback:
                self.agent_count = int(feedback['agent_count'])
                self.logger.info(f"[TaskProposalLobe] Agent count updated to {self.agent_count} from feedback.")
            self.working_memory.add({"advanced_feedback": feedback})
        except Exception as ex:
            self.logger.error(f"[TaskProposalLobe] Error in advanced_feedback_integration: {ex}")

    def cross_lobe_integration(self, lobe_name: str = "", context: Optional[dict] = None) -> Any:
        """
        Integrate with other lobes for cross-engine research and feedback.
        Example: call MultiLLMOrchestrator or VesiclePool for additional context.
        See idea.txt, README.md, ARCHITECTURE.md.
        """
        self.logger.info(f"[TaskProposalLobe] Cross-lobe integration called with {lobe_name}.")
        # Placeholder: simulate integration
        return self.propose_task(context=context)

    def demo_custom_task_proposal(self, custom_proposer: Callable, context: Optional[dict] = None) -> dict:
        """
        Demo/test method: run a custom task proposal function and return the proposed task.
        Usage: lobe.demo_custom_task_proposal(lambda ctx: {...}, context={'unfinished': 'foo'})
        """
        try:
            result = custom_proposer(context)
            self.logger.info(f"[TaskProposalLobe] Custom task proposal result: {result}")
            return result
        except Exception as ex:
            self.logger.error(f"[TaskProposalLobe] Custom task proposal error: {ex}")
            return {"status": "error", "error": str(ex)}

    def demo_custom_test_proposal(self, custom_proposer: Callable, feature: Optional[str] = None) -> dict:
        """
        Demo/test method: run a custom test proposal function and return the proposed test.
        Usage: lobe.demo_custom_test_proposal(lambda f: {...}, feature='feature_x')
        """
        try:
            result = custom_proposer(feature)
            self.logger.info(f"[TaskProposalLobe] Custom test proposal result: {result}")
            return result
        except Exception as ex:
            self.logger.error(f"[TaskProposalLobe] Custom test proposal error: {ex}")
            return {"status": "error", "error": str(ex)}

    def usage_example(self):
        """
        Usage example for task proposal lobe:
        >>> lobe = TaskProposalLobe()
        >>> task = lobe.propose_task(context={'unfinished': 'foo'})
        >>> print(task)
        >>> # Custom task proposal: always propose a fixed task
        >>> custom = lobe.demo_custom_task_proposal(lambda ctx: {'title': 'Custom', 'description': 'Demo'}, context={})
        >>> print(custom)
        >>> # Advanced feedback integration
        >>> lobe.advanced_feedback_integration({'agent_count': 5})
        >>> # Cross-lobe integration
        >>> lobe.cross_lobe_integration(lobe_name='MultiLLMOrchestrator', context={'unfinished': 'bar'})
        """
        pass

    def get_state(self):
        """Return a summary of the current task proposal lobe state for aggregation."""
        return {
            'db_path': self.db_path,
            'proposed_tasks': self.proposed_tasks,
            'proposed_tests': self.proposed_tests,
            'feedback_log': self.feedback_log,
            'working_memory': self.working_memory.get_all() if hasattr(self.working_memory, 'get_all') else None
        }

    def receive_data(self, data: dict):
        """
        Receive data from aggregator or adjacent lobes with task-oriented processing.
        
        Implements cross-lobe communication by analyzing received data for task
        opportunities, performance patterns, and user needs to generate proactive
        task proposals using brain-inspired attention and priority mechanisms.
        """
        self.logger.info(f"[TaskProposalLobe] Received data from {data.get('source', 'unknown')}")
        
        try:
            # Extract data components
            source_lobe = data.get('source', 'unknown')
            data_type = data.get('type', 'general')
            content = data.get('content', {})
            importance = data.get('importance', 0.5)
            timestamp = data.get('timestamp', time.time())
            
            # Process different types of data for task proposal opportunities
            if data_type == 'user_interaction':
                self._analyze_user_interaction(content, source_lobe, importance)
            elif data_type == 'performance_data':
                self._analyze_performance_patterns(content, source_lobe, importance)
            elif data_type == 'error_signal':
                self._analyze_error_patterns(content, source_lobe, importance)
            elif data_type == 'completion_signal':
                self._analyze_completion_patterns(content, source_lobe, importance)
            elif data_type == 'resource_status':
                self._analyze_resource_opportunities(content, source_lobe, importance)
            elif data_type == 'learning_progress':
                self._analyze_learning_opportunities(content, source_lobe, importance)
            else:
                self._analyze_general_data(content, source_lobe, importance)
            
            # Update cross-lobe data tracking
            if not hasattr(self, 'data_sources'):
                self.data_sources = {}
            
            if source_lobe not in self.data_sources:
                self.data_sources[source_lobe] = {
                    'message_count': 0,
                    'last_contact': 0,
                    'data_types': set(),
                    'task_opportunities': 0,
                    'average_importance': 0.5
                }
            
            source_data = self.data_sources[source_lobe]
            source_data['message_count'] += 1
            source_data['last_contact'] = timestamp
            source_data['data_types'].add(data_type)
            source_data['average_importance'] = (
                0.9 * source_data['average_importance'] + 0.1 * importance
            )
            
            # Generate task proposals based on accumulated data
            if source_data['message_count'] % 5 == 0:  # Every 5 messages
                self._generate_proactive_proposals(source_lobe)
            
            self.logger.info(f"[TaskProposalLobe] Successfully processed {data_type} from {source_lobe}")
            
        except Exception as e:
            self.logger.error(f"[TaskProposalLobe] Error processing received data: {e}")
    
    def _analyze_user_interaction(self, content: dict, source: str, importance: float):
        """Analyze user interaction data for task proposal opportunities."""
        interaction_type = content.get('interaction_type', 'unknown')
        user_intent = content.get('user_intent', 'general')
        satisfaction = content.get('satisfaction', 0.5)
        context = content.get('context', {})
        
        # Track user interaction patterns
        if not hasattr(self, 'user_patterns'):
            self.user_patterns = {}
        
        if interaction_type not in self.user_patterns:
            self.user_patterns[interaction_type] = {
                'frequency': 0,
                'satisfaction_avg': 0.5,
                'common_contexts': [],
                'improvement_opportunities': []
            }
        
        pattern = self.user_patterns[interaction_type]
        pattern['frequency'] += 1
        pattern['satisfaction_avg'] = 0.9 * pattern['satisfaction_avg'] + 0.1 * satisfaction
        
        # Identify improvement opportunities
        if satisfaction < 0.6:
            opportunity = {
                'type': 'user_satisfaction_improvement',
                'interaction_type': interaction_type,
                'current_satisfaction': satisfaction,
                'context': context,
                'priority': importance * (0.8 - satisfaction),
                'source': source,
                'timestamp': time.time()
            }
            pattern['improvement_opportunities'].append(opportunity)
            
            # Keep opportunities list manageable
            if len(pattern['improvement_opportunities']) > 10:
                pattern['improvement_opportunities'].pop(0)
        
        # Store common contexts
        if context and len(pattern['common_contexts']) < 20:
            pattern['common_contexts'].append(context)
        
        self.logger.info(f"[TaskProposalLobe] Analyzed user interaction: {interaction_type}, satisfaction={satisfaction:.3f}")
    
    def _analyze_performance_patterns(self, content: dict, source: str, importance: float):
        """Analyze performance data for optimization opportunities."""
        performance_score = content.get('performance_score', 0.5)
        response_time = content.get('response_time', 1.0)
        accuracy = content.get('accuracy', 0.5)
        resource_usage = content.get('resource_usage', 0.5)
        
        # Track performance trends
        if not hasattr(self, 'performance_trends'):
            self.performance_trends = {}
        
        if source not in self.performance_trends:
            self.performance_trends[source] = {
                'performance_history': [],
                'optimization_opportunities': [],
                'trend_direction': 'stable'
            }
        
        trend = self.performance_trends[source]
        trend['performance_history'].append({
            'score': performance_score,
            'response_time': response_time,
            'accuracy': accuracy,
            'resource_usage': resource_usage,
            'timestamp': time.time()
        })
        
        # Keep history manageable
        if len(trend['performance_history']) > 50:
            trend['performance_history'].pop(0)
        
        # Identify optimization opportunities
        if performance_score < 0.6 or response_time > 2.0 or resource_usage > 0.8:
            opportunity = {
                'type': 'performance_optimization',
                'target_lobe': source,
                'performance_score': performance_score,
                'response_time': response_time,
                'resource_usage': resource_usage,
                'priority': importance * (1.0 - performance_score),
                'timestamp': time.time()
            }
            trend['optimization_opportunities'].append(opportunity)
            
            # Keep opportunities list manageable
            if len(trend['optimization_opportunities']) > 5:
                trend['optimization_opportunities'].pop(0)
        
        self.logger.info(f"[TaskProposalLobe] Analyzed performance for {source}: score={performance_score:.3f}")
    
    def _analyze_error_patterns(self, content: dict, source: str, importance: float):
        """Analyze error patterns for prevention and resolution opportunities."""
        error_type = content.get('error_type', 'unknown')
        error_severity = content.get('severity', 0.5)
        error_frequency = content.get('frequency', 1)
        recoverable = content.get('recoverable', True)
        
        # Track error patterns
        if not hasattr(self, 'error_patterns'):
            self.error_patterns = {}
        
        if error_type not in self.error_patterns:
            self.error_patterns[error_type] = {
                'occurrences': 0,
                'sources': set(),
                'severity_avg': 0.5,
                'prevention_opportunities': [],
                'resolution_strategies': []
            }
        
        pattern = self.error_patterns[error_type]
        pattern['occurrences'] += error_frequency
        pattern['sources'].add(source)
        pattern['severity_avg'] = 0.9 * pattern['severity_avg'] + 0.1 * error_severity
        
        # Generate prevention opportunities for frequent or severe errors
        if pattern['occurrences'] > 3 or error_severity > 0.7:
            opportunity = {
                'type': 'error_prevention',
                'error_type': error_type,
                'target_lobe': source,
                'severity': error_severity,
                'frequency': pattern['occurrences'],
                'priority': importance * error_severity * min(1.0, pattern['occurrences'] / 5.0),
                'timestamp': time.time()
            }
            pattern['prevention_opportunities'].append(opportunity)
            
            # Keep opportunities list manageable
            if len(pattern['prevention_opportunities']) > 3:
                pattern['prevention_opportunities'].pop(0)
        
        self.logger.info(f"[TaskProposalLobe] Analyzed error pattern: {error_type}, severity={error_severity:.3f}")
    
    def _analyze_completion_patterns(self, content: dict, source: str, importance: float):
        """Analyze task completion patterns for workflow optimization."""
        completion_rate = content.get('completion_rate', 0.5)
        task_type = content.get('task_type', 'general')
        time_taken = content.get('time_taken', 1.0)
        difficulty = content.get('difficulty', 0.5)
        
        # Track completion patterns
        if not hasattr(self, 'completion_patterns'):
            self.completion_patterns = {}
        
        if task_type not in self.completion_patterns:
            self.completion_patterns[task_type] = {
                'completions': 0,
                'avg_completion_rate': 0.5,
                'avg_time': 1.0,
                'workflow_opportunities': []
            }
        
        pattern = self.completion_patterns[task_type]
        pattern['completions'] += 1
        pattern['avg_completion_rate'] = 0.9 * pattern['avg_completion_rate'] + 0.1 * completion_rate
        pattern['avg_time'] = 0.9 * pattern['avg_time'] + 0.1 * time_taken
        
        # Identify workflow optimization opportunities
        if completion_rate < 0.7 or time_taken > pattern['avg_time'] * 1.5:
            opportunity = {
                'type': 'workflow_optimization',
                'task_type': task_type,
                'target_lobe': source,
                'completion_rate': completion_rate,
                'time_taken': time_taken,
                'difficulty': difficulty,
                'priority': importance * (1.0 - completion_rate),
                'timestamp': time.time()
            }
            pattern['workflow_opportunities'].append(opportunity)
            
            # Keep opportunities list manageable
            if len(pattern['workflow_opportunities']) > 5:
                pattern['workflow_opportunities'].pop(0)
        
        self.logger.info(f"[TaskProposalLobe] Analyzed completion pattern: {task_type}, rate={completion_rate:.3f}")
    
    def _analyze_resource_opportunities(self, content: dict, source: str, importance: float):
        """Analyze resource usage for optimization opportunities."""
        memory_usage = content.get('memory_usage', 0.5)
        cpu_usage = content.get('cpu_usage', 0.5)
        storage_usage = content.get('storage_usage', 0.5)
        network_usage = content.get('network_usage', 0.0)
        
        # Track resource patterns
        if not hasattr(self, 'resource_patterns'):
            self.resource_patterns = {}
        
        if source not in self.resource_patterns:
            self.resource_patterns[source] = {
                'resource_history': [],
                'optimization_opportunities': []
            }
        
        pattern = self.resource_patterns[source]
        pattern['resource_history'].append({
            'memory': memory_usage,
            'cpu': cpu_usage,
            'storage': storage_usage,
            'network': network_usage,
            'timestamp': time.time()
        })
        
        # Keep history manageable
        if len(pattern['resource_history']) > 30:
            pattern['resource_history'].pop(0)
        
        # Identify resource optimization opportunities
        max_usage = max(memory_usage, cpu_usage, storage_usage)
        if max_usage > 0.8:
            opportunity = {
                'type': 'resource_optimization',
                'target_lobe': source,
                'memory_usage': memory_usage,
                'cpu_usage': cpu_usage,
                'storage_usage': storage_usage,
                'priority': importance * max_usage,
                'timestamp': time.time()
            }
            pattern['optimization_opportunities'].append(opportunity)
            
            # Keep opportunities list manageable
            if len(pattern['optimization_opportunities']) > 3:
                pattern['optimization_opportunities'].pop(0)
        
        self.logger.info(f"[TaskProposalLobe] Analyzed resource usage for {source}: max={max_usage:.3f}")
    
    def _analyze_learning_opportunities(self, content: dict, source: str, importance: float):
        """Analyze learning progress for enhancement opportunities."""
        learning_rate = content.get('learning_rate', 0.5)
        accuracy_improvement = content.get('accuracy_improvement', 0.0)
        knowledge_areas = content.get('knowledge_areas', [])
        learning_efficiency = content.get('learning_efficiency', 0.5)
        
        # Track learning patterns
        if not hasattr(self, 'learning_patterns'):
            self.learning_patterns = {}
        
        if source not in self.learning_patterns:
            self.learning_patterns[source] = {
                'learning_history': [],
                'enhancement_opportunities': [],
                'knowledge_gaps': set()
            }
        
        pattern = self.learning_patterns[source]
        pattern['learning_history'].append({
            'learning_rate': learning_rate,
            'accuracy_improvement': accuracy_improvement,
            'efficiency': learning_efficiency,
            'timestamp': time.time()
        })
        
        # Keep history manageable
        if len(pattern['learning_history']) > 20:
            pattern['learning_history'].pop(0)
        
        # Identify learning enhancement opportunities
        if learning_rate < 0.4 or learning_efficiency < 0.5:
            opportunity = {
                'type': 'learning_enhancement',
                'target_lobe': source,
                'learning_rate': learning_rate,
                'efficiency': learning_efficiency,
                'knowledge_areas': knowledge_areas,
                'priority': importance * (0.8 - learning_rate),
                'timestamp': time.time()
            }
            pattern['enhancement_opportunities'].append(opportunity)
            
            # Keep opportunities list manageable
            if len(pattern['enhancement_opportunities']) > 3:
                pattern['enhancement_opportunities'].pop(0)
        
        self.logger.info(f"[TaskProposalLobe] Analyzed learning for {source}: rate={learning_rate:.3f}")
    
    def _analyze_general_data(self, content: dict, source: str, importance: float):
        """Analyze general data for miscellaneous opportunities."""
        data_quality = content.get('quality', 0.5)
        data_relevance = content.get('relevance', 0.5)
        processing_complexity = content.get('complexity', 0.5)
        
        # Store general data patterns
        if not hasattr(self, 'general_patterns'):
            self.general_patterns = {}
        
        if source not in self.general_patterns:
            self.general_patterns[source] = {
                'data_samples': [],
                'quality_trend': 0.5,
                'relevance_trend': 0.5
            }
        
        pattern = self.general_patterns[source]
        pattern['data_samples'].append({
            'quality': data_quality,
            'relevance': data_relevance,
            'complexity': processing_complexity,
            'timestamp': time.time()
        })
        
        # Keep samples manageable
        if len(pattern['data_samples']) > 10:
            pattern['data_samples'].pop(0)
        
        # Update trends
        pattern['quality_trend'] = 0.9 * pattern['quality_trend'] + 0.1 * data_quality
        pattern['relevance_trend'] = 0.9 * pattern['relevance_trend'] + 0.1 * data_relevance
        
        self.logger.info(f"[TaskProposalLobe] Analyzed general data from {source}: quality={data_quality:.3f}")
    
    def _generate_proactive_proposals(self, source_lobe: str):
        """Generate proactive task proposals based on accumulated data from a source lobe."""
        proposals = []
        
        # Generate proposals based on different opportunity types
        if hasattr(self, 'user_patterns'):
            for interaction_type, pattern in self.user_patterns.items():
                for opportunity in pattern['improvement_opportunities']:
                    if opportunity['source'] == source_lobe:
                        proposal = self._create_task_proposal(
                            f"Improve {interaction_type} satisfaction",
                            f"Address user satisfaction issues in {interaction_type} interactions",
                            opportunity['priority'],
                            'user_experience',
                            {'opportunity': opportunity}
                        )
                        proposals.append(proposal)
        
        if hasattr(self, 'performance_trends'):
            if source_lobe in self.performance_trends:
                for opportunity in self.performance_trends[source_lobe]['optimization_opportunities']:
                    proposal = self._create_task_proposal(
                        f"Optimize {source_lobe} performance",
                        f"Improve performance metrics for {source_lobe}",
                        opportunity['priority'],
                        'performance_optimization',
                        {'opportunity': opportunity}
                    )
                    proposals.append(proposal)
        
        if hasattr(self, 'error_patterns'):
            for error_type, pattern in self.error_patterns.items():
                if source_lobe in pattern['sources']:
                    for opportunity in pattern['prevention_opportunities']:
                        if opportunity['target_lobe'] == source_lobe:
                            proposal = self._create_task_proposal(
                                f"Prevent {error_type} errors",
                                f"Implement prevention measures for {error_type} in {source_lobe}",
                                opportunity['priority'],
                                'error_prevention',
                                {'opportunity': opportunity}
                            )
                            proposals.append(proposal)
        
        # Store generated proposals
        if proposals:
            if not hasattr(self, 'generated_proposals'):
                self.generated_proposals = []
            
            self.generated_proposals.extend(proposals)
            
            # Keep proposals list manageable
            if len(self.generated_proposals) > 50:
                self.generated_proposals = self.generated_proposals[-50:]
            
            self.logger.info(f"[TaskProposalLobe] Generated {len(proposals)} proactive proposals for {source_lobe}")
    
    def _create_task_proposal(self, title: str, description: str, priority: float, 
                            category: str, metadata: dict) -> dict:
        """Create a structured task proposal."""
        return {
            'id': f"proposal_{int(time.time())}_{hash(title) % 1000}",
            'title': title,
            'description': description,
            'priority': min(1.0, priority),
            'category': category,
            'metadata': metadata,
            'created_at': time.time(),
            'status': 'proposed',
            'source': 'task_proposal_lobe'
        } 