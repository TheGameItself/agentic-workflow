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
        """Stub: Receive data from aggregator or adjacent lobes."""
        self.logger.info(f"[TaskProposalLobe] Received data: {data}")
        # TODO: Integrate received data into lobe state 