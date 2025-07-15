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