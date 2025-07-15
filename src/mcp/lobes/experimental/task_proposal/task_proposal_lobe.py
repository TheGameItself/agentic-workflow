from src.mcp.lobes.experimental.advanced_engram.advanced_engram_engine import WorkingMemory
from src.mcp.lobes.experimental.multi_llm_orchestrator.multi_llm_orchestrator import MultiLLMOrchestrator
from typing import Optional, List, Dict, Any
import logging
from src.mcp.lobes.experimental.vesicle_pool import VesiclePool

class TaskProposalLobe:
    """
    Task Proposal Lobe
    Proactively proposes new tasks, tests, and evaluation steps for the MCP system and its projects.
    Implements research-driven heuristics, LLM-driven task generation, and feedback integration (see idea.txt).
    Each instance has its own working memory for short-term, context-sensitive storage and feedback.
    Integrates with MultiLLMOrchestrator for LLM-based proposals. See idea.txt (metatasks, feedback, research-driven heuristics, line 185).
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.proposed_tasks: List[Dict[str, Any]] = []
        self.proposed_tests: List[Dict[str, Any]] = []
        self.working_memory = WorkingMemory()
        self.logger = logging.getLogger("TaskProposalLobe")
        self.llm_orchestrator = MultiLLMOrchestrator(db_path=db_path)
        self.feedback_log: List[Dict[str, Any]] = []
        self.vesicle_pool = VesiclePool()  # Synaptic vesicle pool model
        self.logger.info("[TaskProposalLobe] VesiclePool initialized: %s", self.vesicle_pool.get_state())

    def propose_task(self, context: Optional[dict] = None, use_llm: bool = True) -> dict:
        """
        Propose a new task based on project state, feedback, and LLM-driven heuristics.
        References: idea.txt (metatasks, feedback, research-driven heuristics, line 185).
        If use_llm is True, use MultiLLMOrchestrator to generate tasks; otherwise, use heuristics and feedback.
        Stores proposals and feedback in working memory for continual improvement.
        """
        self.logger.info("[TaskProposalLobe] Proposing task based on context, feedback, and LLM heuristics.")
        # Gather recent feedback and context
        recent_feedback = self.working_memory.get_recent(5)
        # If LLM-driven generation is enabled, use orchestrator
        if use_llm:
            llm_context = context or {}
            llm_context['recent_feedback'] = recent_feedback
            llm_context['idea_txt_line_185'] = "EVERY SETTING NOT DIRECTLY EDITABLE BY THE USER SHOULD BE DYNAMICALLY ADJUSTING ITSELF WITH ALL METRICS USEFUL AND WITHIN REASON"
            llm_tasks = self.llm_orchestrator.orchestrate([
                {"prompt": "Propose a new research-driven, feedback-aligned task for the MCP project.", "context": llm_context}
            ])
            if llm_tasks and llm_tasks.get('results'):
                task = llm_tasks['results'][0]['task']
                self.working_memory.add(task)
                self.proposed_tasks.append(task)
                return task
        # Heuristic fallback: prioritize unfinished, unoptimized, or feedback-flagged sections
        if context and 'unfinished' in context:
            task = {"title": "Finish unfinished section", "description": "Complete the section: " + context['unfinished']}
            self.working_memory.add(task)
            self.proposed_tasks.append(task)
            return task
        # Use feedback to suggest improvements
        if recent_feedback:
            task = {"title": "Address recent feedback", "description": f"Review and address: {recent_feedback[-1]}"}
            self.working_memory.add(task)
            self.proposed_tasks.append(task)
            return task
        # Default: review for TODOs, stubs, and incomplete features
        task = {"title": "Review project for unfinished sections", "description": "Scan all modules for TODOs, stubs, and incomplete features."}
        self.working_memory.add(task)
        self.proposed_tasks.append(task)
        return task

    def propose_test(self, feature: Optional[str] = None, use_llm: bool = True) -> dict:
        """
        Propose a new test for a feature, including integration and regression tests.
        References: idea.txt (test coverage, feedback loops, research-driven heuristics).
        If use_llm is True, use MultiLLMOrchestrator to generate tests; otherwise, use heuristics and feedback.
        Stores proposals and feedback in working memory for continual improvement.
        """
        self.logger.info(f"[TaskProposalLobe] Proposing test for feature: {feature}")
        recent_feedback = self.working_memory.get_recent(5)
        if use_llm:
            llm_context = {"feature": feature, "recent_feedback": recent_feedback}
            llm_tests = self.llm_orchestrator.orchestrate([
                {"prompt": f"Propose a new test for feature '{feature}'.", "context": llm_context}
            ])
            if llm_tests and llm_tests.get('results'):
                test = llm_tests['results'][0]['task']
                self.working_memory.add(test)
                self.proposed_tests.append(test)
                return test
        # Heuristic fallback: minimal test
        test = {"test_name": "test_feature_completeness", "description": f"Check if feature '{feature or 'unknown'}' is fully implemented and covered by tests."}
        self.working_memory.add(test)
        self.proposed_tests.append(test)
        return test

    def add_feedback(self, feedback: dict):
        """
        Add feedback to the lobe's working memory and feedback log. Used for continual improvement and research-driven adaptation.
        """
        self.feedback_log.append(feedback)
        self.working_memory.add(feedback)
        self.logger.info(f"[TaskProposalLobe] Feedback added: {feedback}") 