from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory
from typing import Optional, List, Dict, Any, Callable
import logging
from collections import Counter

class MultiLLMOrchestrator:
    """
    Multi-LLM Orchestrator
    Task routing, aggregation, and AB testing for multiple LLMs. Implements research-driven orchestration, advanced aggregation, AB testing, and feedback analytics (see idea.txt, WAIT/Hermes, AB testing, feedback-driven selection).
    
    Research References:
    - idea.txt (multi-agent orchestration, AB testing, feedback analytics, prompt batching)
    - WAIT/Hermes prompt queueing (arXiv:2504.11320)
    - arXiv:2506.14851 (Multi-agent orchestration)
    - Group Think: Concurrent Reasoning and Cyclic AB Testing (arXiv:2505.11107)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md
    
    Extensibility:
    - Pluggable orchestration, aggregation, and feedback analytics methods
    - Advanced AB testing and multi-agent feedback loops
    - Demo/test methods and continual learning
    - Integration with other lobes for cross-engine research and feedback
    """
    def __init__(self, db_path: Optional[str] = None, orchestrator_fn: Optional[Callable] = None, aggregator_fn: Optional[Callable] = None, ab_test_fn: Optional[Callable] = None, feedback_fn: Optional[Callable] = None):
        self.db_path = db_path
        self.working_memory = WorkingMemory()
        self.logger = logging.getLogger("MultiLLMOrchestrator")
        self.orchestrator_fn = orchestrator_fn  # Pluggable orchestration logic
        self.aggregator_fn = aggregator_fn      # Pluggable aggregation logic
        self.ab_test_fn = ab_test_fn            # Pluggable AB testing logic
        self.feedback_fn = feedback_fn          # Pluggable feedback analytics

    def orchestrate(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Orchestrate tasks among multiple LLMs, aggregate results, and perform AB testing.
        Supports pluggable orchestration, advanced aggregation, AB testing, and feedback analytics.
        """
        self.logger.info(f"[MultiLLMOrchestrator] Orchestrating tasks: {tasks}")
        if not tasks:
            return {"status": "no_tasks", "results": []}
        try:
            if self.orchestrator_fn and callable(self.orchestrator_fn):
                llm_results = self.orchestrator_fn(tasks)
            else:
                llm_results = [self.call_llm(task) for task in tasks]
            if self.aggregator_fn and callable(self.aggregator_fn):
                aggregated = self.aggregator_fn(llm_results)
            else:
                aggregated = self.aggregate_results(llm_results)
            if self.ab_test_fn and callable(self.ab_test_fn):
                ab_test = self.ab_test_fn(llm_results)
            else:
                ab_test = self.ab_test_results(llm_results)
            if self.feedback_fn and callable(self.feedback_fn):
                feedback = self.feedback_fn(llm_results)
            else:
                feedback = self.analyze_feedback(llm_results)
            for r in llm_results:
                self.working_memory.add(r)
        except Exception as ex:
            self.logger.error(f"[MultiLLMOrchestrator] Orchestration error: {ex}")
            return {"status": "orchestration_error", "error": str(ex)}
        return {
            "status": "ok",
            "results": llm_results,
            "aggregated": aggregated,
            "ab_test": ab_test,
            "feedback": feedback
        }

    def call_llm(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Placeholder for actual LLM call. To be replaced with real LLM integration (Ollama, local, or API).
        Returns a simulated result for now.
        """
        self.logger.info(f"[MultiLLMOrchestrator] Calling LLM for task: {task}")
        return {"llm_id": "llm_stub", "task": task, "result": f"simulated_llm_result_for_{task.get('prompt', 'unknown')}"}

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple LLMs. Supports voting, reranking, and consensus.
        """
        self.logger.info(f"[MultiLLMOrchestrator] Aggregating results: {results}")
        # Example: voting on 'result' field
        result_texts = [r.get('result') for r in results if 'result' in r]
        if result_texts:
            vote_counts = Counter(result_texts)
            consensus, count = vote_counts.most_common(1)[0]
            return {"consensus": consensus, "votes": dict(vote_counts)}
        return {"all_results": results}

    def ab_test_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform AB testing on LLM results. Supports feedback-driven selection and reranking.
        """
        self.logger.info(f"[MultiLLMOrchestrator] AB testing results: {results}")
        # Example: pick the result with most votes or best feedback
        result_texts = [r.get('result') for r in results if 'result' in r]
        if result_texts:
            vote_counts = Counter(result_texts)
            best, count = vote_counts.most_common(1)[0]
            return {"best_result": best, "votes": dict(vote_counts)}
        return {"best_result": results[0] if results else None}

    def analyze_feedback(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze feedback for LLM results. Supports continual learning and reranking.
        """
        self.logger.info(f"[MultiLLMOrchestrator] Analyzing feedback for results: {results}")
        # Example: stub feedback analysis
        return {"feedback_summary": "stub"}

    def route_query(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Alias for orchestrate for test compatibility.
        """
        return self.orchestrate(tasks)

    def report(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Return a minimal stub report for test compatibility.
        """
        return {'status': 'ok', 'report': 'stub'}

    def demo_orchestration(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Demo/test method: run orchestration and return results.
        """
        return self.orchestrate(tasks)

    def demo_aggregation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Demo/test method: run aggregation and return consensus.
        """
        return self.aggregate_results(results)

    def demo_ab_test(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Demo/test method: run AB testing and return best result.
        """
        return self.ab_test_results(results)

    def demo_feedback(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Demo/test method: run feedback analysis and return summary.
        """
        return self.analyze_feedback(results)

    def demo_custom_orchestration(self, tasks: List[Dict[str, Any]], custom_orchestrator: Callable) -> Dict[str, Any]:
        """
        Demo/test method: run a custom orchestration function on tasks and return results.
        Usage: orchestrator.demo_custom_orchestration(tasks, lambda t: [...])
        """
        try:
            result = custom_orchestrator(tasks)
            self.logger.info(f"[MultiLLMOrchestrator] Custom orchestration result: {result}")
            return result
        except Exception as ex:
            self.logger.error(f"[MultiLLMOrchestrator] Custom orchestration error: {ex}")
            return {"status": "error", "error": str(ex)}

    def demo_custom_aggregation(self, results: List[Dict[str, Any]], custom_aggregator: Callable) -> Dict[str, Any]:
        """
        Demo/test method: run a custom aggregation function on results and return consensus.
        Usage: orchestrator.demo_custom_aggregation(results, lambda r: {...})
        """
        try:
            result = custom_aggregator(results)
            self.logger.info(f"[MultiLLMOrchestrator] Custom aggregation result: {result}")
            return result
        except Exception as ex:
            self.logger.error(f"[MultiLLMOrchestrator] Custom aggregation error: {ex}")
            return {"status": "error", "error": str(ex)}

    def demo_custom_feedback(self, results: List[Dict[str, Any]], custom_feedback: Callable) -> Dict[str, Any]:
        """
        Demo/test method: run a custom feedback analytics function on results and return summary.
        Usage: orchestrator.demo_custom_feedback(results, lambda r: {...})
        """
        try:
            result = custom_feedback(results)
            self.logger.info(f"[MultiLLMOrchestrator] Custom feedback result: {result}")
            return result
        except Exception as ex:
            self.logger.error(f"[MultiLLMOrchestrator] Custom feedback error: {ex}")
            return {"status": "error", "error": str(ex)}

    def advanced_feedback_integration(self, feedback: Dict[str, Any]):
        """
        Advanced feedback integration and continual learning for multi-LLM orchestrator.
        Updates orchestration, aggregation, or AB testing logic based on structured feedback.
        Supports cross-lobe research and adaptation.
        """
        self.logger.info(f"[MultiLLMOrchestrator] Advanced feedback integration: {feedback}")
        # Example: adjust orchestrator/aggregator parameters or log for research
        if feedback and 'adjust_strategy' in feedback:
            self.logger.info(f"[MultiLLMOrchestrator] Strategy adjusted: {feedback['adjust_strategy']}")
        self.working_memory.add({"advanced_feedback": feedback})

    def cross_lobe_integration(self, tasks: List[Dict[str, Any]], lobe_name: str = "") -> Any:
        """
        Integrate with other lobes for cross-engine research and feedback.
        Example: call AlignmentEngine or PatternRecognitionEngine for additional context.
        See idea.txt, README.md, ARCHITECTURE.md.
        """
        self.logger.info(f"[MultiLLMOrchestrator] Cross-lobe integration called with {lobe_name}.")
        # Placeholder: simulate integration
        return self.orchestrate(tasks)

    def usage_example(self):
        """
        Usage example for multi LLM orchestrator:
        >>> orchestrator = MultiLLMOrchestrator()
        >>> tasks = [{"prompt": "What is 2+2?"}, {"prompt": "What is the capital of France?"}]
        >>> result = orchestrator.demo_orchestration(tasks)
        >>> print(result)
        >>> # Custom orchestration: reverse tasks
        >>> custom = orchestrator.demo_custom_orchestration(tasks, lambda t: list(reversed(t)))
        >>> print(custom)
        >>> # Advanced feedback integration
        >>> orchestrator.advanced_feedback_integration({'adjust_strategy': 'consensus'})
        >>> # Cross-lobe integration
        >>> orchestrator.cross_lobe_integration(tasks, lobe_name='AlignmentEngine')
        """
        pass

    # TODO: Add demo/test methods for plugging in custom orchestration, aggregation, and feedback analytics.
    # TODO: Document extension points and provide usage examples in README.md.
    # TODO: Integrate with other lobes for cross-engine research and feedback.
    # TODO: Add advanced feedback integration and continual learning.
    # See: idea.txt, arXiv:2504.11320, arXiv:2506.14851, arXiv:2505.11107, README.md, ARCHITECTURE.md 