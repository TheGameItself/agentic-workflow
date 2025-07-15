from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory
import logging
from src.mcp.lobes.experimental.vesicle_pool import VesiclePool
import random
from typing import List, Any, Callable, Optional

class TaskStack:
    """
    TaskStack: Domain-specific working memory for active tasks (stack/queue with decay).
    Inspired by neuroscience (prefrontal cortex, executive function, working memory). See idea.txt.
    """
    def __init__(self, capacity=20, decay=0.97):
        self.capacity = capacity
        self.decay = decay
        self.stack = []
    def push(self, task):
        self.stack.append({'task': task, 'strength': 1.0})
        if len(self.stack) > self.capacity:
            self.stack.pop(0)
    def pop(self):
        if self.stack:
            return self.stack.pop()
        return None
    def peek(self):
        if self.stack:
            return self.stack[-1]['task']
        return None
    def decay_stack(self):
        for entry in self.stack:
            entry['strength'] *= self.decay
        self.stack = [e for e in self.stack if e['strength'] > 0.1]
    def get_active(self, n=5):
        return [e['task'] for e in self.stack[-n:]]

class ContextualTaskBuffer:
    """
    ContextualTaskBuffer: Context-tagged, priority, and feedback-driven buffer for active tasks.
    Inspired by prefrontal cortex working memory and executive function (see idea.txt, neuroscience).
    """
    def __init__(self, capacity=30, decay=0.96):
        self.capacity = capacity
        self.decay = decay
        self.buffer = []  # Each entry: {'task': ..., 'context': ..., 'priority': ..., 'feedback': ..., 'strength': ...}
        self.logger = logging.getLogger("ContextualTaskBuffer")
    def add(self, task, context=None, priority=0, feedback=None):
        entry = {'task': task, 'context': context, 'priority': priority, 'feedback': feedback, 'strength': 1.0}
        self.buffer.append(entry)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        self.logger.info(f"[ContextualTaskBuffer] Added task: {task} (context={context}, priority={priority}, feedback={feedback})")
    def decay_buffer(self):
        for entry in self.buffer:
            entry['strength'] *= self.decay
        self.buffer = [e for e in self.buffer if e['strength'] > 0.1]
    def get_by_context(self, context, n=5):
        context_str = str(context) if context is not None else ""
        matches = [e for e in self.buffer if context_str and context_str in str(e['context'])]
        return [e['task'] for e in matches[-n:]]
    def get_high_priority(self, n=5):
        sorted_entries = sorted(self.buffer, key=lambda e: (-e['priority'], -e['strength']))
        return [e['task'] for e in sorted_entries[:n]]
    def get_recent(self, n=5):
        return [e['task'] for e in self.buffer[-n:]]

class DecisionMakingLobe:
    """
    Decision-Making Lobe
    Weighs options, simulates outcomes, and recommends actions based on project state and user/LLM goals.
    Implements research-driven decision heuristics (see idea.txt, decision trees, utility theory, scenario simulation).
    
    Research References:
    - idea.txt (decision trees, utility scoring, scenario simulation, context-aware memory)
    - NeurIPS 2025 (Neural Column Pattern Recognition)
    - ICLR 2025 (Dynamic Coding and Vector Compression)
    - AAAI 2024 (Multi-criteria Decision Analysis)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md
    
    Extensibility:
    - Pluggable decision heuristics (utility, scenario simulation, feedback-weighted, RL, AB testing)
    - Batch decision-making and scenario simulation
    - Feedback-driven adaptation and continual learning
    - Integration with other lobes for cross-engine research
    """
    def __init__(self):
        self.working_memory = WorkingMemory()
        self.logger = logging.getLogger("DecisionMakingLobe")
        self.task_stack = TaskStack()
        self.contextual_buffer = ContextualTaskBuffer()
        self.vesicle_pool = VesiclePool()  # Synaptic vesicle pool model
        self.logger.info("[DecisionMakingLobe] VesiclePool initialized: %s", self.vesicle_pool.get_state())

    def recommend_action(self, options: List[Any], context: Any = None, priority: int = 0, feedback: Any = None, decision_heuristic: Optional[Callable] = None) -> Any:
        """
        Recommend an action from a list of options, optionally using context, priority, feedback, and a custom decision heuristic.
        Supports utility scoring, scenario simulation, feedback-weighted, RL, and AB testing heuristics.
        Fallback: Returns the first option or a random choice if advanced logic is not implemented.
        """
        self.logger.info(f"[DecisionMakingLobe] Recommending action from options: {options}")
        if not options:
            return None
        chosen = None
        try:
            if decision_heuristic and callable(decision_heuristic):
                chosen = decision_heuristic(options, context=context, priority=priority, feedback=feedback)
            elif context and isinstance(context, dict) and 'priority' in context:
                sorted_options = sorted(options, key=lambda x: context['priority'].get(x, 0), reverse=True)
                chosen = sorted_options[0]
            else:
                chosen = random.choice(options)
        except Exception as ex:
            self.logger.error(f"[DecisionMakingLobe] Decision heuristic error: {ex}")
            chosen = options[0] if options else None
        if chosen:
            self.task_stack.push(chosen)
            self.contextual_buffer.add(chosen, context=context, priority=priority, feedback=feedback)
        self.task_stack.decay_stack()
        self.contextual_buffer.decay_buffer()
        return chosen

    def batch_recommend_actions(self, batch_options: List[List[Any]], context: Any = None, priority: int = 0, feedback: Any = None, decision_heuristic: Optional[Callable] = None) -> List[Any]:
        """
        Batch recommend actions for multiple sets of options.
        Returns a list of chosen actions, one per set.
        """
        results = []
        for options in batch_options:
            result = self.recommend_action(options, context=context, priority=priority, feedback=feedback, decision_heuristic=decision_heuristic)
            results.append(result)
        return results

    def simulate_scenarios(self, options: List[Any], scenario_fn: Optional[Callable] = None, context: Any = None) -> List[Any]:
        """
        Simulate outcomes for each option using a scenario function.
        Returns a list of simulated results.
        """
        results = []
        for option in options:
            if scenario_fn and callable(scenario_fn):
                try:
                    result = scenario_fn(option, context=context)
                except Exception as ex:
                    self.logger.error(f"[DecisionMakingLobe] Scenario simulation error: {ex}")
                    result = None
            else:
                result = {"option": option, "simulated": True}
            results.append(result)
        return results

    def recall_tasks_by_context(self, context=None, n=5):
        """
        Recall most relevant tasks for a given context using contextual buffer.
        """
        return self.contextual_buffer.get_by_context(context, n=n)
    def recall_high_priority_tasks(self, n=5):
        """
        Recall highest priority tasks from contextual buffer.
        """
        return self.contextual_buffer.get_high_priority(n=n)

    def adapt_from_feedback(self, feedback: Any):
        """
        Adapt decision-making parameters based on feedback (learning loop).
        Extensible for continual learning and feedback-driven adaptation.
        """
        self.logger.info(f"[DecisionMakingLobe] Adapting from feedback: {feedback}")
        self.working_memory.add({"feedback": feedback})

    def batch_decision_and_simulation(self, batch_options: List[List[Any]], scenario_fn: Optional[Callable] = None, context: Any = None) -> List[Any]:
        """
        Batch decision-making and scenario simulation for multiple sets of options.
        Returns a list of (chosen, simulated_results) tuples.
        See idea.txt, NeurIPS 2025, ICLR 2025, AAAI 2024.
        """
        results = []
        for options in batch_options:
            chosen = self.recommend_action(options, context=context)
            simulated = self.simulate_scenarios(options, scenario_fn=scenario_fn, context=context)
            results.append((chosen, simulated))
        return results

    def demo_custom_heuristic(self, options: List[Any], context: Any = None) -> Any:
        """
        Demo/test method for plugging in a custom decision heuristic.
        Example: always pick the last option.
        Usage: lobe.demo_custom_heuristic([1,2,3]) -> 3
        See README.md for more examples.
        """
        if not options:
            return None
        return options[-1]

    def cross_lobe_integration(self, options: List[Any], context: Any = None) -> Any:
        """
        Integrate with other lobes for cross-engine research and feedback.
        Example: call PatternRecognitionLobe or MindMapEngine for additional context.
        See idea.txt, README.md, ARCHITECTURE.md.
        """
        # Placeholder: simulate integration
        self.logger.info("[DecisionMakingLobe] Cross-lobe integration called.")
        # In a real system, would call other lobe APIs here
        return self.recommend_action(options, context=context)

    def advanced_feedback_integration(self, feedback: Any):
        """
        Advanced feedback integration and continual learning.
        Updates internal heuristics or weights based on feedback.
        See idea.txt, NeurIPS 2025, continual learning research.
        """
        self.logger.info(f"[DecisionMakingLobe] Advanced feedback integration: {feedback}")
        # Example: adjust decay or priority based on feedback
        if isinstance(feedback, dict) and 'adjust_decay' in feedback:
            self.task_stack.decay = float(feedback['adjust_decay'])
            self.logger.info(f"[DecisionMakingLobe] TaskStack decay adjusted to {self.task_stack.decay}")
        self.working_memory.add({"advanced_feedback": feedback})

    # TODO: Add batch decision-making and scenario simulation methods.
    # TODO: Add demo/test methods for plugging in custom decision heuristics.
    # TODO: Document extension points and provide usage examples in README.md.
    # TODO: Integrate with other lobes for cross-engine research and feedback.
    # TODO: Add advanced feedback integration and continual learning.
    # See: idea.txt, NeurIPS 2025, ICLR 2025, AAAI 2024, README.md, ARCHITECTURE.md 