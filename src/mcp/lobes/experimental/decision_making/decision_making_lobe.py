from src.mcp.lobes.experimental.advanced_engram.advanced_engram_engine import WorkingMemory
import logging
from src.mcp.lobes.experimental.vesicle_pool import VesiclePool

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
    - Plug in custom decision heuristics (utility, scenario simulation, feedback-weighted, RL, AB testing)
    - Add batch decision-making and scenario simulation
    - Integrate with other lobes for cross-engine research
    """
    def __init__(self):
        self.working_memory = WorkingMemory()
        self.logger = logging.getLogger("DecisionMakingLobe")
        self.task_stack = TaskStack()
        self.contextual_buffer = ContextualTaskBuffer()
        self.vesicle_pool = VesiclePool()  # Synaptic vesicle pool model
        self.logger.info("[DecisionMakingLobe] VesiclePool initialized: %s", self.vesicle_pool.get_state())
        # TODO: Add support for pluggable decision heuristics and batch scenario simulation

    def recommend_action(self, options, context=None, priority=0, feedback=None, decision_heuristic=None):
        """
        Recommend an action from a list of options, optionally using context, priority, feedback, and a custom decision heuristic.
        References: idea.txt (decision trees, utility scoring, scenario simulation, context-aware memory).
        Fallback: Returns the first option or a random choice if advanced logic is not implemented.
        TODO: Add support for utility scoring, scenario simulation, feedback-weighted, RL, and AB testing heuristics.
        """
        self.logger.info(f"[DecisionMakingLobe] Recommending action from options: {options}")
        if not options:
            return None
        chosen = None
        try:
            if decision_heuristic and callable(decision_heuristic):
                chosen = decision_heuristic(options, context=context, priority=priority, feedback=feedback)
            elif context and 'priority' in context:
                sorted_options = sorted(options, key=lambda x: context['priority'].get(x, 0), reverse=True)
                chosen = sorted_options[0]
            else:
                import random
                chosen = options[0] if options else None
        except Exception as ex:
            self.logger.error(f"[DecisionMakingLobe] Decision heuristic error: {ex}")
            chosen = options[0] if options else None
        if chosen:
            self.task_stack.push(chosen)
            self.contextual_buffer.add(chosen, context=context, priority=priority, feedback=feedback)
        self.task_stack.decay_stack()
        self.contextual_buffer.decay_buffer()
        return chosen

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

    # TODO: Add batch decision-making and scenario simulation methods.
    # TODO: Add demo/test methods for plugging in custom decision heuristics.
    # TODO: Document extension points and provide usage examples in README.md.
    # TODO: Integrate with other lobes for cross-engine research and feedback.
    # TODO: Add advanced feedback integration and continual learning.
    # See: idea.txt, NeurIPS 2025, ICLR 2025, AAAI 2024, README.md, ARCHITECTURE.md 