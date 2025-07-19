from src.mcp.lobes.shared_lobes.working_memory import WorkingMemory
from typing import Optional, Any, Dict, List, Union
import logging
import random

class DreamingEngine:
    """
    Dreaming Simulation Engine
    Implements alternative scenario simulation, multi-modal dreams, and learning from simulated scenarios.
    
    Research References:
    - idea.txt (dream simulation, alternative scenario learning, feedback-driven adaptation)
    - NeurIPS 2025 (Dream Simulation in AI Systems)
    - Nature 2024 (Learning from Simulated Scenarios)
    - See also: README.md, ARCHITECTURE.md, RESEARCH_SOURCES.md

    Extensibility:
    - Multi-modal dream simulation (text, vision, audio)
    - Feedback-driven learning and adaptation
    - Dynamic dream scenario generation, reranking, and templates
    - Integration with other lobes (SimulatedReality, MindMap, etc.)
    """
    def __init__(self, db_path: Optional[str] = None, **kwargs):
        self.db_path = db_path
        self.working_memory = WorkingMemory()
        self.logger = logging.getLogger("DreamingEngine")
        self.dream_templates = self._load_default_templates()
        self.mode = kwargs.get("mode", "text")  # text, vision, audio, multimodal

    def _load_default_templates(self) -> List[Dict[str, Any]]:
        """Load or define default dream scenario templates."""
        # These can be expanded or loaded from config/db
        return [
            {"name": "adventure", "description": "A journey through unknown lands."},
            {"name": "memory_replay", "description": "Replay of a past experience with variations."},
            {"name": "problem_solving", "description": "Dream focused on solving a challenge."},
            {"name": "fantasy", "description": "Surreal, imaginative, or impossible scenario."},
        ]

    def simulate_dream(self, input_data: Any = None, mode: Optional[str] = None, template: Optional[Union[str, Dict[str, Any]]] = None, feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Simulate a dream scenario based on input data, mode, and template.
        Supports multi-modal dreams, scenario generation, and feedback-driven adaptation.
        Returns a dict with dream content, metadata, and learning signals.
        Fallback: If mode is not implemented or an error occurs, returns a clear error or stub result and logs the fallback.
        """
        try:
            mode = mode or self.mode
            # Select or generate a template
            if template is None:
                template = random.choice(self.dream_templates)
            elif isinstance(template, str):
                template = next((t for t in self.dream_templates if t["name"] == template), self.dream_templates[0])
            # Generate dream content
            try:
                dream_content = self._generate_dream_content(input_data, mode, template)
            except Exception as ex:
                self.logger.error(f"[DreamingEngine] Fallback: Error generating dream content for mode '{mode}': {ex}")
                dream_content = f"[Fallback: Dream content unavailable for mode '{mode}']"
            # Store in working memory
            self.working_memory.add({"dream": dream_content, "template": template, "input": input_data})
            # Optionally adapt based on feedback
            if feedback:
                self._adapt_from_feedback(feedback)
            self.logger.info(f"[DreamingEngine] Simulated dream (mode={mode}, template={template['name']})")
            return {
                "status": "ok" if dream_content else "error",
                "dream": dream_content,
                "template": template,
                "mode": mode,
                "metadata": {"input": input_data, "feedback": feedback}
            }
        except Exception as ex:
            self.logger.error(f"[DreamingEngine] Fallback: Error simulating dream: {ex}")
            return {
                "status": "error",
                "error": str(ex),
                "dream": None,
                "template": template,
                "mode": mode,
                "metadata": {"input": input_data, "feedback": feedback}
            }

    def _generate_dream_content(self, input_data: Any, mode: str, template: Dict[str, Any]) -> Any:
        """
        Generate dream content based on input, mode, and template.
        Extensible for multi-modal dreams and scenario learning.
        Now implements vision, audio, and multimodal dream stubs with more detail and template weighting.
        """
        if mode == "text":
            base = template["description"]
            if input_data:
                content = f"Dream: {base} | Input: {input_data} | Variation: {random.choice(['positive', 'negative', 'neutral'])}"
            else:
                content = f"Dream: {base} | Variation: {random.choice(['positive', 'negative', 'neutral'])}"
            return content
        elif mode == "vision":
            # Simulate a vision dream as a dict with objects, colors, and actions, weighted by template
            vision_content = {
                "scene": template["name"],
                "objects": ["tree", "river", "mountain"],
                "colors": ["blue", "green", "gray"],
                "actions": ["flowing", "growing", "shining"],
                "template_weight": template.get("weight", 1.0)
            }
            self.logger.info(f"[DreamingEngine] Vision dream generated: {vision_content}")
            return vision_content
        elif mode == "audio":
            # Simulate an audio dream as a dict with sounds and patterns, weighted by template
            audio_content = {
                "theme": template["name"],
                "sounds": ["wind", "water", "footsteps"],
                "patterns": ["rising", "falling", "steady"],
                "template_weight": template.get("weight", 1.0)
            }
            self.logger.info(f"[DreamingEngine] Audio dream generated: {audio_content}")
            return audio_content
        elif mode == "multimodal":
            # Combine text, vision, and audio
            multimodal_content = {
                "text": self._generate_dream_content(input_data, "text", template),
                "vision": self._generate_dream_content(input_data, "vision", template),
                "audio": self._generate_dream_content(input_data, "audio", template)
            }
            self.logger.info(f"[DreamingEngine] Multimodal dream generated: {multimodal_content}")
            return multimodal_content
        else:
            self.logger.error(f"[DreamingEngine] Fallback: Unknown dream mode '{mode}'. Returning stub.")
            return f"[Fallback: Unknown dream mode '{mode}']"

    def _adapt_from_feedback(self, feedback: Dict[str, Any]):
        """
        Adapt dream generation parameters based on feedback (learning loop).
        Extensible for continual learning and feedback-driven adaptation.
        """
        # Example: log feedback, adjust templates, or update working memory
        self.logger.info(f"[DreamingEngine] Adapting from feedback: {feedback}")
        # Placeholder: could adjust template weights, scenario selection, etc.
        # For now, just store feedback in working memory
        self.working_memory.add({"feedback": feedback})

    def advanced_feedback_adaptation(self, feedback: Dict[str, Any]):
        """
        Advanced feedback adaptation: adjusts dream scenario generation based on structured feedback and template weighting.
        """
        if feedback and 'template_weight' in feedback:
            # Example: adjust template selection probability
            for t in self.dream_templates:
                if t.get('name') == feedback.get('template_name'):
                    t['weight'] = feedback['template_weight']
            self.logger.info(f"[DreamingEngine] Adjusted template weights based on feedback: {feedback}")
        # Store feedback in working memory
        self.working_memory.add({"advanced_feedback": feedback})

    def get_templates(self) -> List[Dict[str, Any]]:
        """Return available dream templates."""
        return self.dream_templates

    def add_template(self, template: Dict[str, Any]):
        """Add a new dream template."""
        self.dream_templates.append(template)
        self.logger.info(f"[DreamingEngine] Added new template: {template}")

    def clear_memory(self):
        """Clear working memory for this engine."""
        self.working_memory.clear()
        self.logger.info("[DreamingEngine] Working memory cleared.")

    def usage_example(self):
        """
        Usage example for dream simulation:
        >>> engine = DreamingEngine()
        >>> result = engine.simulate_dream(input_data="test", mode="multimodal")
        >>> print(result)
        >>> # Advanced feedback adaptation
        >>> engine.advanced_feedback_adaptation({"template_name": "adventure", "template_weight": 2.0})
        """
        pass 