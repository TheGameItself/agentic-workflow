from src.mcp.lobes.experimental.advanced_engram.advanced_engram_engine import WorkingMemory
from typing import Optional, List, Dict, Any
import logging
import random
from datetime import datetime
from src.mcp.lobes.experimental.vesicle_pool import VesiclePool

class ScientificProcessEngine:
    """
    Scientific Process Engine
    Hypothesis testing, evidence tracking, and truth determination. Implements research-driven scientific process automation (see idea.txt).
    Tracks evidence, confidence, and status for each hypothesis. Stores evidence and results in working memory.
    """
    def __init__(self, db_path: Optional[str] = None, **kwargs):
        self.db_path = db_path
        self.working_memory = WorkingMemory()
        self.logger = logging.getLogger("ScientificProcessEngine")
        self.hypotheses: Dict[str, Dict[str, Any]] = {}
        self.evidence: Dict[str, List[Dict[str, Any]]] = {}
        self.vesicle_pool = VesiclePool()  # Synaptic vesicle pool model
        self.logger.info("[ScientificProcessEngine] VesiclePool initialized: %s", self.vesicle_pool.get_state())

    def test_hypothesis(self, hypothesis_id: str) -> Dict[str, Any]:
        """
        Test a hypothesis, track evidence, and determine truth status.
        References: idea.txt (evidence tracking, hypothesis testing, feedback).
        Returns result, confidence, and status.
        """
        self.logger.info(f"[ScientificProcessEngine] Testing hypothesis: {hypothesis_id}")
        hypothesis = self.hypotheses.get(hypothesis_id)
        if not hypothesis:
            return {"result": None, "status": "no_hypothesis"}
        # Gather evidence
        evidence_list = self.evidence.get(hypothesis_id, [])
        confidence = self._calculate_confidence(evidence_list)
        # Determine status
        if confidence > 0.8:
            status = "validated"
        elif confidence > 0.6:
            status = "partially_supported"
        elif confidence < 0.2:
            status = "refuted"
        else:
            status = "inconclusive"
        result = {
            "result": status,
            "confidence": confidence,
            "evidence_count": len(evidence_list),
            "evidence": evidence_list
        }
        self.working_memory.add(result)
        hypothesis["status"] = status
        hypothesis["confidence"] = confidence
        return result

    def propose_hypothesis(self, description=None, assumptions=None, variables=None, category="general", confidence=0.5) -> Dict[str, Any]:
        """
        Propose a new hypothesis for testing purposes. Stores in memory and returns hypothesis ID.
        """
        hypothesis_id = f"hyp_{len(self.hypotheses)+1}_{int(datetime.now().timestamp())}"
        hypothesis = {
            "id": hypothesis_id,
            "description": description or "default hypothesis",
            "assumptions": assumptions or [],
            "variables": variables or [],
            "category": category,
            "confidence": confidence,
            "status": "proposed",
            "created_at": datetime.now().isoformat()
        }
        self.hypotheses[hypothesis_id] = hypothesis
        self.working_memory.add(hypothesis)
        return {"hypothesis_id": hypothesis_id, **hypothesis}

    def add_evidence(self, hypothesis_id: str, evidence_type: str, data: Any, source: str = "experiment", confidence: float = 0.5) -> Dict[str, Any]:
        """
        Add evidence to a hypothesis. Stores in working memory and evidence dict.
        """
        evidence_entry = {
            "evidence_type": evidence_type,
            "data": data,
            "source": source,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        if hypothesis_id not in self.evidence:
            self.evidence[hypothesis_id] = []
        self.evidence[hypothesis_id].append(evidence_entry)
        self.working_memory.add(evidence_entry)
        return {"status": "added", **evidence_entry}

    def analyze_hypothesis(self, hypothesis_id: str) -> Dict[str, Any]:
        """
        Analyze a hypothesis based on all available evidence. Returns status, confidence, and summary.
        """
        hypothesis = self.hypotheses.get(hypothesis_id)
        if not hypothesis:
            return {"status": "no_hypothesis"}
        evidence_list = self.evidence.get(hypothesis_id, [])
        confidence = self._calculate_confidence(evidence_list)
        status = hypothesis.get("status", "proposed")
        summary = {
            "hypothesis_id": hypothesis_id,
            "description": hypothesis.get("description"),
            "status": status,
            "confidence": confidence,
            "evidence_count": len(evidence_list),
            "evidence": evidence_list
        }
        self.working_memory.add(summary)
        return summary

    def _calculate_confidence(self, evidence_list: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence in a hypothesis based on evidence. Simple weighted average.
        """
        if not evidence_list:
            return 0.5
        total = sum(e.get("confidence", 0.5) for e in evidence_list)
        return min(1.0, max(0.0, total / len(evidence_list))) 