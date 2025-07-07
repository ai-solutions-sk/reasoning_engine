# import time
# import random
# import os
# import re
# import google.generativeai as genai
# import numpy as np
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import json
# import math
# from typing import Dict, List, Tuple, Any, Optional
# from dataclasses import dataclass, field
# from enum import Enum
# import networkx as nx
# from collections import defaultdict
# import sympy as sp
# from sympy import symbols, solve, diff, integrate, simplify
# from sympy.logic import simplify_logic
# import itertools
# import zlib

# # --- Configuration ---
# # It is recommended to use environment variables for API keys for security.
# # For this example, we will use the key provided, but os.environ.get is safer.
# API_KEY = "AIzaSyBLXXuiqpx9BfDxGi28Ci8szlsb3qAm9Dw"
# # API_KEY = os.environ.get("GEMINI_API_KEY") 
# if API_KEY:
#     genai.configure(api_key=API_KEY)
# else:
#     print("API Key not found. Please set the GEMINI_API_KEY environment variable.")


# app = Flask(__name__)
# CORS(app)

# # --- Advanced Reasoning Types ---
# class ReasoningType(Enum):
#     DEDUCTIVE = "deductive"
#     INDUCTIVE = "inductive"
#     ABDUCTIVE = "abductive"
#     ANALOGICAL = "analogical"
#     CAUSAL = "causal"
#     PROBABILISTIC = "probabilistic"
#     COUNTERFACTUAL = "counterfactual"
#     MODAL = "modal"
#     TEMPORAL = "temporal"
#     SPATIAL = "spatial"
#     METACOGNITIVE = "metacognitive"
#     DIALECTICAL = "dialectical"

# @dataclass
# class ReasoningPath:
#     """Represents a single reasoning path with mathematical backing"""
#     path_id: str
#     reasoning_type: Any # Can be ReasoningType or a dynamic Enum
#     mathematical_foundation: str
#     logical_structure: str
#     confidence_score: float
#     pattern_complexity: int
#     epistemic_depth: int

# @dataclass
# class PatternSignature:
#     """Mathematical signature of discovered patterns"""
#     pattern_id: str
#     mathematical_expression: str
#     symmetry_group: str = "N/A"
#     invariants: List[str] = field(default_factory=list)
#     transformations: List[str] = field(default_factory=list)
#     information_content: float = 0.0

# # --- Supporting Classes ---

# class SymbolicReasoningEngine:
#     """Advanced symbolic mathematical reasoning"""
    
#     def __init__(self):
#         self.symbol_registry = {}
        
#     def analyze_symbolically(self, problem: str, patterns: List[PatternSignature]) -> Dict[str, Any]:
#         """Perform symbolic analysis using SymPy"""
#         expressions = self._extract_mathematical_expressions(problem)
#         for p in patterns:
#             if p.mathematical_expression:
#                 expressions.append(p.mathematical_expression)
        
#         symbolic_results = {}
#         for expr in set(expressions):
#             try:
#                 parsed = sp.sympify(expr, locals=self.symbol_registry)
#                 simplified = simplify(parsed)
                
#                 if parsed.free_symbols:
#                     derivatives = {str(var): str(diff(parsed, var)) for var in parsed.free_symbols}
#                     integrals = {str(var): str(integrate(parsed, var)) for var in parsed.free_symbols}
                    
#                     symbolic_results[expr] = {
#                         "simplified": str(simplified),
#                         "derivatives": derivatives,
#                         "integrals": integrals,
#                         "free_symbols": [str(s) for s in parsed.free_symbols]
#                     }
#             except (sp.SympifyError, TypeError, SyntaxError) as e:
#                 print(f"Symbolic analysis error for '{expr}': {e}")
#         return symbolic_results
    
#     def _extract_mathematical_expressions(self, problem: str) -> List[str]:
#         """Extract mathematical expressions from problem text"""
#         patterns = [
#             r'[a-zA-Z]+\s*=\s*[0-9a-zA-Z\+\-\*/\^\(\)\s\.,_]+',
#             r'[a-zA-Z]\([a-zA-Z0-9\,\s]+\)\s*=\s*.+',
#             r'\\frac\{[^}]+\}\{[^}]+\}',
#             r'\b[a-zA-Z]\w*\^[0-9.]+\b',
#             r'\b(log|sin|cos|tan|exp)\([^)]+\)',
#             r'\b\d+\s*[\+\-\*/]\s*\d+\b'
#         ]
#         expressions = []
#         for pattern in patterns:
#             try:
#                 expressions.extend(re.findall(pattern, problem))
#             except re.error as e:
#                 print(f"Regex error with pattern '{pattern}': {e}")
#         return expressions

# class MetaCognitiveMonitor:
#     """Monitors and optimizes reasoning processes"""
    
#     def optimize_reasoning(self, paths: List[ReasoningPath]) -> List[ReasoningPath]:
#         """Optimize reasoning paths using meta-cognitive strategies"""
#         sorted_paths = sorted(paths, 
#                               key=lambda p: (p.confidence_score * 0.5 + p.epistemic_depth * 0.3 + p.pattern_complexity * 0.2), 
#                               reverse=True)
#         return [path for path in sorted_paths if self._passes_meta_cognitive_filter(path)]
    
#     def _passes_meta_cognitive_filter(self, path: ReasoningPath) -> bool:
#         """Apply meta-cognitive criteria to filter reasoning paths"""
#         if path.confidence_score < 0.3 or path.epistemic_depth < 2:
#             return False
#         return self._check_logical_consistency(path)
    
#     def _check_logical_consistency(self, path: ReasoningPath) -> bool:
#         """Check logical consistency of reasoning path"""
#         if path.reasoning_type == ReasoningType.DIALECTICAL:
#             return "P ∧ ¬P" in path.mathematical_foundation
#         return True

# class PatternDetector:
#     """Detects deep patterns in problems and solutions"""
    
#     def detect_deep_patterns(self, problem: str, problem_structure: Dict[str, Any]) -> List[PatternSignature]:
#         """Detect deep mathematical and linguistic patterns"""
#         patterns = []
#         patterns.extend(self._detect_symmetries(problem))
#         patterns.extend(self._detect_recursive_patterns(problem))
#         patterns.extend(self._detect_scaling_patterns(problem))
#         patterns.extend(self._detect_topological_patterns(problem_structure))
#         return patterns
    
#     def _detect_symmetries(self, problem: str) -> List[PatternSignature]:
#         """Detect symmetry patterns (e.g., if x=y, then y=x)"""
#         matches = re.findall(r'(\w+)\s*=\s*(\w+)', problem)
#         symmetries = []
#         for a, b in matches:
#             if f"{b} = {a}" in problem or f"{b}=={a}" in problem:
#                 symmetries.append(PatternSignature(
#                     pattern_id=f"symmetry_{a}_{b}",
#                     mathematical_expression=f"{a}={b} <=> {b}={a}",
#                     symmetry_group="Z2"
#                 ))
#         return symmetries
    
#     def _detect_recursive_patterns(self, problem: str) -> List[PatternSignature]:
#         """Detect recursive patterns like f(n) = ... f(n-1)"""
#         matches = re.findall(r'(\w+)\(n\)\s*=\s*.*?\1\(n\s*-\s*1\)', problem)
#         return [PatternSignature(pattern_id=f"recursion_{m}", mathematical_expression=f"{m}(n) = f({m}(n-1))") for m in matches]
    
#     def _detect_scaling_patterns(self, problem: str) -> List[PatternSignature]:
#         """Detect scaling patterns (power laws)"""
#         matches = re.findall(r'(\w+)\s*(?:is proportional to|scales with)\s*(\w+)\^(\w+)', problem)
#         return [PatternSignature(pattern_id=f"scaling_{y}_{x}", mathematical_expression=f"{y} ∝ {x}^{a}") for y, x, a in matches]
    
#     def _detect_topological_patterns(self, problem_structure: Dict[str, Any]) -> List[PatternSignature]:
#         """Detect topological patterns from the concept graph"""
#         patterns = []
#         topology = problem_structure.get("topology", {})
#         if topology.get("clustering", 0) > 0.5 and topology.get("nodes", 0) > 10:
#             patterns.append(PatternSignature(pattern_id="high_clustering_topology", mathematical_expression="High Clustering Coefficient C > 0.5"))
#         if topology.get("components", 1) > 1:
#             patterns.append(PatternSignature(pattern_id="disconnected_topology", mathematical_expression="Disconnected Concept Graph"))
#         return patterns

# class CrossModalIntegrator:
#     """Integrates insights across different reasoning modalities"""
    
#     def integrate_insights(self, reasoning_paths: List[ReasoningPath], symbolic_insights: Dict[str, Any], patterns: List[PatternSignature]) -> Dict[str, Any]:
#         """Integrate insights from different reasoning modalities"""
#         return {
#             "reasoning_paths": [path.__dict__ for path in reasoning_paths],
#             "symbolic_insights": symbolic_insights,
#             "patterns": [p.__dict__ for p in patterns],
#             "integration_score": self._calculate_integration_score(reasoning_paths, symbolic_insights, patterns)
#         }
    
#     def _calculate_integration_score(self, reasoning_paths: List[ReasoningPath], symbolic_insights: Dict[str, Any], patterns: List[PatternSignature]) -> float:
#         """Calculate how well different insights integrate"""
#         if not reasoning_paths: return 0.0
#         path_score = sum(p.confidence_score for p in reasoning_paths) / len(reasoning_paths)
#         symbolic_validation_count = sum(1 for p in patterns if p.mathematical_expression in symbolic_insights)
#         symbolic_score = symbolic_validation_count / len(patterns) if patterns else 0
#         pattern_score = len(patterns) / 10.0
#         return min(1.0, (path_score * 0.5 + symbolic_score * 0.3 + pattern_score * 0.2))

# # --- Main Reasoning Engine ---
# class AdvancedReasoningEngine:
#     """The main class that orchestrates the entire advanced reasoning process."""
#     def __init__(self, model_name: str = "gemini-1.5-pro"):
#         if API_KEY:
#             self.model = genai.GenerativeModel(model_name)
#         else:
#             self.model = None
#         self.symbolic_engine = SymbolicReasoningEngine()
#         self.meta_cognitive_monitor = MetaCognitiveMonitor()
#         self.pattern_detector = PatternDetector()
#         self.cross_modal_integrator = CrossModalIntegrator()
        
#     def ultra_reasoning(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
#         """Orchestrates the ultra-advanced reasoning pipeline."""
#         if not self.model:
#             return {"error": "Gemini API key not configured."}

#         problem_structure = self._decompose_problem_mathematically(problem)
#         patterns = self.pattern_detector.detect_deep_patterns(problem, problem_structure)
#         reasoning_paths = self._generate_reasoning_paths(problem, patterns, context)
#         symbolic_insights = self.symbolic_engine.analyze_symbolically(problem, patterns)
#         optimized_paths = self.meta_cognitive_monitor.optimize_reasoning(reasoning_paths)
#         integrated_solution = self.cross_modal_integrator.integrate_insights(optimized_paths, symbolic_insights, patterns)
#         exceptional_insights = self._discover_exceptional_patterns(integrated_solution)
        
#         return self._synthesize_ultra_reasoning(problem, integrated_solution, exceptional_insights)
    
#     def _decompose_problem_mathematically(self, problem: str) -> Dict[str, Any]:
#         """Decomposes a text problem into multiple mathematical representations."""
#         concept_graph = self._build_concept_graph(problem)
#         return {
#             "entropy": self._calculate_semantic_entropy(problem),
#             "complexity": self._calculate_kolmogorov_complexity(problem),
#             "topology": self._analyze_concept_topology(concept_graph),
#             "functional_components": self._extract_functional_components(problem),
#             "algebraic_structure": self._identify_algebraic_structure(problem),
#             "concept_graph": nx.node_link_data(concept_graph)
#         }

#     def _calculate_semantic_entropy(self, text: str) -> float:
#         """Calculates semantic entropy of text based on word frequency."""
#         words = re.findall(r'\b\w+\b', text.lower())
#         if not words: return 0.0
#         freq_dist = defaultdict(int)
#         for word in words:
#             freq_dist[word] += 1
#         entropy = 0.0
#         total_words = len(words)
#         for freq in freq_dist.values():
#             prob = freq / total_words
#             entropy -= prob * math.log2(prob)
#         return entropy

#     def _calculate_kolmogorov_complexity(self, text: str) -> float:
#         """Approximates Kolmogorov complexity using compression ratio."""
#         if not text: return 0.0
#         return len(zlib.compress(text.encode('utf-8'))) / len(text.encode('utf-8'))

#     def _build_concept_graph(self, problem: str) -> nx.Graph:
#         """Builds a concept graph from problem text."""
#         words = re.findall(r'\b[a-zA-Z]{3,}\b', problem.lower())
#         g = nx.Graph()
#         if not words: return g
#         for i, word in enumerate(words):
#             g.add_node(word)
#             if i > 0: g.add_edge(words[i-1], word)
#         return g

#     def _analyze_concept_topology(self, g: nx.Graph) -> Dict[str, Any]:
#         """Analyzes topological properties of the concept graph."""
#         if g.number_of_nodes() == 0:
#             return {"nodes": 0, "edges": 0, "density": 0, "clustering": 0, "components": 0}
#         return {
#             "nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
#             "density": nx.density(g), "clustering": nx.average_clustering(g),
#             "components": nx.number_connected_components(g)
#         }

#     def _extract_functional_components(self, problem: str) -> List[str]:
#         """Extracts functional components like 'f(x)' or 'if-then' from text."""
#         patterns = [r'\b\w+\([^)]*\)', r'if\s+.*?\s+then\s+.*']
#         components = []
#         for p in patterns:
#             components.extend(re.findall(p, problem, re.IGNORECASE))
#         return components

#     def _identify_algebraic_structure(self, problem: str) -> Dict[str, Any]:
#         """Identifies implicit algebraic structures in the text."""
#         return {
#             'operations': {'addition': len(re.findall(r'\+', problem)), 'subtraction': len(re.findall(r'-', problem)),
#                            'multiplication': len(re.findall(r'[\*×]', problem)), 'division': len(re.findall(r'[/÷]', problem)),
#                            'equality': len(re.findall(r'=', problem))},
#             'variables': len(set(re.findall(r'\b[a-zA-Z_]\w*\b', problem)))
#         }

#     def _generate_reasoning_paths(self, problem: str, patterns: List[PatternSignature], context: Dict[str, Any]) -> List[ReasoningPath]:
#         """Generates multiple sophisticated reasoning paths."""
#         paths = [p for r_type in ReasoningType if (p := self._create_reasoning_path(problem, patterns, r_type, context))]
#         paths.extend(self._create_hybrid_reasoning_paths(paths, patterns))
#         return paths

#     def _create_reasoning_path(self, problem: str, patterns: List[PatternSignature], r_type: ReasoningType, context: Dict[str, Any]) -> Optional[ReasoningPath]:
#         """Creates a single, specific type of reasoning path."""
#         confidence = self._calculate_path_confidence(problem, r_type, patterns)
#         if confidence < 0.1: return None
#         return ReasoningPath(
#             path_id=f"{r_type.value}_{hash(problem) % 10000}",
#             reasoning_type=r_type,
#             mathematical_foundation=self._derive_mathematical_foundation(r_type, patterns),
#             logical_structure=self._construct_logical_structure(problem, r_type),
#             confidence_score=confidence, pattern_complexity=len(patterns),
#             epistemic_depth=self._calculate_epistemic_depth(r_type, patterns)
#         )
    
#     def _create_hybrid_reasoning_paths(self, base_paths: List[ReasoningPath], patterns: List[PatternSignature]) -> List[ReasoningPath]:
#         """Creates hybrid paths by combining two existing high-confidence paths."""
#         hybrid_paths = []
#         high_conf_paths = [p for p in base_paths if p.confidence_score > 0.6]
#         for p1, p2 in itertools.combinations(high_conf_paths, 2):
#             hybrid_type_name = f"hybrid_{p1.reasoning_type.value}_{p2.reasoning_type.value}"
#             HybridReasoningType = Enum('HybridReasoningType', {hybrid_type_name: hybrid_type_name})
#             hybrid_paths.append(ReasoningPath(
#                 path_id=f"hybrid_{p1.path_id}_{p2.path_id}",
#                 reasoning_type=HybridReasoningType[hybrid_type_name],
#                 mathematical_foundation=f"({p1.mathematical_foundation}) ∩ ({p2.mathematical_foundation})",
#                 logical_structure=f"Hybrid of [{p1.logical_structure}] and [{p2.logical_structure}]",
#                 confidence_score=min(1.0, (p1.confidence_score + p2.confidence_score) / 1.8),
#                 pattern_complexity=p1.pattern_complexity + p2.pattern_complexity,
#                 epistemic_depth=max(p1.epistemic_depth, p2.epistemic_depth) + 1
#             ))
#         return hybrid_paths

#     def _derive_mathematical_foundation(self, r_type: ReasoningType, patterns: List[PatternSignature]) -> str:
#         """Derives the mathematical foundation for a given reasoning type."""
#         foundations = {
#             ReasoningType.DEDUCTIVE: "Formal Logic: P → Q, P ⊢ Q", ReasoningType.INDUCTIVE: "Statistical Inference: P(H|E) = P(E|H)P(H)/P(E)",
#             ReasoningType.ABDUCTIVE: "Inference to Best Explanation: arg max P(H|E)", ReasoningType.ANALOGICAL: "Category Theory: Functor F: C → D",
#             ReasoningType.CAUSAL: "Causal Calculus: P(Y|do(X))", ReasoningType.PROBABILISTIC: "Bayesian Networks: P(X₁,...,Xₙ) = Π P(Xᵢ|Parents(Xᵢ))",
#             ReasoningType.COUNTERFACTUAL: "Possible Worlds Semantics", ReasoningType.MODAL: "Modal Logic: □P (necessarily P), ◊P (possibly P)",
#             ReasoningType.TEMPORAL: "Linear Temporal Logic: G(P) (always P), F(P) (eventually P)", ReasoningType.SPATIAL: "Mereotopology: Connectedness C(x,y)",
#             ReasoningType.METACOGNITIVE: "Gödel's Incompleteness Theorems", ReasoningType.DIALECTICAL: "Hegelian Dialectic: Thesis ⊕ Antithesis → Synthesis"
#         }
#         return foundations.get(r_type, "General Logic")

#     def _construct_logical_structure(self, problem: str, r_type: ReasoningType) -> str:
#         """Constructs a summary of the logical structure found in the text."""
#         premises = re.findall(r'(?:if|given|assume|suppose)\s+([^,.]+)', problem, re.IGNORECASE)
#         conclusions = re.findall(r'(?:then|therefore|thus|hence)\s+([^,.]+)', problem, re.IGNORECASE)
#         return f"Type: {r_type.value}, Premises: {len(premises)}, Conclusions: {len(conclusions)}"

#     def _calculate_path_confidence(self, problem: str, r_type: ReasoningType, patterns: List[PatternSignature]) -> float:
#         """Calculates a confidence score for a reasoning path."""
#         base = {
#             ReasoningType.DEDUCTIVE: 0.9, ReasoningType.INDUCTIVE: 0.7, ReasoningType.ABDUCTIVE: 0.6, ReasoningType.ANALOGICAL: 0.5,
#             ReasoningType.CAUSAL: 0.8, ReasoningType.PROBABILISTIC: 0.85, ReasoningType.COUNTERFACTUAL: 0.65, ReasoningType.MODAL: 0.75,
#             ReasoningType.TEMPORAL: 0.7, ReasoningType.SPATIAL: 0.7, ReasoningType.METACOGNITIVE: 0.4, ReasoningType.DIALECTICAL: 0.45
#         }
#         keyword_bonus = len(re.findall(r_type.value, problem, re.IGNORECASE)) * 0.1
#         pattern_bonus = len(patterns) * 0.05
#         return min(1.0, base.get(r_type, 0.3) + keyword_bonus + pattern_bonus)
        
#     def _calculate_epistemic_depth(self, r_type: ReasoningType, patterns: List[PatternSignature]) -> int:
#         """Calculates the epistemic depth (layers of knowledge needed)."""
#         base_depth = {
#             ReasoningType.DEDUCTIVE: 2, ReasoningType.INDUCTIVE: 3, ReasoningType.ABDUCTIVE: 3, ReasoningType.ANALOGICAL: 4,
#             ReasoningType.CAUSAL: 3, ReasoningType.PROBABILISTIC: 3, ReasoningType.COUNTERFACTUAL: 4, ReasoningType.MODAL: 4,
#             ReasoningType.TEMPORAL: 2, ReasoningType.SPATIAL: 2, ReasoningType.METACOGNITIVE: 5, ReasoningType.DIALECTICAL: 5
#         }
#         pattern_depth = sum(1 for p in patterns if "topology" in p.pattern_id or "recursion" in p.pattern_id)
#         return base_depth.get(r_type, 1) + pattern_depth

#     def _discover_exceptional_patterns(self, integrated_solution: Dict[str, Any]) -> Dict[str, Any]:
#         """Discovers patterns that only exceptional thinkers would notice."""
#         # This method looks for meta-patterns within the solution itself.
#         insights = {}
#         paths = integrated_solution.get("reasoning_paths", [])
#         if len(paths) > 2:
#             # Hidden Symmetries
#             for p1, p2 in itertools.combinations(paths, 2):
#                 if abs(p1['confidence_score'] - p2['confidence_score']) < 0.05 and p1['epistemic_depth'] == p2['epistemic_depth']:
#                     insights.setdefault("hidden_symmetries", []).append(f"Symmetry between {p1['reasoning_type']} and {p2['reasoning_type']} paths.")
#             # Non-obvious Invariants
#             ratios = [p['confidence_score'] / p['epistemic_depth'] for p in paths if p['epistemic_depth'] > 0]
#             if ratios and np.std(ratios) < 0.1:
#                 insights.setdefault("non_obvious_invariants", []).append(f"Invariant Found: Confidence/Depth ratio is nearly constant at ~{np.mean(ratios):.2f}")
#         return insights

#     def _synthesize_ultra_reasoning(self, problem: str, integrated_solution: Dict[str, Any], exceptional_insights: Dict[str, Any]) -> Dict[str, Any]:
#         """Synthesizes the final ultra-reasoning response."""
#         ultra_prompt = self._construct_ultra_reasoning_prompt(problem, integrated_solution, exceptional_insights)
#         response, latency, p_tokens, c_tokens = self._call_gemini_ultra_reasoning(ultra_prompt)
#         return {
#             "ultra_reasoning": response, "mathematical_proofs": self._generate_mathematical_proofs(integrated_solution),
#             "pattern_analysis": exceptional_insights, "reasoning_confidence": self._calculate_overall_confidence(integrated_solution),
#             "novelty_score": self._calculate_novelty_score(exceptional_insights), "latency": latency,
#             "token_usage": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens}
#         }

#     def _construct_ultra_reasoning_prompt(self, problem: str, integrated_solution: Dict[str, Any], exceptional_insights: Dict[str, Any]) -> str:
#         """Constructs the final, complex prompt for the generative model."""
#         # Sanitize solution for the prompt by removing unserializable objects
#         integrated_solution['reasoning_paths'] = [
#             {k: (v.value if isinstance(v, Enum) else v) for k, v in p.items()} 
#             for p in integrated_solution['reasoning_paths']
#         ]
#         return f"""
# You are an ultra-advanced reasoning system with 200+ IQ cognitive capabilities. Analyze the provided problem and data to produce a profound, multi-layered synthesis.

# PROBLEM:
# {problem}

# PRE-COMPUTED ANALYSIS:
# - Integrated Solution (reasoning paths, symbolic analysis): {json.dumps(integrated_solution, indent=2, default=str)}
# - Exceptional Patterns (hidden connections, paradoxes): {json.dumps(exceptional_insights, indent=2, default=str)}

# YOUR TASK:
# Synthesize these findings into a coherent, genius-level explanation. Structure your response precisely as follows:

# 1.  **EXECUTIVE SUMMARY**: A single, profound sentence capturing the absolute core insight.
# 2.  **MATHEMATICAL ANALYSIS**: A rigorous mathematical treatment of the problem's core structure. Use LaTeX for formulas (e.g., $E=mc^2$).
# 3.  **PATTERN SYNTHESIS**: Explain how the identified patterns integrate to form a larger picture.
# 4.  **EXCEPTIONAL INSIGHTS**: Elaborate on the non-obvious insights. Explain *why* these are significant.
# 5.  **BROADER IMPLICATIONS**: Connect the solution to universal principles in science, mathematics, or philosophy.
# 6.  **NOVEL HYPOTHESES**: Propose a new, testable hypothesis based on your analysis.

# Adhere strictly to this format. Your reasoning must be deep, precise, and reveal connections that ordinary analysis would miss.
# """

#     def _call_gemini_ultra_reasoning(self, prompt: str) -> Tuple[str, float, int, int]:
#         """Calls the generative model with the ultra-reasoning prompt."""
#         if not self.model: return "Error: Model not initialized.", 0.0, 0, 0
#         try:
#             start_time = time.time()
#             response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.2, max_output_tokens=8192))
#             latency = time.time() - start_time
#             usage = response.usage_metadata
#             return response.text, latency, usage.prompt_token_count, usage.candidates_token_count
#         except Exception as e:
#             print(f"Ultra-reasoning API Error: {e}")
#             return f"Ultra-reasoning Error: {e}", 0.0, 0, 0

#     def _generate_mathematical_proofs(self, integrated_solution: Dict[str, Any]) -> List[str]:
#         """Generates formal proofs for key insights based on symbolic analysis."""
#         proofs = []
#         for expr, analysis in integrated_solution.get("symbolic_insights", {}).items():
#             if 'simplified' in analysis and expr != analysis['simplified']:
#                 proofs.append(f"Proof of Simplification for '{expr}':\n1. Original: {expr}\n2. Simplified: {analysis['simplified']}\n(Proof via symbolic algebra engine).")
#         return proofs

#     def _calculate_overall_confidence(self, integrated_solution: Dict[str, Any]) -> float:
#         """Calculates a single confidence score for the final result."""
#         paths = integrated_solution.get("reasoning_paths", [])
#         if not paths: return 0.0
#         avg_confidence = sum(p['confidence_score'] for p in paths) / len(paths)
#         return (avg_confidence * 0.7) + (integrated_solution.get("integration_score", 0.0) * 0.3)

#     def _calculate_novelty_score(self, exceptional_insights: Dict[str, Any]) -> float:
#         """Scores how novel or surprising the exceptional insights are."""
#         if not exceptional_insights: return 0.0
#         weights = {"hidden_symmetries": 0.2, "non_obvious_invariants": 0.35, "emergent_properties": 0.25}
#         score = sum(weights[k] for k, v in exceptional_insights.items() if v and k in weights)
#         return min(1.0, score)

# # --- Agent and API Endpoints ---
# class AdvancedLLMAgent:
#     """The agent class that interfaces with the reasoning engine."""
#     def __init__(self, model_name: str = "gemini-1.5-pro"):
#         self.reasoning_engine = AdvancedReasoningEngine(model_name)
#         if API_KEY:
#             self.model = genai.GenerativeModel(model_name)
#         else:
#             self.model = None
        
#     def get_ultra_cognition(self, task_prompt: str, mode: str, reasoning_framework: dict) -> Dict[str, Any]:
#         """Gets ultra-advanced cognition with mathematical-linguistic reasoning."""
#         if mode == "reasoning":
#             context = {"framework": reasoning_framework, "mode": mode, "timestamp": time.time()}
#             return self.reasoning_engine.ultra_reasoning(task_prompt, context)
#         else:
#             return self._standard_reasoning(task_prompt)
            
#     def _standard_reasoning(self, task_prompt: str) -> Dict[str, Any]:
#         """Handles standard reasoning for non-ultra modes."""
#         if not self.model: return {"response": "Error: Model not initialized.", "latency": 0.0}
#         prompt = f"Provide a concise, intuitive answer to the following: {task_prompt}"
#         try:
#             start_time = time.time()
#             response = self.model.generate_content(prompt)
#             return {"response": response.text, "latency": time.time() - start_time}
#         except Exception as e:
#             return {"response": f"Error: {e}", "latency": 0.0}

# agent = AdvancedLLMAgent()

# @app.route('/reason', methods=['POST'])
# def reason():
#     """Flask endpoint to handle reasoning requests."""
#     data = request.get_json()
#     if not data or 'prompt' not in data:
#         return jsonify({"error": "Invalid request. 'prompt' is required."}), 400
    
#     try:
#         result = agent.get_ultra_cognition(
#             task_prompt=data['prompt'],
#             mode=data.get('mode', 'reasoning'),
#             reasoning_framework=data.get('framework', {})
#         )
#         # Convert enums to strings for JSON serialization if any remain
#         return jsonify(json.loads(json.dumps(result, default=str)))
#     except Exception as e:
#         print(f"API Error: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     # To run this Flask app:
#     # 1. pip install Flask flask-cors google-generativeai numpy networkx sympy
#     # 2. Set your API key in the script or as an environment variable.
#     # 3. Run this script: python your_script_name.py
#     # 4. Send POST requests to http://127.0.0.1:5000/reason
#     app.run(debug=True, port=5000)


import time
import random
import os
import re
import google.generativeai as genai
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict
import sympy as sp
from sympy import symbols, solve, diff, integrate, simplify
from sympy.logic import simplify_logic
import itertools
import zlib

# --- Configuration ---
# It is recommended to use environment variables for API keys for security.
API_KEY = "AIzaSyBLXXuiqpx9BfDxGi28Ci8szlsb3qAm9Dw"
# API_KEY = os.environ.get("GEMINI_API_KEY") 
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("API Key not found. Please set the GEMINI_API_KEY environment variable.")


app = Flask(__name__)
CORS(app)

# --- Advanced Reasoning Types ---
class ReasoningType(Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    PROBABILISTIC = "probabilistic"
    COUNTERFACTUAL = "counterfactual"
    MODAL = "modal"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    METACOGNITIVE = "metacognitive"
    DIALECTICAL = "dialectical"

# --- NEW: Legal Reasoning Types ---
class LegalReasoningType(Enum):
    STATUTORY_INTERPRETATION = "statutory_interpretation"
    PRECEDENT_ANALYSIS = "precedent_analysis" # Stare decisis
    POLICY_BASED = "policy_based"
    TEXTUALISM = "textualism"
    ORIGINALISM = "originalism"
    PURPOSIVISM = "purposivism"
    BALANCING_TEST = "balancing_test"

@dataclass
class ReasoningPath:
    """Represents a single reasoning path with mathematical or legal backing"""
    path_id: str
    reasoning_type: Any
    foundation: str # Renamed from mathematical_foundation for generality
    logical_structure: str
    confidence_score: float
    pattern_complexity: int
    epistemic_depth: int

@dataclass
class PatternSignature:
    """Mathematical signature of discovered patterns"""
    pattern_id: str
    mathematical_expression: str
    symmetry_group: str = "N/A"
    invariants: List[str] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)
    information_content: float = 0.0

# --- NEW: Legal Pattern Signature ---
@dataclass
class LegalPatternSignature:
    """Signature of discovered legal patterns or principles"""
    pattern_id: str
    legal_principle: str # e.g., "stare decisis", "mens rea"
    source_citation: str # e.g., "Marbury v. Madison"
    applicability_score: float

# --- Supporting Classes ---

class SymbolicReasoningEngine:
    """Advanced symbolic mathematical reasoning"""
    # ... (existing class remains the same)
    def __init__(self):
        self.symbol_registry = {}
        
    def analyze_symbolically(self, problem: str, patterns: List[PatternSignature]) -> Dict[str, Any]:
        expressions = self._extract_mathematical_expressions(problem)
        for p in patterns:
            if p.mathematical_expression:
                expressions.append(p.mathematical_expression)
        
        symbolic_results = {}
        for expr in set(expressions):
            try:
                parsed = sp.sympify(expr, locals=self.symbol_registry)
                simplified = simplify(parsed)
                
                if parsed.free_symbols:
                    derivatives = {str(var): str(diff(parsed, var)) for var in parsed.free_symbols}
                    integrals = {str(var): str(integrate(parsed, var)) for var in parsed.free_symbols}
                    
                    symbolic_results[expr] = {
                        "simplified": str(simplified),
                        "derivatives": derivatives,
                        "integrals": integrals,
                        "free_symbols": [str(s) for s in parsed.free_symbols]
                    }
            except (sp.SympifyError, TypeError, SyntaxError) as e:
                print(f"Symbolic analysis error for '{expr}': {e}")
        return symbolic_results
    
    def _extract_mathematical_expressions(self, problem: str) -> List[str]:
        patterns = [
            r'[a-zA-Z]+\s*=\s*[0-9a-zA-Z\+\-\*/\^\(\)\s\.,_]+',
            r'[a-zA-Z]\([a-zA-Z0-9\,\s]+\)\s*=\s*.+',
            r'\\frac\{[^}]+\}\{[^}]+\}',
            r'\b[a-zA-Z]\w*\^[0-9.]+\b',
            r'\b(log|sin|cos|tan|exp)\([^)]+\)',
            r'\b\d+\s*[\+\-\*/]\s*\d+\b'
        ]
        expressions = []
        for pattern in patterns:
            try:
                expressions.extend(re.findall(pattern, problem))
            except re.error as e:
                print(f"Regex error with pattern '{pattern}': {e}")
        return expressions

# --- NEW: Legal Symbolic Engine ---
class LegalSymbolicEngine:
    """Handles reasoning over abstract legal principles and doctrines."""
    def __init__(self):
        # A mock database of legal precedents and statutes
        self.legal_db = {
            "Marbury v. Madison": {"principle": "Judicial Review", "holding": "Establishes the principle of judicial review."},
            "Brown v. Board": {"principle": "Equal Protection", "holding": "State-sanctioned segregation in public schools is unconstitutional."},
            "U.S. Constitution, Art. I": {"principle": "Legislative Power", "text": "All legislative Powers herein granted shall be vested in a Congress..."},
        }
    
    def analyze_legal_principles(self, problem: str) -> List[LegalPatternSignature]:
        """Identifies relevant legal doctrines, statutes, and precedents."""
        patterns = []
        # Search for citations in the text
        for citation, data in self.legal_db.items():
            if re.search(citation, problem, re.IGNORECASE):
                patterns.append(LegalPatternSignature(
                    pattern_id=f"citation_{citation.replace(' ', '_')}",
                    legal_principle=data["principle"],
                    source_citation=citation,
                    applicability_score=0.9 # High score if directly cited
                ))
        
        # Search for legal keywords
        if "duty of care" in problem:
            patterns.append(LegalPatternSignature("principle_duty_of_care", "Duty of Care", "Torts Common Law", 0.7))
        if "mens rea" in problem or "guilty mind" in problem:
            patterns.append(LegalPatternSignature("principle_mens_rea", "Mens Rea", "Criminal Law Doctrine", 0.8))
            
        return patterns

class MetaCognitiveMonitor:
    # ... (existing class remains the same)
    def optimize_reasoning(self, paths: List[ReasoningPath]) -> List[ReasoningPath]:
        sorted_paths = sorted(paths, 
                              key=lambda p: (p.confidence_score * 0.5 + p.epistemic_depth * 0.3 + p.pattern_complexity * 0.2), 
                              reverse=True)
        return [path for path in sorted_paths if self._passes_meta_cognitive_filter(path)]
    
    def _passes_meta_cognitive_filter(self, path: ReasoningPath) -> bool:
        if path.confidence_score < 0.3 or path.epistemic_depth < 2:
            return False
        return self._check_logical_consistency(path)
    
    def _check_logical_consistency(self, path: ReasoningPath) -> bool:
        if path.reasoning_type == ReasoningType.DIALECTICAL:
            return "P ∧ ¬P" in path.foundation
        return True

class PatternDetector:
    # ... (existing class remains the same)
    def detect_deep_patterns(self, problem: str, problem_structure: Dict[str, Any]) -> List[PatternSignature]:
        patterns = []
        patterns.extend(self._detect_symmetries(problem))
        patterns.extend(self._detect_recursive_patterns(problem))
        patterns.extend(self._detect_scaling_patterns(problem))
        patterns.extend(self._detect_topological_patterns(problem_structure))
        return patterns
    
    def _detect_symmetries(self, problem: str) -> List[PatternSignature]:
        matches = re.findall(r'(\w+)\s*=\s*(\w+)', problem)
        symmetries = []
        for a, b in matches:
            if f"{b} = {a}" in problem or f"{b}=={a}" in problem:
                symmetries.append(PatternSignature(pattern_id=f"symmetry_{a}_{b}", mathematical_expression=f"{a}={b} <=> {b}={a}", symmetry_group="Z2"))
        return symmetries
    
    def _detect_recursive_patterns(self, problem: str) -> List[PatternSignature]:
        matches = re.findall(r'(\w+)\(n\)\s*=\s*.*?\1\(n\s*-\s*1\)', problem)
        return [PatternSignature(pattern_id=f"recursion_{m}", mathematical_expression=f"{m}(n) = f({m}(n-1))") for m in matches]
    
    def _detect_scaling_patterns(self, problem: str) -> List[PatternSignature]:
        matches = re.findall(r'(\w+)\s*(?:is proportional to|scales with)\s*(\w+)\^(\w+)', problem)
        return [PatternSignature(pattern_id=f"scaling_{y}_{x}", mathematical_expression=f"{y} ∝ {x}^{a}") for y, x, a in matches]
    
    def _detect_topological_patterns(self, problem_structure: Dict[str, Any]) -> List[PatternSignature]:
        patterns = []
        topology = problem_structure.get("topology", {})
        if topology.get("clustering", 0) > 0.5 and topology.get("nodes", 0) > 10:
            patterns.append(PatternSignature(pattern_id="high_clustering_topology", mathematical_expression="High Clustering Coefficient C > 0.5"))
        if topology.get("components", 1) > 1:
            patterns.append(PatternSignature(pattern_id="disconnected_topology", mathematical_expression="Disconnected Concept Graph"))
        return patterns

class CrossModalIntegrator:
    # ... (existing class remains the same, but logic will handle new types)
    def integrate_insights(self, reasoning_paths: List[ReasoningPath], symbolic_insights: Dict[str, Any], patterns: List[Any]) -> Dict[str, Any]:
        return {
            "reasoning_paths": [path.__dict__ for path in reasoning_paths],
            "symbolic_insights": symbolic_insights,
            "patterns": [p.__dict__ for p in patterns],
            "integration_score": self._calculate_integration_score(reasoning_paths, symbolic_insights, patterns)
        }
    
    def _calculate_integration_score(self, reasoning_paths: List[ReasoningPath], symbolic_insights: Dict[str, Any], patterns: List[Any]) -> float:
        if not reasoning_paths: return 0.0
        path_score = sum(p.confidence_score for p in reasoning_paths) / len(reasoning_paths)
        
        math_patterns = [p for p in patterns if isinstance(p, PatternSignature)]
        symbolic_validation_count = sum(1 for p in math_patterns if p.mathematical_expression in symbolic_insights)
        symbolic_score = symbolic_validation_count / len(math_patterns) if math_patterns else 0
        
        pattern_score = len(patterns) / 10.0
        return min(1.0, (path_score * 0.5 + symbolic_score * 0.3 + pattern_score * 0.2))

# --- Main Reasoning Engine (Updated) ---
class AdvancedReasoningEngine:
    """The main class that orchestrates advanced reasoning across multiple domains."""
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        if API_KEY:
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None
        # Domain-specific engines
        self.symbolic_engine = SymbolicReasoningEngine()
        self.legal_symbolic_engine = LegalSymbolicEngine() # NEW
        
        self.meta_cognitive_monitor = MetaCognitiveMonitor()
        self.pattern_detector = PatternDetector()
        self.cross_modal_integrator = CrossModalIntegrator()
        
    def ultra_reasoning(self, problem: str, context: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Orchestrates the ultra-advanced reasoning pipeline for a specific domain."""
        if not self.model:
            return {"error": "Gemini API key not configured."}

        # --- Domain-specific pipeline ---
        if domain == "legal":
            # Legal-specific analysis
            patterns = self.legal_symbolic_engine.analyze_legal_principles(problem)
            reasoning_paths = self._generate_reasoning_paths(problem, patterns, context, domain="legal")
            symbolic_insights = {} # No sympy for legal text
        else: # Default to mathematical/logical
            problem_structure = self._decompose_problem_mathematically(problem)
            patterns = self.pattern_detector.detect_deep_patterns(problem, problem_structure)
            reasoning_paths = self._generate_reasoning_paths(problem, patterns, context, domain="math")
            symbolic_insights = self.symbolic_engine.analyze_symbolically(problem, patterns)

        optimized_paths = self.meta_cognitive_monitor.optimize_reasoning(reasoning_paths)
        integrated_solution = self.cross_modal_integrator.integrate_insights(optimized_paths, symbolic_insights, patterns)
        
        return self._synthesize_ultra_reasoning(problem, integrated_solution, domain)

    def _generate_reasoning_paths(self, problem: str, patterns: List[Any], context: Dict[str, Any], domain: str) -> List[ReasoningPath]:
        """Generates reasoning paths for the specified domain."""
        paths = []
        reasoning_types = LegalReasoningType if domain == "legal" else ReasoningType
        
        for r_type in reasoning_types:
            path = self._create_reasoning_path(problem, patterns, r_type, context)
            if path:
                paths.append(path)
        return paths

    def _create_reasoning_path(self, problem: str, patterns: List[Any], r_type: Enum, context: Dict[str, Any]) -> Optional[ReasoningPath]:
        """Creates a single, specific type of reasoning path."""
        confidence = self._calculate_path_confidence(problem, r_type, patterns)
        if confidence < 0.1: return None
        return ReasoningPath(
            path_id=f"{r_type.value}_{hash(problem) % 10000}",
            reasoning_type=r_type,
            foundation=self._derive_foundation(r_type, patterns),
            logical_structure=self._construct_logical_structure(problem, r_type),
            confidence_score=confidence,
            pattern_complexity=len(patterns),
            epistemic_depth=self._calculate_epistemic_depth(r_type, patterns)
        )
    
    def _derive_foundation(self, r_type: Enum, patterns: List[Any]) -> str:
        """Derives the foundation for a given reasoning type (math or legal)."""
        # Mathematical Foundations
        math_foundations = {
            ReasoningType.DEDUCTIVE: "Formal Logic: P → Q, P ⊢ Q", ReasoningType.INDUCTIVE: "Statistical Inference",
            ReasoningType.ABDUCTIVE: "Inference to Best Explanation", ReasoningType.ANALOGICAL: "Category Theory",
            ReasoningType.CAUSAL: "Causal Calculus", ReasoningType.PROBABILISTIC: "Bayesian Networks",
            # ... and so on for all math types
        }
        # Legal Foundations
        legal_foundations = {
            LegalReasoningType.STATUTORY_INTERPRETATION: "Canons of Construction",
            LegalReasoningType.PRECEDENT_ANALYSIS: "Doctrine of Stare Decisis",
            LegalReasoningType.POLICY_BASED: "Law and Economics / Social Goals",
            LegalReasoningType.TEXTUALISM: "Plain Meaning Rule",
            LegalReasoningType.ORIGINALISM: "Original Public Meaning / Intent",
            LegalReasoningType.PURPOSIVISM: "Legislative Purpose Analysis",
            LegalReasoningType.BALANCING_TEST: "Proportionality Analysis (e.g., Strict Scrutiny)",
        }
        
        if isinstance(r_type, LegalReasoningType):
            return legal_foundations.get(r_type, "General Legal Principles")
        return math_foundations.get(r_type, "General Logic")

    def _calculate_path_confidence(self, problem: str, r_type: Enum, patterns: List[Any]) -> float:
        """Calculates confidence, sensitive to domain."""
        # Base confidence scores can be defined for legal types as well
        # This is a simplified placeholder
        keyword_bonus = len(re.findall(r_type.value.replace('_', ' '), problem, re.IGNORECASE)) * 0.2
        pattern_bonus = len(patterns) * 0.1
        return min(1.0, 0.5 + keyword_bonus + pattern_bonus)

    def _calculate_epistemic_depth(self, r_type: Enum, patterns: List[Any]) -> int:
        """Calculates epistemic depth, sensitive to domain."""
        # Can be defined for legal types (e.g., constitutional analysis is deeper than statutory)
        return 2 + len(patterns) # Simplified placeholder

    def _synthesize_ultra_reasoning(self, problem: str, integrated_solution: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Synthesizes the final ultra-reasoning response."""
        ultra_prompt = self._construct_ultra_reasoning_prompt(problem, integrated_solution, domain)
        response, latency, p_tokens, c_tokens = self._call_gemini_ultra_reasoning(ultra_prompt)
        return {
            "ultra_reasoning": response, "integrated_solution": integrated_solution,
            "latency": latency, "token_usage": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens}
        }

    def _construct_ultra_reasoning_prompt(self, problem: str, integrated_solution: Dict[str, Any], domain: str) -> str:
        """Constructs the final prompt, tailored to the reasoning domain."""
        # Sanitize solution for the prompt
        integrated_solution['reasoning_paths'] = [
            {k: (v.value if isinstance(v, Enum) else v) for k, v in p.items()} 
            for p in integrated_solution['reasoning_paths']
        ]
        domain_instructions = {
            "legal": "Focus on legal doctrines, precedential weight, and statutory interpretation. Structure as a legal memo.",
            "math": "Focus on mathematical rigor, formal proofs, and structural patterns. Structure as a scientific paper."
        }
        return f"""
You are an ultra-advanced reasoning system. Your task is to analyze the provided problem within the specified domain.

DOMAIN: {domain.upper()}
PROBLEM: {problem}
PRE-COMPUTED ANALYSIS: {json.dumps(integrated_solution, indent=2, default=str)}

YOUR TASK:
Synthesize these findings into a coherent, genius-level explanation. 
{domain_instructions[domain]}
Reveal connections that ordinary analysis would miss.
"""
    
    # --- Other existing methods like _call_gemini_ultra_reasoning, _decompose_problem_mathematically etc. remain here ---
    def _call_gemini_ultra_reasoning(self, prompt: str) -> Tuple[str, float, int, int]:
        if not self.model: return "Error: Model not initialized.", 0.0, 0, 0
        try:
            start_time = time.time()
            response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.2, max_output_tokens=8192))
            latency = time.time() - start_time
            usage = response.usage_metadata
            return response.text, latency, usage.prompt_token_count, usage.candidates_token_count
        except Exception as e:
            print(f"Ultra-reasoning API Error: {e}")
            return f"Ultra-reasoning Error: {e}", 0.0, 0, 0

    def _decompose_problem_mathematically(self, problem: str) -> Dict[str, Any]:
        concept_graph = self._build_concept_graph(problem)
        return {
            "entropy": self._calculate_semantic_entropy(problem),
            "complexity": self._calculate_kolmogorov_complexity(problem),
            "topology": self._analyze_concept_topology(concept_graph),
            "functional_components": self._extract_functional_components(problem),
            "algebraic_structure": self._identify_algebraic_structure(problem),
            "concept_graph": nx.node_link_data(concept_graph)
        }

    def _calculate_semantic_entropy(self, text: str) -> float:
        words = re.findall(r'\b\w+\b', text.lower())
        if not words: return 0.0
        freq_dist = defaultdict(int)
        for word in words: freq_dist[word] += 1
        entropy = 0.0
        total_words = len(words)
        for freq in freq_dist.values():
            prob = freq / total_words
            entropy -= prob * math.log2(prob)
        return entropy

    def _calculate_kolmogorov_complexity(self, text: str) -> float:
        if not text: return 0.0
        return len(zlib.compress(text.encode('utf-8'))) / len(text.encode('utf-8'))

    def _build_concept_graph(self, problem: str) -> nx.Graph:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', problem.lower())
        g = nx.Graph()
        if not words: return g
        for i, word in enumerate(words):
            g.add_node(word)
            if i > 0: g.add_edge(words[i-1], word)
        return g

    def _analyze_concept_topology(self, g: nx.Graph) -> Dict[str, Any]:
        if g.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0, "density": 0, "clustering": 0, "components": 0}
        return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), "density": nx.density(g),
                "clustering": nx.average_clustering(g), "components": nx.number_connected_components(g)}

    def _extract_functional_components(self, problem: str) -> List[str]:
        patterns = [r'\b\w+\([^)]*\)', r'if\s+.*?\s+then\s+.*']
        components = []
        for p in patterns: components.extend(re.findall(p, problem, re.IGNORECASE))
        return components

    def _identify_algebraic_structure(self, problem: str) -> Dict[str, Any]:
        return {'operations': {'addition': len(re.findall(r'\+', problem)), 'subtraction': len(re.findall(r'-', problem)),
                               'multiplication': len(re.findall(r'[\*×]', problem)), 'division': len(re.findall(r'[/÷]', problem)),
                               'equality': len(re.findall(r'=', problem))},
                'variables': len(set(re.findall(r'\b[a-zA-Z_]\w*\b', problem)))}


# --- Agent and API Endpoints (Updated) ---
class AdvancedLLMAgent:
    """The agent class that interfaces with the reasoning engine."""
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.reasoning_engine = AdvancedReasoningEngine(model_name)
        if API_KEY:
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None
        
    def get_ultra_cognition(self, task_prompt: str, mode: str, domain: str) -> Dict[str, Any]:
        """Gets ultra-advanced cognition, now with domain selection."""
        if mode == "reasoning":
            context = {"mode": mode, "timestamp": time.time()}
            return self.reasoning_engine.ultra_reasoning(task_prompt, context, domain)
        else:
            return self._standard_reasoning(task_prompt)
            
    def _standard_reasoning(self, task_prompt: str) -> Dict[str, Any]:
        if not self.model: return {"response": "Error: Model not initialized.", "latency": 0.0}
        prompt = f"Provide a concise, intuitive answer to the following: {task_prompt}"
        try:
            start_time = time.time()
            response = self.model.generate_content(prompt)
            return {"response": response.text, "latency": time.time() - start_time}
        except Exception as e:
            return {"response": f"Error: {e}", "latency": 0.0}

agent = AdvancedLLMAgent()

@app.route('/reason', methods=['POST'])
def reason():
    """Flask endpoint, now requires a 'domain' field."""
    data = request.get_json()
    if not data or 'prompt' not in data or 'domain' not in data:
        return jsonify({"error": "Invalid request. 'prompt' and 'domain' are required."}), 400
    
    if data['domain'] not in ['legal', 'math']:
        return jsonify({"error": "Invalid domain. Must be 'legal' or 'math'."}), 400

    try:
        result = agent.get_ultra_cognition(
            task_prompt=data['prompt'],
            mode=data.get('mode', 'reasoning'),
            domain=data['domain']
        )
        return jsonify(json.loads(json.dumps(result, default=str)))
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)


import time
import random
import os
import re
import google.generativeai as genai
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict
import sympy as sp
from sympy import symbols, solve, diff, integrate, simplify
from sympy.logic import simplify_logic
import itertools
import zlib

# --- Configuration ---
# It is recommended to use environment variables for API keys for security.
API_KEY = "AIzaSyBLXXuiqpx9BfDxGi28Ci8szlsb3qAm9Dw"
# API_KEY = os.environ.get("GEMINI_API_KEY") 
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("API Key not found. Please set the GEMINI_API_KEY environment variable.")


app = Flask(__name__)
CORS(app)

# --- Advanced Reasoning Types ---
class ReasoningType(Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    PROBABILISTIC = "probabilistic"
    COUNTERFACTUAL = "counterfactual"
    MODAL = "modal"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    METACOGNITIVE = "metacognitive"
    DIALECTICAL = "dialectical"

# --- NEW: Legal Reasoning Types ---
class LegalReasoningType(Enum):
    STATUTORY_INTERPRETATION = "statutory_interpretation"
    PRECEDENT_ANALYSIS = "precedent_analysis" # Stare decisis
    POLICY_BASED = "policy_based"
    TEXTUALISM = "textualism"
    ORIGINALISM = "originalism"
    PURPOSIVISM = "purposivism"
    BALANCING_TEST = "balancing_test"

@dataclass
class ReasoningPath:
    """Represents a single reasoning path with mathematical or legal backing"""
    path_id: str
    reasoning_type: Any
    foundation: str # Renamed from mathematical_foundation for generality
    logical_structure: str
    confidence_score: float
    pattern_complexity: int
    epistemic_depth: int

@dataclass
class PatternSignature:
    """Mathematical signature of discovered patterns"""
    pattern_id: str
    mathematical_expression: str
    symmetry_group: str = "N/A"
    invariants: List[str] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)
    information_content: float = 0.0

# --- NEW: Legal Pattern Signature ---
@dataclass
class LegalPatternSignature:
    """Signature of discovered legal patterns or principles"""
    pattern_id: str
    legal_principle: str # e.g., "stare decisis", "mens rea"
    source_citation: str # e.g., "Marbury v. Madison"
    applicability_score: float

# --- Supporting Classes ---

class SymbolicReasoningEngine:
    """Advanced symbolic mathematical reasoning"""
    def __init__(self):
        self.symbol_registry = {}
        
    def analyze_symbolically(self, problem: str, patterns: List[PatternSignature]) -> Dict[str, Any]:
        expressions = self._extract_mathematical_expressions(problem)
        for p in patterns:
            if p.mathematical_expression:
                expressions.append(p.mathematical_expression)
        
        symbolic_results = {}
        for expr in set(expressions):
            try:
                parsed = sp.sympify(expr, locals=self.symbol_registry)
                simplified = simplify(parsed)
                
                if parsed.free_symbols:
                    derivatives = {str(var): str(diff(parsed, var)) for var in parsed.free_symbols}
                    integrals = {str(var): str(integrate(parsed, var)) for var in parsed.free_symbols}
                    
                    symbolic_results[expr] = {
                        "simplified": str(simplified),
                        "derivatives": derivatives,
                        "integrals": integrals,
                        "free_symbols": [str(s) for s in parsed.free_symbols]
                    }
            except (sp.SympifyError, TypeError, SyntaxError) as e:
                print(f"Symbolic analysis error for '{expr}': {e}")
        return symbolic_results
    
    def _extract_mathematical_expressions(self, problem: str) -> List[str]:
        patterns = [
            r'[a-zA-Z]+\s*=\s*[0-9a-zA-Z\+\-\*/\^\(\)\s\.,_]+',
            r'[a-zA-Z]\([a-zA-Z0-9\,\s]+\)\s*=\s*.+',
            r'\\frac\{[^}]+\}\{[^}]+\}',
            r'\b[a-zA-Z]\w*\^[0-9.]+\b',
            r'\b(log|sin|cos|tan|exp)\([^)]+\)',
            r'\b\d+\s*[\+\-\*/]\s*\d+\b'
        ]
        expressions = []
        for pattern in patterns:
            try:
                expressions.extend(re.findall(pattern, problem))
            except re.error as e:
                print(f"Regex error with pattern '{pattern}': {e}")
        return expressions

class LegalSymbolicEngine:
    """Handles reasoning over abstract legal principles and doctrines."""
    def __init__(self):
        # A mock database of legal precedents and statutes
        self.legal_db = {
            "Marbury v. Madison": {"principle": "Judicial Review", "holding": "Establishes the principle of judicial review."},
            "Brown v. Board of Education": {"principle": "Equal Protection", "holding": "State-sanctioned segregation in public schools is unconstitutional."},
            "Miranda v. Arizona": {"principle": "Right to Counsel", "holding": "Criminal suspects must be informed of their right to an attorney."},
            "U.S. Constitution, Art. I": {"principle": "Legislative Power", "text": "All legislative Powers herein granted shall be vested in a Congress..."},
            "U.S. Constitution, Amend. XIV": {"principle": "Equal Protection Clause", "text": "No State shall... deny to any person within its jurisdiction the equal protection of the laws."}
        }
    
    def analyze_legal_principles(self, problem: str) -> List[LegalPatternSignature]:
        """Identifies relevant legal doctrines, statutes, and precedents."""
        patterns = []
        # Search for citations in the text
        for citation, data in self.legal_db.items():
            # Handle different citation formats e.g. "Brown v. Board"
            short_citation = re.sub(r' of Education', '', citation)
            if re.search(citation, problem, re.IGNORECASE) or (citation != short_citation and re.search(short_citation, problem, re.IGNORECASE)):
                patterns.append(LegalPatternSignature(
                    pattern_id=f"citation_{citation.replace(' ', '_')}",
                    legal_principle=data["principle"],
                    source_citation=citation,
                    applicability_score=0.9 # High score if directly cited
                ))
        
        # Search for legal keywords using word boundaries for precision
        legal_keywords = {
            r"\bduty of care\b": ("Duty of Care", "Torts Common Law", 0.7),
            r"\bmens rea\b": ("Mens Rea", "Criminal Law Doctrine", 0.8),
            r"\bguilty mind\b": ("Mens Rea", "Criminal Law Doctrine", 0.8),
            r"\bstrict scrutiny\b": ("Strict Scrutiny", "Constitutional Law Test", 0.85),
            r"\bequal protection\b": ("Equal Protection", "14th Amendment", 0.8)
        }
        for keyword_regex, (principle, source, score) in legal_keywords.items():
             if re.search(keyword_regex, problem, re.IGNORECASE):
                # Avoid adding duplicate principles
                if not any(p.legal_principle == principle for p in patterns):
                    patterns.append(LegalPatternSignature(f"principle_{principle.replace(' ', '_')}", principle, source, score))
            
        return patterns

class MetaCognitiveMonitor:
    def optimize_reasoning(self, paths: List[ReasoningPath]) -> List[ReasoningPath]:
        sorted_paths = sorted(paths, 
                              key=lambda p: (p.confidence_score * 0.5 + p.epistemic_depth * 0.3 + p.pattern_complexity * 0.2), 
                              reverse=True)
        return [path for path in sorted_paths if self._passes_meta_cognitive_filter(path)]
    
    def _passes_meta_cognitive_filter(self, path: ReasoningPath) -> bool:
        if path.confidence_score < 0.3 or path.epistemic_depth < 2:
            return False
        return self._check_logical_consistency(path)
    
    def _check_logical_consistency(self, path: ReasoningPath) -> bool:
        if path.reasoning_type == ReasoningType.DIALECTICAL:
            return "P ∧ ¬P" in path.foundation
        return True

class PatternDetector:
    def detect_deep_patterns(self, problem: str, problem_structure: Dict[str, Any]) -> List[PatternSignature]:
        patterns = []
        patterns.extend(self._detect_symmetries(problem))
        patterns.extend(self._detect_recursive_patterns(problem))
        patterns.extend(self._detect_scaling_patterns(problem))
        patterns.extend(self._detect_topological_patterns(problem_structure))
        return patterns
    
    def _detect_symmetries(self, problem: str) -> List[PatternSignature]:
        matches = re.findall(r'(\w+)\s*=\s*(\w+)', problem)
        symmetries = []
        for a, b in matches:
            if f"{b} = {a}" in problem or f"{b}=={a}" in problem:
                symmetries.append(PatternSignature(pattern_id=f"symmetry_{a}_{b}", mathematical_expression=f"{a}={b} <=> {b}={a}", symmetry_group="Z2"))
        return symmetries
    
    def _detect_recursive_patterns(self, problem: str) -> List[PatternSignature]:
        matches = re.findall(r'(\w+)\(n\)\s*=\s*.*?\1\(n\s*-\s*1\)', problem)
        return [PatternSignature(pattern_id=f"recursion_{m}", mathematical_expression=f"{m}(n) = f({m}(n-1))") for m in matches]
    
    def _detect_scaling_patterns(self, problem: str) -> List[PatternSignature]:
        matches = re.findall(r'(\w+)\s*(?:is proportional to|scales with)\s*(\w+)\^(\w+)', problem)
        return [PatternSignature(pattern_id=f"scaling_{y}_{x}", mathematical_expression=f"{y} ∝ {x}^{a}") for y, x, a in matches]
    
    def _detect_topological_patterns(self, problem_structure: Dict[str, Any]) -> List[PatternSignature]:
        patterns = []
        topology = problem_structure.get("topology", {})
        if topology.get("clustering", 0) > 0.5 and topology.get("nodes", 0) > 10:
            patterns.append(PatternSignature(pattern_id="high_clustering_topology", mathematical_expression="High Clustering Coefficient C > 0.5"))
        if topology.get("components", 1) > 1:
            patterns.append(PatternSignature(pattern_id="disconnected_topology", mathematical_expression="Disconnected Concept Graph"))
        return patterns

class CrossModalIntegrator:
    def integrate_insights(self, reasoning_paths: List[ReasoningPath], symbolic_insights: Dict[str, Any], patterns: List[Any]) -> Dict[str, Any]:
        return {
            "reasoning_paths": [path.__dict__ for path in reasoning_paths],
            "symbolic_insights": symbolic_insights,
            "patterns": [p.__dict__ for p in patterns],
            "integration_score": self._calculate_integration_score(reasoning_paths, symbolic_insights, patterns)
        }
    
    def _calculate_integration_score(self, reasoning_paths: List[ReasoningPath], symbolic_insights: Dict[str, Any], patterns: List[Any]) -> float:
        if not reasoning_paths: return 0.0
        path_score = sum(p.confidence_score for p in reasoning_paths) / len(reasoning_paths)
        
        math_patterns = [p for p in patterns if isinstance(p, PatternSignature)]
        symbolic_validation_count = sum(1 for p in math_patterns if p.mathematical_expression in symbolic_insights)
        symbolic_score = symbolic_validation_count / len(math_patterns) if math_patterns else 0
        
        pattern_score = len(patterns) / 10.0
        return min(1.0, (path_score * 0.5 + symbolic_score * 0.3 + pattern_score * 0.2))

# --- Main Reasoning Engine (Updated) ---
class AdvancedReasoningEngine:
    """The main class that orchestrates advanced reasoning across multiple domains."""
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        if API_KEY:
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None
        self.symbolic_engine = SymbolicReasoningEngine()
        self.legal_symbolic_engine = LegalSymbolicEngine()
        self.meta_cognitive_monitor = MetaCognitiveMonitor()
        self.pattern_detector = PatternDetector()
        self.cross_modal_integrator = CrossModalIntegrator()
        
    def ultra_reasoning(self, problem: str, context: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Orchestrates the ultra-advanced reasoning pipeline for a specific domain."""
        if not self.model:
            return {"error": "Gemini API key not configured."}

        if domain == "legal":
            patterns = self.legal_symbolic_engine.analyze_legal_principles(problem)
            reasoning_paths = self._generate_reasoning_paths(problem, patterns, context, domain="legal")
            symbolic_insights = {}
            exceptional_insights = self._discover_exceptional_legal_patterns(reasoning_paths, patterns)
        else: # Default to mathematical/logical
            problem_structure = self._decompose_problem_mathematically(problem)
            patterns = self.pattern_detector.detect_deep_patterns(problem, problem_structure)
            reasoning_paths = self._generate_reasoning_paths(problem, patterns, context, domain="math")
            symbolic_insights = self.symbolic_engine.analyze_symbolically(problem, patterns)
            exceptional_insights = self._discover_exceptional_math_patterns(reasoning_paths, problem_structure)

        optimized_paths = self.meta_cognitive_monitor.optimize_reasoning(reasoning_paths)
        integrated_solution = self.cross_modal_integrator.integrate_insights(optimized_paths, symbolic_insights, patterns)
        
        return self._synthesize_ultra_reasoning(problem, integrated_solution, exceptional_insights, domain)

    def _generate_reasoning_paths(self, problem: str, patterns: List[Any], context: Dict[str, Any], domain: str) -> List[ReasoningPath]:
        """Generates reasoning paths for the specified domain."""
        paths = []
        reasoning_types = LegalReasoningType if domain == "legal" else ReasoningType
        
        for r_type in reasoning_types:
            path = self._create_reasoning_path(problem, patterns, r_type, context)
            if path:
                paths.append(path)
        return paths

    def _create_reasoning_path(self, problem: str, patterns: List[Any], r_type: Enum, context: Dict[str, Any]) -> Optional[ReasoningPath]:
        """Creates a single, specific type of reasoning path."""
        confidence = self._calculate_path_confidence(problem, r_type, patterns)
        if confidence < 0.1: return None
        return ReasoningPath(
            path_id=f"{r_type.value}_{hash(problem) % 10000}",
            reasoning_type=r_type,
            foundation=self._derive_foundation(r_type, patterns),
            logical_structure=self._construct_logical_structure(problem, r_type),
            confidence_score=confidence,
            pattern_complexity=len(patterns),
            epistemic_depth=self._calculate_epistemic_depth(r_type, patterns)
        )
    
    # FIXED: Added the missing _construct_logical_structure method
    def _construct_logical_structure(self, problem: str, r_type: Enum) -> str:
        """Constructs a summary of the logical structure found in the text."""
        premises = re.findall(r'(?:if|given|assume|suppose)\s+([^,.]+)', problem, re.IGNORECASE)
        conclusions = re.findall(r'(?:then|therefore|thus|hence)\s+([^,.]+)', problem, re.IGNORECASE)
        return f"Type: {r_type.value}, Premises: {len(premises)}, Conclusions: {len(conclusions)}"

    def _derive_foundation(self, r_type: Enum, patterns: List[Any]) -> str:
        """Derives the foundation for a given reasoning type (math or legal)."""
        math_foundations = {
            ReasoningType.DEDUCTIVE: "Formal Logic: P → Q, P ⊢ Q", ReasoningType.INDUCTIVE: "Statistical Inference",
            ReasoningType.ABDUCTIVE: "Inference to Best Explanation", ReasoningType.ANALOGICAL: "Category Theory",
            ReasoningType.CAUSAL: "Causal Calculus", ReasoningType.PROBABILISTIC: "Bayesian Networks",
            ReasoningType.COUNTERFACTUAL: "Possible Worlds Semantics", ReasoningType.MODAL: "Modal Logic: □P, ◊P",
            ReasoningType.TEMPORAL: "Linear Temporal Logic: G(P), F(P)", ReasoningType.SPATIAL: "Mereotopology: C(x,y)",
            ReasoningType.METACOGNITIVE: "Gödel's Incompleteness Theorems", ReasoningType.DIALECTICAL: "Hegelian Dialectic: T ⊕ A → S"
        }
        legal_foundations = {
            LegalReasoningType.STATUTORY_INTERPRETATION: "Canons of Construction",
            LegalReasoningType.PRECEDENT_ANALYSIS: "Doctrine of Stare Decisis",
            LegalReasoningType.POLICY_BASED: "Law and Economics / Social Goals",
            LegalReasoningType.TEXTUALISM: "Plain Meaning Rule",
            LegalReasoningType.ORIGINALISM: "Original Public Meaning / Intent",
            LegalReasoningType.PURPOSIVISM: "Legislative Purpose Analysis",
            LegalReasoningType.BALANCING_TEST: "Proportionality Analysis (e.g., Strict Scrutiny)",
        }
        
        if isinstance(r_type, LegalReasoningType):
            return legal_foundations.get(r_type, "General Legal Principles")
        return math_foundations.get(r_type, "General Logic")

    def _calculate_path_confidence(self, problem: str, r_type: Enum, patterns: List[Any]) -> float:
        """Calculates confidence, sensitive to domain."""
        keyword_bonus = len(re.findall(r'\b' + r_type.value.replace('_', r'\s') + r'\b', problem, re.IGNORECASE)) * 0.2
        pattern_bonus = len(patterns) * 0.1
        return min(1.0, 0.5 + keyword_bonus + pattern_bonus)

    def _calculate_epistemic_depth(self, r_type: Enum, patterns: List[Any]) -> int:
        """Calculates epistemic depth, sensitive to domain."""
        return 2 + len(patterns)

    def _discover_exceptional_math_patterns(self, paths: List[ReasoningPath], structure: Dict[str, Any]) -> Dict[str, Any]:
        """Discovers meta-patterns in the mathematical analysis."""
        insights = {}
        if len(paths) > 2:
            ratios = [p.confidence_score / p.epistemic_depth for p in paths if p.epistemic_depth > 0]
            if ratios and np.std(ratios) < 0.1:
                insights["invariant_ratio"] = f"Confidence/Depth ratio is invariant at ~{np.mean(ratios):.2f}"
        return insights

    def _discover_exceptional_legal_patterns(self, paths: List[ReasoningPath], patterns: List[LegalPatternSignature]) -> Dict[str, Any]:
        """Discovers meta-patterns in the legal analysis, like conflicting doctrines."""
        insights = {}
        path_types = {p.reasoning_type for p in paths}
        if LegalReasoningType.TEXTUALISM in path_types and LegalReasoningType.PURPOSIVISM in path_types:
            insights["doctrinal_conflict"] = "Conflict detected between Textualism and Purposivism paths. This is a core judicial tension."
        
        cited_principles = {p.legal_principle for p in patterns}
        if "Equal Protection" in cited_principles and "Legislative Power" in cited_principles:
             insights["principle_tension"] = "Tension identified between the principles of Equal Protection and Legislative Power."
        return insights

    def _synthesize_ultra_reasoning(self, problem: str, integrated_solution: Dict[str, Any], exceptional_insights: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Synthesizes the final ultra-reasoning response."""
        ultra_prompt = self._construct_ultra_reasoning_prompt(problem, integrated_solution, exceptional_insights, domain)
        response, latency, p_tokens, c_tokens = self._call_gemini_ultra_reasoning(ultra_prompt)
        return {
            "ultra_reasoning": response, "integrated_solution": integrated_solution,
            "exceptional_insights": exceptional_insights,
            "latency": latency, "token_usage": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens}
        }

    def _construct_ultra_reasoning_prompt(self, problem: str, integrated_solution: Dict[str, Any], exceptional_insights: Dict[str, Any], domain: str) -> str:
        """Constructs the final prompt, tailored to the reasoning domain."""
        integrated_solution['reasoning_paths'] = [
            {k: (v.value if isinstance(v, Enum) else v) for k, v in p.items()} 
            for p in integrated_solution['reasoning_paths']
        ]
        domain_instructions = {
            "legal": "Focus on legal doctrines, precedential weight, and statutory interpretation. Structure as a legal memo, identifying the core legal question, relevant rules, application, and conclusion (IRAC).",
            "math": "Focus on mathematical rigor, formal proofs, and structural patterns. Structure as a scientific paper, with an abstract, analysis, and conclusion."
        }
        return f"""
You are an ultra-advanced reasoning system. Your task is to analyze the provided problem within the specified domain.

DOMAIN: {domain.upper()}
PROBLEM: {problem}
PRE-COMPUTED ANALYSIS: {json.dumps(integrated_solution, indent=2, default=str)}
EXCEPTIONAL INSIGHTS: {json.dumps(exceptional_insights, indent=2, default=str)}

YOUR TASK:
Synthesize these findings into a coherent, genius-level explanation. 
{domain_instructions[domain]}
Reveal connections and tensions that ordinary analysis would miss.
"""
    
    def _call_gemini_ultra_reasoning(self, prompt: str) -> Tuple[str, float, int, int]:
        if not self.model: return "Error: Model not initialized.", 0.0, 0, 0
        try:
            start_time = time.time()
            response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.2, max_output_tokens=8192))
            latency = time.time() - start_time
            usage = response.usage_metadata
            return response.text, latency, usage.prompt_token_count, usage.candidates_token_count
        except Exception as e:
            print(f"Ultra-reasoning API Error: {e}")
            return f"Ultra-reasoning Error: {e}", 0.0, 0, 0

    def _decompose_problem_mathematically(self, problem: str) -> Dict[str, Any]:
        concept_graph = self._build_concept_graph(problem)
        return {
            "entropy": self._calculate_semantic_entropy(problem),
            "complexity": self._calculate_kolmogorov_complexity(problem),
            "topology": self._analyze_concept_topology(concept_graph),
            "functional_components": self._extract_functional_components(problem),
            "algebraic_structure": self._identify_algebraic_structure(problem),
            "concept_graph": nx.node_link_data(concept_graph)
        }

    def _calculate_semantic_entropy(self, text: str) -> float:
        words = re.findall(r'\b\w+\b', text.lower())
        if not words: return 0.0
        freq_dist = defaultdict(int)
        for word in words: freq_dist[word] += 1
        entropy = 0.0
        total_words = len(words)
        for freq in freq_dist.values():
            prob = freq / total_words
            entropy -= prob * math.log2(prob)
        return entropy

    def _calculate_kolmogorov_complexity(self, text: str) -> float:
        if not text: return 0.0
        return len(zlib.compress(text.encode('utf-8'))) / len(text.encode('utf-8'))

    def _build_concept_graph(self, problem: str) -> nx.Graph:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', problem.lower())
        g = nx.Graph()
        if not words: return g
        for i, word in enumerate(words):
            g.add_node(word)
            if i > 0: g.add_edge(words[i-1], word)
        return g

    def _analyze_concept_topology(self, g: nx.Graph) -> Dict[str, Any]:
        if g.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0, "density": 0, "clustering": 0, "components": 0}
        return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), "density": nx.density(g),
                "clustering": nx.average_clustering(g), "components": nx.number_connected_components(g)}

    def _extract_functional_components(self, problem: str) -> List[str]:
        patterns = [r'\b\w+\([^)]*\)', r'if\s+.*?\s+then\s+.*']
        components = []
        for p in patterns: components.extend(re.findall(p, problem, re.IGNORECASE))
        return components

    def _identify_algebraic_structure(self, problem: str) -> Dict[str, Any]:
        return {'operations': {'addition': len(re.findall(r'\+', problem)), 'subtraction': len(re.findall(r'-', problem)),
                               'multiplication': len(re.findall(r'[\*×]', problem)), 'division': len(re.findall(r'[/÷]', problem)),
                               'equality': len(re.findall(r'=', problem))},
                'variables': len(set(re.findall(r'\b[a-zA-Z_]\w*\b', problem)))}


# --- Agent and API Endpoints (Updated) ---
class AdvancedLLMAgent:
    """The agent class that interfaces with the reasoning engine."""
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.reasoning_engine = AdvancedReasoningEngine(model_name)
        if API_KEY:
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None
        
    def get_ultra_cognition(self, task_prompt: str, mode: str, domain: str) -> Dict[str, Any]:
        """Gets ultra-advanced cognition, now with domain selection."""
        if mode == "reasoning":
            context = {"mode": mode, "timestamp": time.time()}
            return self.reasoning_engine.ultra_reasoning(task_prompt, context, domain)
        else:
            return self._standard_reasoning(task_prompt)
            
    def _standard_reasoning(self, task_prompt: str) -> Dict[str, Any]:
        if not self.model: return {"response": "Error: Model not initialized.", "latency": 0.0}
        prompt = f"Provide a concise, intuitive answer to the following: {task_prompt}"
        try:
            start_time = time.time()
            response = self.model.generate_content(prompt)
            return {"response": response.text, "latency": time.time() - start_time}
        except Exception as e:
            return {"response": f"Error: {e}", "latency": 0.0}

agent = AdvancedLLMAgent()

@app.route('/reason', methods=['POST'])
def reason():
    """Flask endpoint, now requires a 'domain' field."""
    data = request.get_json()
    if not data or 'prompt' not in data or 'domain' not in data:
        return jsonify({"error": "Invalid request. 'prompt' and 'domain' are required."}), 400
    
    if data['domain'] not in ['legal', 'math']:
        return jsonify({"error": "Invalid domain. Must be 'legal' or 'math'."}), 400

    try:
        result = agent.get_ultra_cognition(
            task_prompt=data['prompt'],
            mode=data.get('mode', 'reasoning'),
            domain=data['domain']
        )
        return jsonify(json.loads(json.dumps(result, default=str)))
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
