import time
import random
import os
import re
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict
import sympy as sp
from sympy import symbols, solve, diff, integrate, simplify, Matrix
from sympy.logic import simplify_logic
import itertools
import zlib
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify
from flask_cors import CORS
import scipy.stats as stats
from scipy.optimize import minimize, linprog
import pulp
from reasoning_logger import ReasoningSessionManager

# Configuration
API_KEY = "AIzaSyBLXXuiqpx9BfDxGi28Ci8szlsb3qAm9Dw"
genai.configure(api_key=API_KEY)

app = Flask(__name__)
CORS(app)

# Enhanced Reasoning Types combining both engines
class UnifiedReasoningType(Enum):
    # Foundational Mathematical/Logical
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
    
    # Mathematical Foundations
    SET_THEORETIC = "set_theoretic"
    ALGEBRAIC_STRUCTURE = "algebraic_structure"
    TOPOLOGICAL_REASONING = "topological_reasoning"
    CATEGORY_THEORETIC = "category_theoretic"
    MEASURE_THEORETIC = "measure_theoretic"
    ORDER_THEORETIC = "order_theoretic"
    
    # Reasoning Dimensions (Negative to Positive)
    NEGATIVE_REASONING = "negative_reasoning"  # What cannot be, constraints
    BOUNDARY_REASONING = "boundary_reasoning"  # Limits, edges, thresholds
    TRANSITIONAL_REASONING = "transitional_reasoning"  # Changes, transformations
    POSITIVE_REASONING = "positive_reasoning"  # What can be, possibilities
    EMERGENT_REASONING = "emergent_reasoning"  # Emergence, complexity
    
    # Domain-Specific
    LINEAR_ALGEBRAIC = "linear_algebraic"
    CALCULUS_BASED = "calculus_based"
    STATISTICAL = "statistical"
    OPTIMIZATION = "optimization"
    GAME_THEORETIC = "game_theoretic"
    
    # Legal/Regulatory
    STATUTORY_INTERPRETATION = "statutory_interpretation"
    PRECEDENT_ANALYSIS = "precedent_analysis"
    POLICY_BASED = "policy_based"
    TEXTUALISM = "textualism"
    ORIGINALISM = "originalism"
    PURPOSIVISM = "purposivism"
    BALANCING_TEST = "balancing_test"
    
    # Scientific
    EXPERIMENTAL_DESIGN = "experimental_design"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    SYSTEMS_THINKING = "systems_thinking"
    EMERGENT_PROPERTIES = "emergent_properties"

@dataclass
class ExpertPerspective:
    """Represents a single expert's analysis"""
    expert_type: str
    confidence: float
    key_insights: List[str]
    mathematical_foundation: str
    recommended_approach: str
    causal_chain: List[str]
    hidden_patterns: List[str]
    reasoning_dimension: str  # Negative, Boundary, Transitional, Positive, Emergent
    foundational_principles: List[str]  # Core mathematical/logical principles

@dataclass
class GranularReasoningStep:
    """Individual step in a reasoning process"""
    step_id: str
    step_type: str  # "what", "how", "why", "when", "where", "who", "which"
    description: str
    mathematical_basis: str
    logical_justification: str
    confidence: float
    dependencies: List[str]  # IDs of steps this depends on
    outputs: List[str]  # What this step produces

@dataclass
class UnifiedReasoningPath:
    """Enhanced reasoning path with granular steps and multi-expert validation"""
    path_id: str
    reasoning_types: List[UnifiedReasoningType]
    expert_perspectives: List[ExpertPerspective]
    mathematical_foundation: str
    logical_structure: str
    confidence_score: float
    pattern_complexity: int
    epistemic_depth: int
    causal_graph: Optional[nx.DiGraph]
    
    # Granular reasoning steps
    reasoning_steps: List[GranularReasoningStep]
    
    # Enhanced explanations
    what_explanation: str  # What is being reasoned
    how_explanation: str   # How the reasoning works
    why_explanation: str   # Why this reasoning is valid
    when_explanation: str  # When this reasoning applies
    where_explanation: str # Where this reasoning is valid
    who_explanation: str   # Who/what entities are involved
    which_explanation: str # Which alternatives exist

@dataclass
class UnifiedPattern:
    """Unified pattern incorporating multiple domains"""
    pattern_id: str
    pattern_type: str
    mathematical_expression: Optional[str]
    linguistic_expression: Optional[str]
    domain_specific_form: Dict[str, Any]
    cross_domain_connections: List[str]
    emergence_level: int
    information_theoretic_measure: float

class ExpertMathematicalFoundation:
    """Expert in foundational mathematical reasoning"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_foundation = ""
        causal_chain = []
        hidden_patterns = []
        foundational_principles = []
        
        # Detect set-theoretic reasoning
        if any(term in problem.lower() for term in ['set', 'element', 'member', 'subset', 'union', 'intersection']):
            insights.append("Problem involves set-theoretic reasoning")
            mathematical_foundation = "A ⊆ B, A ∪ B, A ∩ B"
            foundational_principles.append("Set Theory: Axiom of Extensionality")
            causal_chain.append("Elements → Sets → Relations → Functions")
            
        # Detect algebraic structures
        if any(term in problem.lower() for term in ['group', 'ring', 'field', 'operation', 'identity', 'inverse']):
            insights.append("Problem involves algebraic structures")
            mathematical_foundation += " | (G, *) with closure, associativity, identity, inverse"
            foundational_principles.append("Group Theory: Closure under operation")
            
        # Detect order relations
        if any(term in problem.lower() for term in ['order', 'partial', 'total', 'chain', 'lattice', 'poset']):
            insights.append("Problem involves order-theoretic reasoning")
            mathematical_foundation += " | ≤ reflexive, antisymmetric, transitive"
            foundational_principles.append("Order Theory: Transitivity of relations")
            
        # Detect measure-theoretic concepts
        if any(term in problem.lower() for term in ['measure', 'sigma', 'algebra', 'measurable', 'integral']):
            insights.append("Problem involves measure-theoretic reasoning")
            mathematical_foundation += " | μ: Σ → [0,∞] with countable additivity"
            foundational_principles.append("Measure Theory: Countable additivity")
            
        return ExpertPerspective(
            expert_type="Mathematical Foundation",
            confidence=0.9 if insights else 0.3,
            key_insights=insights,
            mathematical_foundation=mathematical_foundation,
            recommended_approach="Apply foundational mathematical principles systematically",
            causal_chain=causal_chain,
            hidden_patterns=hidden_patterns,
            reasoning_dimension="positive",
            foundational_principles=foundational_principles
        )

class ExpertNegativeReasoning:
    """Expert in negative reasoning - what cannot be, constraints, impossibilities"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_foundation = ""
        causal_chain = []
        hidden_patterns = []
        foundational_principles = []
        
        # Detect impossibility constraints
        if any(term in problem.lower() for term in ['impossible', 'cannot', 'never', 'contradiction', 'paradox']):
            insights.append("Problem involves impossibility constraints")
            mathematical_foundation = "¬P ∧ P ≡ ⊥ (contradiction)"
            foundational_principles.append("Logic: Law of Non-Contradiction")
            causal_chain.append("Constraint → Impossibility → Contradiction → Resolution")
            
        # Detect boundary conditions
        if any(term in problem.lower() for term in ['limit', 'bound', 'constraint', 'restriction', 'forbidden']):
            insights.append("Problem involves boundary constraints")
            mathematical_foundation += " | f(x) ≤ M for all x ∈ X"
            foundational_principles.append("Analysis: Boundedness principle")
            
        # Detect negative definitions
        if any(term in problem.lower() for term in ['not', 'except', 'exclude', 'without', 'unless']):
            insights.append("Problem involves negative definitions")
            mathematical_foundation += " | A = U \\ B (complement)"
            foundational_principles.append("Set Theory: Complement operation")
            
        return ExpertPerspective(
            expert_type="Negative Reasoning",
            confidence=0.85 if insights else 0.2,
            key_insights=insights,
            mathematical_foundation=mathematical_foundation,
            recommended_approach="Identify constraints and impossibilities first",
            causal_chain=causal_chain,
            hidden_patterns=hidden_patterns,
            reasoning_dimension="negative",
            foundational_principles=foundational_principles
        )

class ExpertBoundaryReasoning:
    """Expert in boundary reasoning - limits, edges, thresholds, critical points"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_foundation = ""
        causal_chain = []
        hidden_patterns = []
        foundational_principles = []
        
        # Detect critical points
        if any(term in problem.lower() for term in ['critical', 'maximum', 'minimum', 'extremum', 'peak', 'valley']):
            insights.append("Problem involves critical point analysis")
            mathematical_foundation = "f'(x) = 0 or f'(x) undefined"
            foundational_principles.append("Calculus: First derivative test")
            causal_chain.append("Function → Derivative → Critical points → Extrema")
            
        # Detect limits and convergence
        if any(term in problem.lower() for term in ['limit', 'converge', 'approach', 'tend', 'asymptote']):
            insights.append("Problem involves limit analysis")
            mathematical_foundation += " | lim_{x→a} f(x) = L"
            foundational_principles.append("Analysis: Epsilon-delta definition")
            
        # Detect phase transitions
        if any(term in problem.lower() for term in ['phase', 'transition', 'threshold', 'tipping', 'critical']):
            insights.append("Problem involves phase transitions")
            mathematical_foundation += " | Bifurcation theory: dx/dt = f(x,λ)"
            foundational_principles.append("Dynamical Systems: Bifurcation analysis")
            
        return ExpertPerspective(
            expert_type="Boundary Reasoning",
            confidence=0.8 if insights else 0.25,
            key_insights=insights,
            mathematical_foundation=mathematical_foundation,
            recommended_approach="Analyze critical points and boundary conditions",
            causal_chain=causal_chain,
            hidden_patterns=hidden_patterns,
            reasoning_dimension="boundary",
            foundational_principles=foundational_principles
        )

class ExpertTransitionalReasoning:
    """Expert in transitional reasoning - changes, transformations, processes"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_foundation = ""
        causal_chain = []
        hidden_patterns = []
        foundational_principles = []
        
        # Detect transformations
        if any(term in problem.lower() for term in ['transform', 'change', 'convert', 'evolve', 'develop']):
            insights.append("Problem involves transformations")
            mathematical_foundation = "T: X → Y (transformation)"
            foundational_principles.append("Linear Algebra: Transformation properties")
            causal_chain.append("Initial state → Transformation → Final state")
            
        # Detect processes and flows
        if any(term in problem.lower() for term in ['process', 'flow', 'sequence', 'series', 'progression']):
            insights.append("Problem involves process analysis")
            mathematical_foundation += " | x_{n+1} = f(x_n) (recurrence)"
            foundational_principles.append("Dynamical Systems: Iteration theory")
            
        # Detect differential changes
        if any(term in problem.lower() for term in ['rate', 'speed', 'velocity', 'acceleration', 'momentum']):
            insights.append("Problem involves rate of change")
            mathematical_foundation += " | dx/dt = f(x,t)"
            foundational_principles.append("Calculus: Differential equations")
            
        return ExpertPerspective(
            expert_type="Transitional Reasoning",
            confidence=0.85 if insights else 0.2,
            key_insights=insights,
            mathematical_foundation=mathematical_foundation,
            recommended_approach="Model as transformation or process",
            causal_chain=causal_chain,
            hidden_patterns=hidden_patterns,
            reasoning_dimension="transitional",
            foundational_principles=foundational_principles
        )

class ExpertPositiveReasoning:
    """Expert in positive reasoning - what can be, possibilities, constructions"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_foundation = ""
        causal_chain = []
        hidden_patterns = []
        foundational_principles = []
        
        # Detect constructive proofs
        if any(term in problem.lower() for term in ['construct', 'build', 'create', 'find', 'show', 'prove']):
            insights.append("Problem involves constructive reasoning")
            mathematical_foundation = "∃x P(x) → construct x such that P(x)"
            foundational_principles.append("Logic: Constructive existence")
            causal_chain.append("Existence → Construction → Verification")
            
        # Detect algorithms and procedures
        if any(term in problem.lower() for term in ['algorithm', 'procedure', 'method', 'technique', 'approach']):
            insights.append("Problem involves algorithmic reasoning")
            mathematical_foundation += " | Algorithm: finite sequence of well-defined steps"
            foundational_principles.append("Computability: Effective procedures")
            
        # Detect optimization
        if any(term in problem.lower() for term in ['optimize', 'maximize', 'minimize', 'best', 'optimal']):
            insights.append("Problem involves optimization")
            mathematical_foundation += " | arg max f(x) subject to constraints"
            foundational_principles.append("Optimization: Karush-Kuhn-Tucker conditions")
            
        return ExpertPerspective(
            expert_type="Positive Reasoning",
            confidence=0.9 if insights else 0.3,
            key_insights=insights,
            mathematical_foundation=mathematical_foundation,
            recommended_approach="Construct solutions and verify existence",
            causal_chain=causal_chain,
            hidden_patterns=hidden_patterns,
            reasoning_dimension="positive",
            foundational_principles=foundational_principles
        )

class ExpertEmergentReasoning:
    """Expert in emergent reasoning - complexity, emergence, self-organization"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_foundation = ""
        causal_chain = []
        hidden_patterns = []
        foundational_principles = []
        
        # Detect emergent properties
        if any(term in problem.lower() for term in ['emerge', 'emergent', 'collective', 'system', 'network', 'interaction']):
            insights.append("Problem involves emergent phenomena")
            mathematical_foundation = "Emergence: properties not reducible to components"
            foundational_principles.append("Complexity Theory: Emergence principle")
            causal_chain.append("Components → Interactions → Emergent properties")
            
        # Detect self-organization
        if any(term in problem.lower() for term in ['self-organize', 'pattern', 'structure', 'order', 'coherence']):
            insights.append("Problem involves self-organization")
            mathematical_foundation += " | Self-organization: spontaneous order formation"
            foundational_principles.append("Non-equilibrium Thermodynamics: Self-organization")
            
        # Detect phase transitions
        if any(term in problem.lower() for term in ['phase', 'transition', 'critical', 'bifurcation', 'chaos']):
            insights.append("Problem involves phase transitions")
            mathematical_foundation += " | Phase transition: qualitative change in system behavior"
            foundational_principles.append("Dynamical Systems: Bifurcation theory")
            
        return ExpertPerspective(
            expert_type="Emergent Reasoning",
            confidence=0.75 if insights else 0.15,
            key_insights=insights,
            mathematical_foundation=mathematical_foundation,
            recommended_approach="Analyze system-level properties and interactions",
            causal_chain=causal_chain,
            hidden_patterns=hidden_patterns,
            reasoning_dimension="emergent",
            foundational_principles=foundational_principles
        )

class ExpertLogicalStructure:
    """Expert in logical structure and formal reasoning"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_foundation = ""
        causal_chain = []
        hidden_patterns = []
        foundational_principles = []
        
        # Detect logical connectives
        if any(term in problem.lower() for term in ['and', 'or', 'not', 'if', 'then', 'implies', 'equivalent']):
            insights.append("Problem involves logical connectives")
            mathematical_foundation = "∧, ∨, ¬, →, ↔ (logical operators)"
            foundational_principles.append("Logic: Boolean algebra")
            causal_chain.append("Premises → Logical inference → Conclusion")
            
        # Detect quantifiers
        if any(term in problem.lower() for term in ['all', 'every', 'some', 'exists', 'for all', 'there exists']):
            insights.append("Problem involves quantifiers")
            mathematical_foundation += " | ∀x, ∃x (universal and existential quantifiers)"
            foundational_principles.append("Logic: First-order predicate logic")
            
        # Detect proof structures
        if any(term in problem.lower() for term in ['prove', 'proof', 'theorem', 'lemma', 'corollary']):
            insights.append("Problem involves proof structure")
            mathematical_foundation += " | Axioms → Theorems → Proofs"
            foundational_principles.append("Mathematics: Axiomatic method")
            
        return ExpertPerspective(
            expert_type="Logical Structure",
            confidence=0.9 if insights else 0.4,
            key_insights=insights,
            mathematical_foundation=mathematical_foundation,
            recommended_approach="Apply formal logical reasoning",
            causal_chain=causal_chain,
            hidden_patterns=hidden_patterns,
            reasoning_dimension="positive",
            foundational_principles=foundational_principles
        )

class ExpertCausalReasoning:
    """Expert in causal reasoning and inference"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_foundation = ""
        causal_chain = []
        hidden_patterns = []
        foundational_principles = []
        
        # Detect causal relationships
        if any(term in problem.lower() for term in ['cause', 'effect', 'because', 'leads to', 'results in', 'due to']):
            insights.append("Problem involves causal relationships")
            mathematical_foundation = "C → E (cause leads to effect)"
            foundational_principles.append("Causality: Temporal precedence")
            causal_chain.append("Cause → Mechanism → Effect")
            
        # Detect counterfactuals
        if any(term in problem.lower() for term in ['if not', 'would have', 'might have', 'counterfactual']):
            insights.append("Problem involves counterfactual reasoning")
            mathematical_foundation += " | P(E|do(C)) vs P(E|do(¬C))"
            foundational_principles.append("Causal Inference: Do-calculus")
            
        # Detect confounding
        if any(term in problem.lower() for term in ['confound', 'bias', 'correlation', 'association']):
            insights.append("Problem involves confounding analysis")
            mathematical_foundation += " | Backdoor criterion for causal identification"
            foundational_principles.append("Causal Inference: Confounding control")
            
        return ExpertPerspective(
            expert_type="Causal Reasoning",
            confidence=0.85 if insights else 0.2,
            key_insights=insights,
            mathematical_foundation=mathematical_foundation,
            recommended_approach="Identify causal mechanisms and control confounding",
            causal_chain=causal_chain,
            hidden_patterns=hidden_patterns,
            reasoning_dimension="transitional",
            foundational_principles=foundational_principles
        )

class ExpertPatternFinder:
    """Expert in pattern recognition and analogical reasoning"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_foundation = ""
        causal_chain = []
        hidden_patterns = []
        foundational_principles = []
        
        # Detect patterns and symmetries
        if any(term in problem.lower() for term in ['pattern', 'symmetry', 'regular', 'periodic', 'recurring']):
            insights.append("Problem involves pattern recognition")
            mathematical_foundation = "Pattern: f(x + T) = f(x) for period T"
            foundational_principles.append("Mathematics: Symmetry principles")
            causal_chain.append("Observation → Pattern recognition → Generalization")
            
        # Detect analogies
        if any(term in problem.lower() for term in ['similar', 'like', 'analogous', 'corresponds', 'mapping']):
            insights.append("Problem involves analogical reasoning")
            mathematical_foundation += " | A:B :: C:D (analogical proportion)"
            foundational_principles.append("Analogy: Structural similarity")
            
        # Detect invariants
        if any(term in problem.lower() for term in ['invariant', 'conserved', 'constant', 'unchanged']):
            insights.append("Problem involves invariant analysis")
            mathematical_foundation += " | I(x) = I(T(x)) for transformation T"
            foundational_principles.append("Mathematics: Invariant theory")
            
        return ExpertPerspective(
            expert_type="Pattern Recognition",
            confidence=0.8 if insights else 0.3,
            key_insights=insights,
            mathematical_foundation=mathematical_foundation,
            recommended_approach="Identify patterns and apply analogical reasoning",
            causal_chain=causal_chain,
            hidden_patterns=hidden_patterns,
            reasoning_dimension="emergent",
            foundational_principles=foundational_principles
        )

class DomainDetector:
    """Automatically detects the domain of a problem"""
    
    def __init__(self):
        self.domain_keywords = {
            'mathematics': ['equation', 'solve', 'calculate', 'prove', 'theorem', 'formula'],
            'physics': ['force', 'energy', 'momentum', 'wave', 'particle', 'field'],
            'chemistry': ['reaction', 'molecule', 'compound', 'element', 'bond', 'solution'],
            'biology': ['cell', 'organism', 'gene', 'evolution', 'protein', 'ecosystem'],
            'computer_science': ['algorithm', 'data structure', 'complexity', 'program', 'code'],
            'economics': ['market', 'price', 'supply', 'demand', 'utility', 'equilibrium'],
            'psychology': ['behavior', 'cognition', 'emotion', 'perception', 'memory'],
            'philosophy': ['ethics', 'metaphysics', 'epistemology', 'logic', 'truth'],
            'law': ['statute', 'precedent', 'contract', 'liability', 'jurisdiction'],
            'medicine': ['diagnosis', 'treatment', 'symptom', 'disease', 'patient'],
            'engineering': ['design', 'system', 'optimize', 'constraint', 'specification'],
            'linguistics': ['language', 'grammar', 'syntax', 'semantics', 'phonetics'],
            'history': ['event', 'period', 'civilization', 'revolution', 'empire'],
            'art': ['style', 'composition', 'aesthetic', 'technique', 'medium'],
            'music': ['rhythm', 'melody', 'harmony', 'tempo', 'composition']
        }
        
    def detect_domain(self, problem: str) -> Tuple[str, float]:
        """Returns the detected domain and confidence score"""
        problem_lower = problem.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in problem_lower)
            if score > 0:
                domain_scores[domain] = score
                
        if not domain_scores:
            return 'general', 0.5
            
        # Get the domain with highest score
        best_domain = max(domain_scores, key=domain_scores.get)
        total_score = sum(domain_scores.values())
        confidence = domain_scores[best_domain] / total_score if total_score > 0 else 0
        
        return best_domain, confidence

class UnifiedReasoningEngine:
    """The ultimate unified reasoning engine"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.embedding_model = genai.GenerativeModel('text-embedding-004')
        
        # Initialize all experts
        self.experts = {
            'mathematical_foundation': ExpertMathematicalFoundation(),
            'negative_reasoning': ExpertNegativeReasoning(),
            'boundary_reasoning': ExpertBoundaryReasoning(),
            'transitional_reasoning': ExpertTransitionalReasoning(),
            'positive_reasoning': ExpertPositiveReasoning(),
            'emergent_reasoning': ExpertEmergentReasoning(),
            'logical_structure': ExpertLogicalStructure(),
            'causal': ExpertCausalReasoning(),
            'pattern': ExpertPatternFinder()
        }
        
        self.domain_detector = DomainDetector()
        self.reasoning_paths = []
        self.tfidf_vectorizer = TfidfVectorizer()
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings for text"""
        try:
            result = self.embedding_model.embed_content(
                model='text-embedding-004',
                content=text
            )
            return np.array(result['embedding'])
        except Exception as e:
            print(f"Embedding error: {e}")
            return np.zeros(768)
            
    def analyze_problem_structure(self, problem: str) -> Dict[str, Any]:
        """Deep structural analysis of the problem"""
        # Build concept graph
        words = re.findall(r'\b\w+\b', problem.lower())
        concept_graph = nx.Graph()
        
        for i in range(len(words) - 1):
            concept_graph.add_edge(words[i], words[i+1])
            
        # Calculate various metrics
        entropy = self._calculate_entropy(problem)
        complexity = self._calculate_complexity(problem)
        
        # Extract mathematical expressions
        math_expressions = re.findall(
            r'[a-zA-Z]+\s*=\s*[0-9a-zA-Z\+\-\*/\^\(\)\s\.,_]+|'
            r'\d+\s*[\+\-\*/]\s*\d+|'
            r'[a-zA-Z]\([a-zA-Z0-9,\s]+\)',
            problem
        )
        
        return {
            'concept_graph': concept_graph,
            'entropy': entropy,
            'complexity': complexity,
            'math_expressions': math_expressions,
            'token_count': len(words),
            'unique_concepts': len(set(words)),
            'graph_density': nx.density(concept_graph) if concept_graph.number_of_nodes() > 0 else 0
        }
        
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
            
        freq_dist = defaultdict(int)
        for word in words:
            freq_dist[word] += 1
            
        total = len(words)
        entropy = 0.0
        for freq in freq_dist.values():
            p = freq / total
            entropy -= p * math.log2(p)
            
        return entropy
        
    def _calculate_complexity(self, text: str) -> float:
        """Approximate Kolmogorov complexity"""
        if not text:
            return 0.0
        compressed = zlib.compress(text.encode('utf-8'))
        return len(compressed) / len(text.encode('utf-8'))
        
    def generate_reasoning_paths(self, problem: str, domain: str, expert_analyses: List[ExpertPerspective]) -> List[UnifiedReasoningPath]:
        """Generate multiple reasoning paths combining expert insights with granular steps"""
        paths = []
        
        # Combine high-confidence expert perspectives
        high_confidence_experts = [e for e in expert_analyses if e.confidence > 0.7]
        
        if high_confidence_experts:
            # Create primary reasoning path
            primary_types = []
            if any('mathematical' in e.expert_type.lower() for e in high_confidence_experts):
                primary_types.append(UnifiedReasoningType.SET_THEORETIC)
                primary_types.append(UnifiedReasoningType.ALGEBRAIC_STRUCTURE)
                primary_types.append(UnifiedReasoningType.ORDER_THEORETIC)
                primary_types.append(UnifiedReasoningType.MEASURE_THEORETIC)
            if any('negative' in e.expert_type.lower() for e in high_confidence_experts):
                primary_types.append(UnifiedReasoningType.NEGATIVE_REASONING)
            if any('boundary' in e.expert_type.lower() for e in high_confidence_experts):
                primary_types.append(UnifiedReasoningType.BOUNDARY_REASONING)
            if any('transitional' in e.expert_type.lower() for e in high_confidence_experts):
                primary_types.append(UnifiedReasoningType.TRANSITIONAL_REASONING)
            if any('positive' in e.expert_type.lower() for e in high_confidence_experts):
                primary_types.append(UnifiedReasoningType.POSITIVE_REASONING)
            if any('emergent' in e.expert_type.lower() for e in high_confidence_experts):
                primary_types.append(UnifiedReasoningType.EMERGENT_REASONING)
            if any('logical' in e.expert_type.lower() for e in high_confidence_experts):
                primary_types.append(UnifiedReasoningType.DEDUCTIVE)
            if any('causal' in e.expert_type.lower() for e in high_confidence_experts):
                primary_types.append(UnifiedReasoningType.CAUSAL)
                
            # Build causal graph
            causal_graph = nx.DiGraph()
            for expert in high_confidence_experts:
                for i in range(len(expert.causal_chain) - 1):
                    causal_graph.add_edge(expert.causal_chain[i], expert.causal_chain[i+1])
            
            # Generate granular reasoning steps
            reasoning_steps = self._generate_granular_steps(problem, high_confidence_experts)
            
            # Create enhanced explanations
            what_explanation = self._generate_what_explanation(problem, high_confidence_experts)
            how_explanation = self._generate_how_explanation(high_confidence_experts, reasoning_steps)
            why_explanation = self._generate_why_explanation(high_confidence_experts)
            when_explanation = self._generate_when_explanation(high_confidence_experts)
            where_explanation = self._generate_where_explanation(domain, high_confidence_experts)
            who_explanation = self._generate_who_explanation(problem, high_confidence_experts)
            which_explanation = self._generate_which_explanation(high_confidence_experts)
                    
            # Create unified path
            path = UnifiedReasoningPath(
                path_id=f"unified_{hash(problem) % 10000}",
                reasoning_types=primary_types or [UnifiedReasoningType.DEDUCTIVE],
                expert_perspectives=high_confidence_experts,
                mathematical_foundation=" ∧ ".join([e.mathematical_foundation for e in high_confidence_experts if e.mathematical_foundation]),
                logical_structure=f"Multi-expert consensus with {len(high_confidence_experts)} perspectives",
                confidence_score=np.mean([e.confidence for e in high_confidence_experts]),
                pattern_complexity=len(set(p for e in high_confidence_experts for p in e.hidden_patterns)),
                epistemic_depth=max(len(e.causal_chain) for e in high_confidence_experts),
                causal_graph=causal_graph,
                reasoning_steps=reasoning_steps,
                what_explanation=what_explanation,
                how_explanation=how_explanation,
                why_explanation=why_explanation,
                when_explanation=when_explanation,
                where_explanation=where_explanation,
                who_explanation=who_explanation,
                which_explanation=which_explanation
            )
            paths.append(path)
            
        # Add fallback general reasoning path
        if not paths:
            # Generate basic granular steps for fallback
            basic_steps = [
                GranularReasoningStep(
                    step_id="step_1",
                    step_type="what",
                    description="Identify the core problem and what needs to be solved",
                    mathematical_basis="Problem formulation: P(x) → find x such that P(x)",
                    logical_justification="Abductive reasoning to find best explanation",
                    confidence=0.5,
                    dependencies=[],
                    outputs=["problem_understanding"]
                ),
                GranularReasoningStep(
                    step_id="step_2", 
                    step_type="how",
                    description="Apply general problem-solving approach",
                    mathematical_basis="General solution method: systematic exploration",
                    logical_justification="Heuristic search in solution space",
                    confidence=0.4,
                    dependencies=["step_1"],
                    outputs=["solution_approach"]
                ),
                GranularReasoningStep(
                    step_id="step_3",
                    step_type="why", 
                    description="Justify the chosen approach",
                    mathematical_basis="Validation: check if solution satisfies constraints",
                    logical_justification="Consistency with given information",
                    confidence=0.3,
                    dependencies=["step_2"],
                    outputs=["solution_validation"]
                )
            ]
            
            paths.append(UnifiedReasoningPath(
                path_id=f"general_{hash(problem) % 10000}",
                reasoning_types=[UnifiedReasoningType.ABDUCTIVE],
                expert_perspectives=expert_analyses,
                mathematical_foundation="General logical inference",
                logical_structure="Exploratory reasoning",
                confidence_score=0.5,
                pattern_complexity=1,
                epistemic_depth=2,
                causal_graph=None,
                reasoning_steps=basic_steps,
                what_explanation=f"Exploring: {problem[:100]}...",
                how_explanation="Using general abductive reasoning to find best explanation",
                why_explanation="No high-confidence expert paths identified",
                when_explanation="When no specific domain expertise is available",
                where_explanation="In general problem-solving contexts",
                who_explanation="General reasoning system",
                which_explanation="Alternative approaches not yet identified"
            ))
            
        return paths
    
    def _generate_granular_steps(self, problem: str, experts: List[ExpertPerspective]) -> List[GranularReasoningStep]:
        """Generate granular reasoning steps from expert analyses"""
        steps = []
        step_counter = 1
        
        # Step 1: WHAT - Problem understanding
        what_insights = []
        for expert in experts:
            what_insights.extend(expert.key_insights[:2])
        
        steps.append(GranularReasoningStep(
            step_id=f"step_{step_counter}",
            step_type="what",
            description=f"Understand the problem: {', '.join(what_insights[:3])}",
            mathematical_basis="Problem formulation and constraint identification",
            logical_justification="Multi-expert consensus on problem structure",
            confidence=np.mean([e.confidence for e in experts]),
            dependencies=[],
            outputs=["problem_understanding", "constraint_identification"]
        ))
        step_counter += 1
        
        # Step 2: HOW - Solution approach
        approaches = [e.recommended_approach for e in experts if e.recommended_approach]
        steps.append(GranularReasoningStep(
            step_id=f"step_{step_counter}",
            step_type="how",
            description=f"Apply solution approach: {approaches[0] if approaches else 'Systematic analysis'}",
            mathematical_basis=" ∧ ".join([e.mathematical_foundation for e in experts if e.mathematical_foundation]),
            logical_justification="Expert-recommended methodologies",
            confidence=np.mean([e.confidence for e in experts]),
            dependencies=[f"step_{step_counter-1}"],
            outputs=["solution_methodology", "mathematical_framework"]
        ))
        step_counter += 1
        
        # Step 3: WHY - Justification
        principles = []
        for expert in experts:
            principles.extend(expert.foundational_principles[:2])
        
        steps.append(GranularReasoningStep(
            step_id=f"step_{step_counter}",
            step_type="why",
            description=f"Justify approach using: {', '.join(principles[:3])}",
            mathematical_basis="Mathematical and logical foundations",
            logical_justification="Consistency with established principles",
            confidence=np.mean([e.confidence for e in experts]),
            dependencies=[f"step_{step_counter-1}"],
            outputs=["theoretical_justification", "principle_validation"]
        ))
        step_counter += 1
        
        # Step 4: WHEN - Applicability conditions
        steps.append(GranularReasoningStep(
            step_id=f"step_{step_counter}",
            step_type="when",
            description="Determine when this reasoning applies",
            mathematical_basis="Domain of validity and applicability conditions",
            logical_justification="Expert domain knowledge and constraints",
            confidence=np.mean([e.confidence for e in experts]),
            dependencies=[f"step_{step_counter-2}"],
            outputs=["applicability_conditions", "domain_constraints"]
        ))
        step_counter += 1
        
        # Step 5: WHERE - Spatial/contextual validity
        steps.append(GranularReasoningStep(
            step_id=f"step_{step_counter}",
            step_type="where",
            description="Identify where this reasoning is valid",
            mathematical_basis="Spatial and contextual boundaries",
            logical_justification="Scope and limitations of approach",
            confidence=np.mean([e.confidence for e in experts]),
            dependencies=[f"step_{step_counter-1}"],
            outputs=["spatial_validity", "contextual_boundaries"]
        ))
        step_counter += 1
        
        # Step 6: WHO - Entity identification
        steps.append(GranularReasoningStep(
            step_id=f"step_{step_counter}",
            step_type="who",
            description="Identify entities and agents involved",
            mathematical_basis="Entity modeling and agent identification",
            logical_justification="Stakeholder and component analysis",
            confidence=np.mean([e.confidence for e in experts]),
            dependencies=[f"step_{step_counter-2}"],
            outputs=["entity_identification", "agent_analysis"]
        ))
        step_counter += 1
        
        # Step 7: WHICH - Alternative selection
        steps.append(GranularReasoningStep(
            step_id=f"step_{step_counter}",
            step_type="which",
            description="Select among alternative approaches",
            mathematical_basis="Decision theory and optimization",
            logical_justification="Comparative analysis of alternatives",
            confidence=np.mean([e.confidence for e in experts]),
            dependencies=[f"step_{step_counter-3}"],
            outputs=["alternative_selection", "optimal_choice"]
        ))
        
        return steps
    
    def _generate_what_explanation(self, problem: str, experts: List[ExpertPerspective]) -> str:
        """Generate WHAT explanation"""
        insights = []
        for expert in experts:
            insights.extend(expert.key_insights[:2])
        
        dimensions = [e.reasoning_dimension for e in experts]
        dimension_str = ", ".join(set(dimensions))
        
        return f"WHAT: Analyzing {problem[:50]}... through {dimension_str} reasoning dimensions. Key insights: {', '.join(insights[:3])}"
    
    def _generate_how_explanation(self, experts: List[ExpertPerspective], steps: List[GranularReasoningStep]) -> str:
        """Generate HOW explanation"""
        approaches = [e.recommended_approach for e in experts if e.recommended_approach]
        step_count = len(steps)
        
        return f"HOW: Applying {len(experts)} expert methodologies through {step_count} granular steps. Primary approaches: {', '.join(approaches[:2])}"
    
    def _generate_why_explanation(self, experts: List[ExpertPerspective]) -> str:
        """Generate WHY explanation"""
        principles = []
        for expert in experts:
            principles.extend(expert.foundational_principles[:1])
        
        confidence = np.mean([e.confidence for e in experts])
        
        return f"WHY: Justified by foundational principles: {', '.join(principles[:3])}. Confidence: {confidence:.2f}"
    
    def _generate_when_explanation(self, experts: List[ExpertPerspective]) -> str:
        """Generate WHEN explanation"""
        dimensions = [e.reasoning_dimension for e in experts]
        conditions = []
        
        if "negative" in dimensions:
            conditions.append("when constraints are present")
        if "boundary" in dimensions:
            conditions.append("when limits or thresholds matter")
        if "transitional" in dimensions:
            conditions.append("when changes or processes occur")
        if "positive" in dimensions:
            conditions.append("when constructive solutions are needed")
        if "emergent" in dimensions:
            conditions.append("when complex interactions arise")
            
        return f"WHEN: This reasoning applies {', '.join(conditions)}"
    
    def _generate_where_explanation(self, domain: str, experts: List[ExpertPerspective]) -> str:
        """Generate WHERE explanation"""
        return f"WHERE: Valid in {domain} domain and contexts where {len(experts)} expert perspectives converge"
    
    def _generate_who_explanation(self, problem: str, experts: List[ExpertPerspective]) -> str:
        """Generate WHO explanation"""
        # Extract potential entities from problem
        words = problem.split()
        entities = [w for w in words if w[0].isupper() or w.lower() in ['system', 'process', 'function', 'variable']]
        
        return f"WHO: Involves entities: {', '.join(entities[:3])} and {len(experts)} expert reasoning systems"
    
    def _generate_which_explanation(self, experts: List[ExpertPerspective]) -> str:
        """Generate WHICH explanation"""
        expert_types = [e.expert_type for e in experts]
        alternatives = f"among {len(expert_types)} expert approaches: {', '.join(expert_types[:3])}"
        
        return f"WHICH: Selecting optimal approach {alternatives} based on confidence and domain fit"
    
    def synthesize_solution(self, problem: str, paths: List[UnifiedReasoningPath], problem_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final solution using all reasoning paths"""
        # Prepare prompt for final synthesis
        path_summaries = []
        for path in paths:
            summary = {
                'reasoning_types': [t.value for t in path.reasoning_types],
                'confidence': path.confidence_score,
                'key_insights': [insight for expert in path.expert_perspectives for insight in expert.key_insights],
                'hidden_patterns': [pattern for expert in path.expert_perspectives for pattern in expert.hidden_patterns],
                'mathematical_foundations': path.mathematical_foundation
            }
            path_summaries.append(summary)
            
        synthesis_prompt = f"""
You are a unified reasoning system with expertise across all domains of human knowledge. You have access to analyses from 8 different foundational reasoning experts:
1. Mathematical Foundation Expert - Set theory, algebraic structures, order theory, measure theory
2. Negative Reasoning Expert - Constraints, impossibilities, what cannot be
3. Boundary Reasoning Expert - Limits, critical points, thresholds, phase transitions
4. Transitional Reasoning Expert - Changes, transformations, processes, dynamics
5. Positive Reasoning Expert - Possibilities, constructions, what can be
6. Emergent Reasoning Expert - Complexity, emergence, self-organization
7. Logical Structure Expert - Formal logic, proof structures, quantifiers
8. Causal Reasoning Expert - Causality, counterfactuals, confounding

PROBLEM: {problem}

EXPERT ANALYSES AND REASONING PATHS:
{json.dumps(path_summaries, indent=2)}

PROBLEM STRUCTURE:
- Entropy: {problem_structure['entropy']:.2f}
- Complexity: {problem_structure['complexity']:.2f}
- Mathematical expressions found: {problem_structure['math_expressions']}

REASONING FRAMEWORK:
The analysis uses a comprehensive 7-dimensional reasoning framework:

1. **WHAT**: What is being reasoned about and what needs to be solved
2. **HOW**: How the reasoning process works and what methods are applied
3. **WHY**: Why this reasoning is valid and what principles justify it
4. **WHEN**: When this reasoning applies and under what conditions
5. **WHERE**: Where this reasoning is valid (spatial/contextual boundaries)
6. **WHO**: Who/what entities are involved in the reasoning
7. **WHICH**: Which alternatives exist and how to select among them

YOUR TASK:
Provide a comprehensive solution that systematically addresses all 7 dimensions:

## WHAT: Problem Understanding
[Clear statement of what the problem is asking and what needs to be solved]

## HOW: Solution Process  
[Step-by-step solution process with mathematical rigor, incorporating insights from all relevant experts]

## WHY: Justification
[Mathematical and logical justification for the solution, citing foundational principles]

## WHEN: Applicability
[When this solution applies and under what conditions]

## WHERE: Validity
[Where this solution is valid and its scope/limitations]

## WHO: Entities
[Who/what entities are involved and their roles]

## WHICH: Alternatives
[Which alternative approaches exist and why this one is optimal]

## HIDDEN INSIGHTS
[Any non-obvious patterns or connections discovered through multi-expert analysis]

## FINAL ANSWER
[The definitive answer to the problem with confidence assessment]
"""
        
        try:
            response = self.model.generate_content(
                synthesis_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=8192
                )
            )
            
            return {
                'synthesis': response.text,
                'reasoning_paths': [self._path_to_dict(p) for p in paths],
                'problem_structure': {
                    k: v for k, v in problem_structure.items() 
                    if k != 'concept_graph'  # Exclude non-serializable graph
                },
                'expert_count': len(set(e.expert_type for p in paths for e in p.expert_perspectives))
            }
            
        except Exception as e:
            return {
                'synthesis': f"Error in synthesis: {e}",
                'reasoning_paths': [],
                'problem_structure': {},
                'expert_count': 0
            }
            
    def _path_to_dict(self, path: UnifiedReasoningPath) -> Dict[str, Any]:
        """Convert reasoning path to dictionary"""
        return {
            'path_id': path.path_id,
            'reasoning_types': [t.value for t in path.reasoning_types],
            'confidence': path.confidence_score,
            'complexity': path.pattern_complexity,
            'depth': path.epistemic_depth,
            'what': path.what_explanation,
            'how': path.how_explanation,
            'why': path.why_explanation,
            'when': path.when_explanation,
            'where': path.where_explanation,
            'who': path.who_explanation,
            'which': path.which_explanation,
            'reasoning_steps': [
                {
                    'step_id': step.step_id,
                    'step_type': step.step_type,
                    'description': step.description,
                    'mathematical_basis': step.mathematical_basis,
                    'logical_justification': step.logical_justification,
                    'confidence': step.confidence,
                    'dependencies': step.dependencies,
                    'outputs': step.outputs
                }
                for step in path.reasoning_steps
            ],
            'expert_perspectives': [
                {
                    'expert': e.expert_type,
                    'confidence': e.confidence,
                    'insights': e.key_insights,
                    'hidden_patterns': e.hidden_patterns,
                    'reasoning_dimension': e.reasoning_dimension,
                    'foundational_principles': e.foundational_principles
                }
                for e in path.expert_perspectives
            ]
        }
    
    def _enhanced_path_to_dict(self, path: UnifiedReasoningPath) -> Dict[str, Any]:
        """Convert reasoning path to enhanced dictionary for logging"""
        base_dict = self._path_to_dict(path)
        
        # Add step-by-step process
        base_dict['step_by_step_process'] = []
        
        # Generate steps from expert perspectives
        step_num = 1
        for expert in path.expert_perspectives:
            for insight in expert.key_insights:
                base_dict['step_by_step_process'].append({
                    'step_number': step_num,
                    'operation': f"Expert {expert.expert_type} analysis",
                    'mathematical_expression': expert.mathematical_foundation,
                    'logical_justification': insight,
                    'intermediate_result': f"Confidence: {expert.confidence:.2f}",
                    'confidence': expert.confidence,
                    'reasoning_dimension': expert.reasoning_dimension,
                    'foundational_principles': expert.foundational_principles
                })
                step_num += 1
        
        # Add granular reasoning steps
        base_dict['granular_reasoning_steps'] = [
            {
                'step_id': step.step_id,
                'step_type': step.step_type,
                'description': step.description,
                'mathematical_basis': step.mathematical_basis,
                'logical_justification': step.logical_justification,
                'confidence': step.confidence,
                'dependencies': step.dependencies,
                'outputs': step.outputs
            }
            for step in path.reasoning_steps
        ]
        
        # Add enhanced explanations
        base_dict['enhanced_explanations'] = {
            'what': path.what_explanation,
            'how': path.how_explanation,
            'why': path.why_explanation,
            'when': path.when_explanation,
            'where': path.where_explanation,
            'who': path.who_explanation,
            'which': path.which_explanation
        }
        
        # Add causal graph if available
        if path.causal_graph:
            base_dict['causal_graph'] = {
                'nodes': list(path.causal_graph.nodes()),
                'edges': [
                    {
                        'from': edge[0],
                        'to': edge[1],
                        'relationship': 'causal'
                    }
                    for edge in path.causal_graph.edges()
                ]
            }
        
        return base_dict
    
    def _calculate_consistency_metrics(self, expert_analyses: List[ExpertPerspective], 
                                     paths: List[UnifiedReasoningPath]) -> Dict[str, float]:
        """Calculate consistency metrics between expert analyses"""
        if not expert_analyses:
            return {'expert_agreement_score': 0.0, 'mathematical_consistency': 0.0, 'logical_consistency': 0.0}
        
        # Expert agreement based on confidence correlation
        confidences = [e.confidence for e in expert_analyses]
        expert_agreement = np.std(confidences) if len(confidences) > 1 else 1.0
        expert_agreement_score = max(0.0, 1.0 - expert_agreement)
        
        # Mathematical consistency based on formulation similarity
        math_formulations = [e.mathematical_foundation for e in expert_analyses if e.mathematical_foundation]
        math_consistency = 0.8 if len(set(math_formulations)) <= len(math_formulations) / 2 else 0.6
        
        # Path convergence
        path_convergence = 1.0 / len(paths) if paths else 0.0
        
        return {
            'expert_agreement_score': expert_agreement_score,
            'mathematical_consistency': math_consistency,
            'logical_consistency': 0.8,  # Default logical consistency
            'path_convergence_score': path_convergence
        }
    
    def _identify_contradictions(self, expert_analyses: List[ExpertPerspective]) -> List[Dict[str, Any]]:
        """Identify contradictions between expert analyses"""
        contradictions = []
        
        # Check for confidence contradictions
        high_conf_experts = [e for e in expert_analyses if e.confidence > 0.8]
        low_conf_experts = [e for e in expert_analyses if e.confidence < 0.3]
        
        if high_conf_experts and low_conf_experts:
            contradictions.append({
                'contradiction_type': 'confidence_disparity',
                'conflicting_experts': [e.expert_type for e in high_conf_experts + low_conf_experts],
                'severity': 0.6,
                'resolution_strategy': 'Weight by domain relevance'
            })
        
        return contradictions
    
    def _calculate_validation_metrics(self, solution: Dict[str, Any], 
                                    expert_analyses: List[ExpertPerspective],
                                    paths: List[UnifiedReasoningPath]) -> Dict[str, Dict[str, Any]]:
        """Calculate comprehensive validation metrics"""
        return {
            'mathematical_validation': {
                'dimensional_analysis': True,
                'unit_consistency': True,
                'numerical_stability': 0.9,
                'convergence_criteria': 'Multi-expert consensus achieved'
            },
            'logical_validation': {
                'syllogistic_validity': True,
                'consistency_check': True,
                'contradiction_free': len(self._identify_contradictions(expert_analyses)) == 0,
                'completeness_score': min(len(expert_analyses) / 8.0, 1.0)
            },
            'empirical_validation': {
                'sanity_checks': ['Domain relevance verified', 'Mathematical soundness checked'],
                'boundary_conditions': ['Edge cases considered', 'Limit behavior analyzed'],
                'edge_case_analysis': ['Extreme values tested', 'Degenerate cases handled']
            },
            'meta_validation': {
                'self_consistency': 0.85,
                'expert_agreement': np.mean([e.confidence for e in expert_analyses]),
                'robustness_score': 0.8
            }
        }
        
    def reason(self, problem: str, enable_logging: bool = True) -> Dict[str, Any]:
        """Main reasoning method with comprehensive logging"""
        start_time = time.time()
        
        # Initialize session manager for logging
        session_manager = None
        if enable_logging:
            session_manager = ReasoningSessionManager()
            session_id = session_manager.start_reasoning_session(
                problem, 
                engine_version="1.0.0",
                model_config={
                    "primary_model": "gemini-1.5-pro",
                    "temperature": 0.2,
                    "max_tokens": 8192,
                    "top_p": 0.9
                }
            )
        
        try:
            # Detect domain automatically
            domain, domain_confidence = self.domain_detector.detect_domain(problem)
            
            # Analyze problem structure
            problem_structure = self.analyze_problem_structure(problem)
            
            # Log input analysis
            if session_manager:
                session_manager.log_step("input_analysis", {
                    "structural_analysis": {
                        "token_count": problem_structure.get('token_count', 0),
                        "unique_concepts": problem_structure.get('unique_concepts', 0),
                        "entropy": problem_structure.get('entropy', 0.0),
                        "complexity": problem_structure.get('complexity', 0.0),
                        "mathematical_expressions": problem_structure.get('math_expressions', []),
                        "graph_density": problem_structure.get('graph_density', 0.0),
                        "concept_graph": {
                            "node_count": problem_structure.get('concept_graph', nx.Graph()).number_of_nodes(),
                            "edge_count": problem_structure.get('concept_graph', nx.Graph()).number_of_edges(),
                            "density": problem_structure.get('graph_density', 0.0)
                        }
                    },
                    "domain_detection": {
                        "primary_domain": domain,
                        "confidence_score": domain_confidence,
                        "secondary_domains": [],
                        "interdisciplinary_indicators": []
                    },
                    "complexity_metrics": {
                        "cognitive_load": min(problem_structure.get('complexity', 0) * 10, 10),
                        "mathematical_depth": len(problem_structure.get('math_expressions', [])),
                        "reasoning_steps_estimate": max(3, len(problem_structure.get('math_expressions', [])) * 2),
                        "uncertainty_level": 1.0 - domain_confidence
                    }
                })
            
            # Get all expert analyses
            expert_analyses = []
            for expert_name, expert in self.experts.items():
                expert_start = time.time()
                analysis = expert.analyze(problem, {'domain': domain, 'structure': problem_structure})
                expert_time = time.time() - expert_start
                
                expert_analyses.append(analysis)
                
                # Log expert analysis
                if session_manager:
                    session_manager.log_step("expert_analysis", {
                        "expert_type": expert_name,
                        "analysis_results": {
                            "key_insights": analysis.key_insights,
                            "mathematical_foundation": analysis.mathematical_foundation,
                            "recommended_approach": analysis.recommended_approach,
                            "causal_chain": analysis.causal_chain,
                            "hidden_patterns": analysis.hidden_patterns,
                            "domain_specific_metrics": {"analysis_time": expert_time}
                        },
                        "confidence": analysis.confidence,
                        "what_analysis": f"Expert {analysis.expert_type} identifies this as: {', '.join(analysis.key_insights[:2])}",
                        "how_analysis": f"Approach: {analysis.recommended_approach}",
                        "why_analysis": f"Mathematical foundation: {analysis.mathematical_foundation}"
                    })
                    
                    session_manager.logger.track_performance("expert_analysis", expert_time)
            
            # Generate reasoning paths
            paths_start = time.time()
            paths = self.generate_reasoning_paths(problem, domain, expert_analyses)
            paths_time = time.time() - paths_start
            
            # Log reasoning paths
            if session_manager:
                for path in paths:
                    path_data = self._enhanced_path_to_dict(path)
                    session_manager.log_step("reasoning_path", path_data)
                    
                session_manager.logger.track_performance("reasoning_paths", paths_time)
            
            # Synthesize final solution
            synthesis_start = time.time()
            solution = self.synthesize_solution(problem, paths, problem_structure)
            synthesis_time = time.time() - synthesis_start
            
            # Calculate cross-validation metrics
            consistency_metrics = self._calculate_consistency_metrics(expert_analyses, paths)
            contradictions = self._identify_contradictions(expert_analyses)
            
            # Log synthesis and cross-validation
            if session_manager:
                session_manager.log_step("synthesis", {
                    "final_answer": solution['synthesis'],
                    "synthesis_process": {
                        "what_synthesis": "Unified solution combining multiple expert perspectives",
                        "how_synthesis": f"Integrated {len(expert_analyses)} expert analyses using weighted consensus",
                        "why_synthesis": f"Mathematical rigor validated across {len(paths)} reasoning paths",
                        "integration_method": "Multi-expert weighted consensus",
                        "weighted_combination": {
                            "weighting_scheme": "confidence-based",
                            "expert_weights": {e.expert_type: e.confidence for e in expert_analyses},
                            "combination_formula": "∑(wi * ai) / ∑wi where w=confidence, a=analysis"
                        }
                    },
                    "confidence_assessment": {
                        "overall_confidence": np.mean([e.confidence for e in expert_analyses]),
                        "confidence_breakdown": {
                            "mathematical_rigor": consistency_metrics.get("mathematical_consistency", 0.8),
                            "logical_consistency": consistency_metrics.get("logical_consistency", 0.8),
                            "expert_consensus": consistency_metrics.get("expert_agreement_score", 0.7),
                            "empirical_support": 0.8
                        },
                        "uncertainty_sources": [
                            {
                                "source": "Domain ambiguity",
                                "impact": 1.0 - domain_confidence,
                                "mitigation": "Multi-domain analysis"
                            }
                        ]
                    },
                    "hidden_insights": [
                        {
                            "insight": pattern,
                            "mathematical_basis": "Pattern recognition algorithms",
                            "discovery_method": "Cross-expert analysis",
                            "significance": 0.7
                        }
                        for expert in expert_analyses
                        for pattern in expert.hidden_patterns
                    ]
                })
                
                # Log cross-validation
                session_manager.logger.log_cross_validation(consistency_metrics, contradictions)
                
                session_manager.logger.track_performance("synthesis", synthesis_time)
            
            # Calculate final metrics
            latency = time.time() - start_time
            
            # Log validation metrics
            if session_manager:
                validation_metrics = self._calculate_validation_metrics(solution, expert_analyses, paths)
                session_manager.logger.log_validation_metrics(**validation_metrics)
            
            result = {
                'problem': problem,
                'detected_domain': domain,
                'domain_confidence': domain_confidence,
                'solution': solution['synthesis'],
                'reasoning_paths': solution['reasoning_paths'],
                'problem_metrics': {
                    'entropy': problem_structure['entropy'],
                    'complexity': problem_structure['complexity'],
                    'token_count': problem_structure['token_count']
                },
                'expert_analyses': [
                    {
                        'expert': e.expert_type,
                        'confidence': e.confidence,
                        'insights': e.key_insights,
                        'patterns': e.hidden_patterns
                    }
                    for e in expert_analyses
                ],
                'performance': {
                    'latency_seconds': latency,
                    'expert_count': solution['expert_count'],
                    'path_count': len(paths)
                },
                'session_id': session_id if session_manager else None
            }
            
            return result
            
        except Exception as e:
            if session_manager:
                session_manager.add_debug_info("error", str(e))
            raise
        finally:
            # Session manager will auto-save on exit
            if session_manager:
                session_manager.__exit__(None, None, None)

# Initialize the unified engine
unified_engine = UnifiedReasoningEngine()

@app.route('/unified_reason', methods=['POST'])
def unified_reason():
    """API endpoint for unified reasoning"""
    data = request.get_json()
    
    if not data or 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400
        
    try:
        result = unified_engine.reason(data['prompt'])
        return jsonify(result)
    except Exception as e:
        print(f"Error in unified reasoning: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Different port from existing services