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
from sympy import symbols, solve, diff, integrate, simplify, Matrix, eigenvals, eigenvects
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

# Configuration
API_KEY = "AIzaSyBLXXuiqpx9BfDxGi28Ci8szlsb3qAm9Dw"
genai.configure(api_key=API_KEY)

app = Flask(__name__)
CORS(app)

# Enhanced Reasoning Types combining both engines
class UnifiedReasoningType(Enum):
    # Mathematical/Logical
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
    # Domain-Specific
    LINEAR_ALGEBRAIC = "linear_algebraic"
    CALCULUS_BASED = "calculus_based"
    STATISTICAL = "statistical"
    OPTIMIZATION = "optimization"
    GAME_THEORETIC = "game_theoretic"
    TOPOLOGICAL = "topological"
    CATEGORY_THEORETIC = "category_theoretic"
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
    mathematical_formulation: str
    recommended_approach: str
    causal_chain: List[str]
    hidden_patterns: List[str]

@dataclass
class UnifiedReasoningPath:
    """Enhanced reasoning path with multi-expert validation"""
    path_id: str
    reasoning_types: List[UnifiedReasoningType]
    expert_perspectives: List[ExpertPerspective]
    mathematical_foundation: str
    logical_structure: str
    confidence_score: float
    pattern_complexity: int
    epistemic_depth: int
    causal_graph: Optional[nx.DiGraph]
    what_explanation: str  # What is being reasoned
    how_explanation: str   # How the reasoning works
    why_explanation: str   # Why this reasoning is valid

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

class ExpertLinearAlgebra:
    """Expert in Linear Algebra"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_formulation = ""
        causal_chain = []
        hidden_patterns = []
        
        # Detect matrix operations
        if any(term in problem.lower() for term in ['matrix', 'vector', 'eigenvalue', 'linear transformation', 'orthogonal']):
            insights.append("Problem involves linear transformations or matrix operations")
            mathematical_formulation = "Ax = λx (eigenvalue problem) or Ax = b (linear system)"
            causal_chain.append("Linear transformation → Change of basis → Eigendecomposition → Solution")
            
            # Look for hidden linear structures
            if 'optimization' in problem.lower():
                hidden_patterns.append("Linear programming structure hidden in constraints")
            
        # Detect vector spaces
        if any(term in problem.lower() for term in ['space', 'dimension', 'basis', 'span', 'orthogonal']):
            insights.append("Problem involves vector space concepts")
            mathematical_formulation += " | V = span{v1, v2, ..., vn}"
            
        return ExpertPerspective(
            expert_type="Linear Algebra",
            confidence=0.8 if insights else 0.2,
            key_insights=insights,
            mathematical_formulation=mathematical_formulation,
            recommended_approach="Apply matrix decomposition techniques or basis transformation",
            causal_chain=causal_chain,
            hidden_patterns=hidden_patterns
        )

class ExpertCalculus:
    """Expert in Calculus"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_formulation = ""
        causal_chain = []
        hidden_patterns = []
        
        # Detect calculus concepts
        if any(term in problem.lower() for term in ['rate', 'change', 'derivative', 'integral', 'limit', 'continuous', 'maximize', 'minimize']):
            insights.append("Problem involves rates of change or optimization")
            mathematical_formulation = "f'(x) = 0 for critical points, ∫f(x)dx for accumulation"
            causal_chain.append("Function → Derivative → Critical points → Optimization")
            
            # Detect hidden calculus patterns
            if 'accumulate' in problem.lower() or 'total' in problem.lower():
                hidden_patterns.append("Integration pattern: accumulation over time/space")
            
        # Detect differential equations
        if 'differential' in problem.lower() or re.search(r'd[yxz]/d[txyz]', problem):
            insights.append("Problem involves differential equations")
            mathematical_formulation += " | dy/dx = f(x,y)"
            
        return ExpertPerspective(
            expert_type="Calculus",
            confidence=0.85 if insights else 0.15,
            key_insights=insights,
            mathematical_formulation=mathematical_formulation,
            recommended_approach="Apply differentiation/integration or solve differential equations",
            causal_chain=causal_chain,
            hidden_patterns=hidden_patterns
        )

class ExpertStatistics:
    """Expert in Statistics"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_formulation = ""
        causal_chain = []
        hidden_patterns = []
        
        # Detect statistical concepts
        if any(term in problem.lower() for term in ['probability', 'distribution', 'mean', 'variance', 'correlation', 'hypothesis', 'significance', 'sample']):
            insights.append("Problem involves statistical analysis or probability")
            mathematical_formulation = "P(A|B) = P(B|A)P(A)/P(B), X ~ N(μ, σ²)"
            causal_chain.append("Data → Distribution → Inference → Decision")
            
            # Detect hidden statistical patterns
            if 'predict' in problem.lower():
                hidden_patterns.append("Regression pattern: predictive modeling opportunity")
            
        # Detect Bayesian reasoning
        if any(term in problem.lower() for term in ['prior', 'posterior', 'evidence', 'belief']):
            insights.append("Bayesian inference applicable")
            mathematical_formulation += " | P(H|E) ∝ P(E|H)P(H)"
            
        return ExpertPerspective(
            expert_type="Statistics",
            confidence=0.9 if insights else 0.1,
            key_insights=insights,
            mathematical_formulation=mathematical_formulation,
            recommended_approach="Apply statistical inference or probabilistic modeling",
            causal_chain=causal_chain,
            hidden_patterns=hidden_patterns
        )

class ExpertLinearProgramming:
    """Expert in Linear Programming and Optimization"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_formulation = ""
        causal_chain = []
        hidden_patterns = []
        
        # Detect optimization concepts
        if any(term in problem.lower() for term in ['maximize', 'minimize', 'optimize', 'constraint', 'objective', 'feasible', 'resource']):
            insights.append("Problem is an optimization problem")
            mathematical_formulation = "max/min c^T x subject to Ax ≤ b, x ≥ 0"
            causal_chain.append("Objective → Constraints → Feasible region → Optimal solution")
            
            # Detect specific optimization types
            if 'integer' in problem.lower() or 'whole' in problem.lower():
                hidden_patterns.append("Integer programming: discrete optimization required")
            
        # Detect resource allocation
        if any(term in problem.lower() for term in ['allocate', 'distribute', 'assign', 'schedule']):
            insights.append("Resource allocation problem detected")
            mathematical_formulation += " | Assignment or scheduling formulation"
            
        return ExpertPerspective(
            expert_type="Linear Programming",
            confidence=0.95 if insights else 0.05,
            key_insights=insights,
            mathematical_formulation=mathematical_formulation,
            recommended_approach="Formulate as LP/IP and solve using simplex or branch-and-bound",
            causal_chain=causal_chain,
            hidden_patterns=hidden_patterns
        )

class ExpertLogicReasoning:
    """Expert in Logic and Formal Reasoning"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_formulation = ""
        causal_chain = []
        hidden_patterns = []
        
        # Detect logical structures
        if any(term in problem.lower() for term in ['if', 'then', 'implies', 'therefore', 'contradiction', 'premise', 'conclusion']):
            insights.append("Problem involves formal logical reasoning")
            mathematical_formulation = "P → Q, P ⊢ Q (modus ponens)"
            causal_chain.append("Premises → Logical rules → Inference → Conclusion")
            
            # Detect paradoxes or self-reference
            if 'paradox' in problem.lower() or problem.count('self') > 1:
                hidden_patterns.append("Self-referential structure: potential paradox")
            
        # Detect modal logic
        if any(term in problem.lower() for term in ['possible', 'necessary', 'must', 'might', 'could']):
            insights.append("Modal logic applicable")
            mathematical_formulation += " | □P (necessary), ◊P (possible)"
            
        return ExpertPerspective(
            expert_type="Logic and Reasoning",
            confidence=0.85 if insights else 0.15,
            key_insights=insights,
            mathematical_formulation=mathematical_formulation,
            recommended_approach="Apply formal logic rules and inference systems",
            causal_chain=causal_chain,
            hidden_patterns=hidden_patterns
        )

class ExpertCausalReasoning:
    """Expert in Causal Reasoning"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_formulation = ""
        causal_chain = []
        hidden_patterns = []
        
        # Detect causal language
        if any(term in problem.lower() for term in ['cause', 'effect', 'because', 'lead to', 'result in', 'influence', 'impact']):
            insights.append("Problem involves causal relationships")
            mathematical_formulation = "P(Y|do(X)) ≠ P(Y|X) (causal vs observational)"
            causal_chain.append("Cause → Mechanism → Effect → Outcome")
            
            # Detect confounders
            if 'correlation' in problem.lower():
                hidden_patterns.append("Correlation vs causation: check for confounders")
            
        # Detect feedback loops
        if any(term in problem.lower() for term in ['feedback', 'cycle', 'recursive', 'self-reinforcing']):
            insights.append("Feedback loops present in causal structure")
            mathematical_formulation += " | Xt+1 = f(Xt) (dynamic system)"
            
        return ExpertPerspective(
            expert_type="Causal Reasoning",
            confidence=0.8 if insights else 0.2,
            key_insights=insights,
            mathematical_formulation=mathematical_formulation,
            recommended_approach="Build causal DAG and apply do-calculus",
            causal_chain=causal_chain,
            hidden_patterns=hidden_patterns
        )

class ExpertPatternFinder:
    """Expert in finding hidden patterns using mathematics and language"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_formulation = ""
        causal_chain = []
        hidden_patterns = []
        
        # Mathematical pattern detection
        numbers = re.findall(r'\d+', problem)
        if len(numbers) > 2:
            # Check for arithmetic/geometric sequences
            nums = [int(n) for n in numbers[:5]]  # First 5 numbers
            if len(nums) >= 3:
                diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
                if len(set(diffs)) == 1:
                    hidden_patterns.append(f"Arithmetic sequence detected: common difference = {diffs[0]}")
                    
        # Language pattern detection
        words = problem.lower().split()
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # Find repeated patterns
        repeated = [w for w, c in word_freq.items() if c > 2]
        if repeated:
            hidden_patterns.append(f"Linguistic repetition pattern: {', '.join(repeated)}")
            
        # Symmetry detection
        if problem == problem[::-1]:
            hidden_patterns.append("Palindromic structure detected")
            
        # Mathematical structure in language
        math_terms = ['equal', 'same', 'different', 'more', 'less', 'between']
        math_count = sum(1 for term in math_terms if term in problem.lower())
        if math_count > 2:
            insights.append("Hidden mathematical relationships in natural language")
            
        return ExpertPerspective(
            expert_type="Pattern Finder",
            confidence=0.7 if hidden_patterns else 0.3,
            key_insights=insights,
            mathematical_formulation="Pattern entropy: H(X) = -Σ p(xi)log(p(xi))",
            recommended_approach="Apply pattern recognition algorithms and information theory",
            causal_chain=["Input → Pattern extraction → Structure analysis → Hidden insights"],
            hidden_patterns=hidden_patterns
        )

class ExpertAIScientist:
    """Expert AI Scientist focused on LLMs"""
    
    def analyze(self, problem: str, context: Dict[str, Any]) -> ExpertPerspective:
        insights = []
        mathematical_formulation = ""
        causal_chain = []
        hidden_patterns = []
        
        # Detect LLM-relevant aspects
        tokens = problem.split()
        token_count = len(tokens)
        
        # Attention mechanism relevance
        if token_count > 50:
            insights.append("Long-range dependencies require attention mechanisms")
            mathematical_formulation = "Attention(Q,K,V) = softmax(QK^T/√d)V"
            
        # Prompt engineering patterns
        if '?' in problem:
            insights.append("Question-answering task: employ QA-optimized prompting")
        if any(term in problem.lower() for term in ['explain', 'describe', 'analyze']):
            insights.append("Explanation task: use chain-of-thought prompting")
            
        # Detect ambiguity
        ambiguous_terms = ['it', 'they', 'this', 'that', 'these', 'those']
        ambiguity_score = sum(1 for term in ambiguous_terms if term in problem.lower())
        if ambiguity_score > 2:
            hidden_patterns.append("High ambiguity: requires context resolution")
            
        # Token efficiency
        unique_tokens = len(set(tokens))
        compression_ratio = unique_tokens / token_count
        if compression_ratio < 0.5:
            hidden_patterns.append(f"High redundancy: compression ratio = {compression_ratio:.2f}")
            
        return ExpertPerspective(
            expert_type="AI/LLM Scientist",
            confidence=0.9,
            key_insights=insights,
            mathematical_formulation=mathematical_formulation,
            recommended_approach="Optimize prompt structure and apply appropriate LLM techniques",
            causal_chain=["Tokens → Embeddings → Attention → Transformations → Output"],
            hidden_patterns=hidden_patterns
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
            'linear_algebra': ExpertLinearAlgebra(),
            'calculus': ExpertCalculus(),
            'statistics': ExpertStatistics(),
            'linear_programming': ExpertLinearProgramming(),
            'logic': ExpertLogicReasoning(),
            'causal': ExpertCausalReasoning(),
            'pattern': ExpertPatternFinder(),
            'ai_scientist': ExpertAIScientist()
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
        """Generate multiple reasoning paths combining expert insights"""
        paths = []
        
        # Combine high-confidence expert perspectives
        high_confidence_experts = [e for e in expert_analyses if e.confidence > 0.7]
        
        if high_confidence_experts:
            # Create primary reasoning path
            primary_types = []
            if any('linear' in e.expert_type.lower() for e in high_confidence_experts):
                primary_types.append(UnifiedReasoningType.LINEAR_ALGEBRAIC)
            if any('calculus' in e.expert_type.lower() for e in high_confidence_experts):
                primary_types.append(UnifiedReasoningType.CALCULUS_BASED)
            if any('statistic' in e.expert_type.lower() for e in high_confidence_experts):
                primary_types.append(UnifiedReasoningType.STATISTICAL)
            if any('programming' in e.expert_type.lower() for e in high_confidence_experts):
                primary_types.append(UnifiedReasoningType.OPTIMIZATION)
            if any('logic' in e.expert_type.lower() for e in high_confidence_experts):
                primary_types.append(UnifiedReasoningType.DEDUCTIVE)
            if any('causal' in e.expert_type.lower() for e in high_confidence_experts):
                primary_types.append(UnifiedReasoningType.CAUSAL)
                
            # Build causal graph
            causal_graph = nx.DiGraph()
            for expert in high_confidence_experts:
                for i in range(len(expert.causal_chain) - 1):
                    causal_graph.add_edge(expert.causal_chain[i], expert.causal_chain[i+1])
                    
            # Create unified path
            path = UnifiedReasoningPath(
                path_id=f"unified_{hash(problem) % 10000}",
                reasoning_types=primary_types or [UnifiedReasoningType.DEDUCTIVE],
                expert_perspectives=high_confidence_experts,
                mathematical_foundation=" ∧ ".join([e.mathematical_formulation for e in high_confidence_experts if e.mathematical_formulation]),
                logical_structure=f"Multi-expert consensus with {len(high_confidence_experts)} perspectives",
                confidence_score=np.mean([e.confidence for e in high_confidence_experts]),
                pattern_complexity=len(set(p for e in high_confidence_experts for p in e.hidden_patterns)),
                epistemic_depth=max(len(e.causal_chain) for e in high_confidence_experts),
                causal_graph=causal_graph,
                what_explanation=f"Analyzing: {problem[:100]}...",
                how_explanation=f"Combining {len(high_confidence_experts)} expert analyses using {', '.join([t.value for t in primary_types])} reasoning",
                why_explanation=f"High confidence ({np.mean([e.confidence for e in high_confidence_experts]):.2f}) from multiple expert validations"
            )
            paths.append(path)
            
        # Add fallback general reasoning path
        if not paths:
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
                what_explanation=f"Exploring: {problem[:100]}...",
                how_explanation="Using general abductive reasoning to find best explanation",
                why_explanation="No high-confidence expert paths identified"
            ))
            
        return paths
        
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
You are a unified reasoning system with expertise across all domains of human knowledge. You have access to analyses from 8 different expert perspectives:
1. Linear Algebra Expert
2. Calculus Expert  
3. Statistics Expert
4. Linear Programming Expert
5. Logic & Reasoning Expert
6. Causal Reasoning Expert
7. Pattern Finding Expert (Mathematics + Language)
8. AI/LLM Scientist

PROBLEM: {problem}

EXPERT ANALYSES AND REASONING PATHS:
{json.dumps(path_summaries, indent=2)}

PROBLEM STRUCTURE:
- Entropy: {problem_structure['entropy']:.2f}
- Complexity: {problem_structure['complexity']:.2f}
- Mathematical expressions found: {problem_structure['math_expressions']}

YOUR TASK:
Provide a comprehensive solution that:
1. **WHAT**: Clearly state what the problem is asking and what needs to be solved
2. **HOW**: Explain step-by-step how to solve it, incorporating insights from all relevant experts
3. **WHY**: Justify why this approach is correct, citing mathematical principles and logical foundations

Structure your response as:

## WHAT: Problem Understanding
[Clear statement of the problem]

## HOW: Solution Process  
[Step-by-step solution with mathematical rigor]

## WHY: Justification
[Mathematical and logical justification for the solution]

## HIDDEN INSIGHTS
[Any non-obvious patterns or connections discovered]

## FINAL ANSWER
[The definitive answer to the problem]
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
            'expert_perspectives': [
                {
                    'expert': e.expert_type,
                    'confidence': e.confidence,
                    'insights': e.key_insights,
                    'hidden_patterns': e.hidden_patterns
                }
                for e in path.expert_perspectives
            ]
        }
        
    def reason(self, problem: str) -> Dict[str, Any]:
        """Main reasoning method"""
        start_time = time.time()
        
        # Detect domain automatically
        domain, domain_confidence = self.domain_detector.detect_domain(problem)
        
        # Analyze problem structure
        problem_structure = self.analyze_problem_structure(problem)
        
        # Get all expert analyses
        expert_analyses = []
        for expert_name, expert in self.experts.items():
            analysis = expert.analyze(problem, {'domain': domain, 'structure': problem_structure})
            expert_analyses.append(analysis)
            
        # Generate reasoning paths
        paths = self.generate_reasoning_paths(problem, domain, expert_analyses)
        
        # Synthesize final solution
        solution = self.synthesize_solution(problem, paths, problem_structure)
        
        # Calculate metrics
        latency = time.time() - start_time
        
        return {
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
            }
        }

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