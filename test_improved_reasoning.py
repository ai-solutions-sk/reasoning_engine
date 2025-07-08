#!/usr/bin/env python3
"""
Test script for the improved Unified Reasoning Engine
Demonstrates the new foundational reasoning experts and granular reasoning framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_reasoning_engine import UnifiedReasoningEngine
import json
import time

def test_improved_reasoning_engine():
    """Test the improved reasoning engine with various problem types"""
    
    print("🧠 Testing Improved Unified Reasoning Engine")
    print("=" * 60)
    
    # Initialize the engine
    engine = UnifiedReasoningEngine()
    
    # Test problems covering different reasoning dimensions
    test_problems = [
        {
            "name": "Mathematical Foundation Problem",
            "problem": "Prove that the set of all subsets of a finite set S has cardinality 2^|S| using set theory and induction.",
            "expected_dimensions": ["positive", "mathematical_foundation"]
        },
        {
            "name": "Negative Reasoning Problem", 
            "problem": "Show that it is impossible to construct a regular heptagon using only compass and straightedge.",
            "expected_dimensions": ["negative", "mathematical_foundation"]
        },
        {
            "name": "Boundary Reasoning Problem",
            "problem": "Find the critical points and determine the maximum value of f(x) = x^3 - 3x^2 + 2 on the interval [0, 3].",
            "expected_dimensions": ["boundary", "transitional"]
        },
        {
            "name": "Transitional Reasoning Problem",
            "problem": "A particle moves along a curve with velocity v(t) = 2t + 1. Find the position function and determine when the particle changes direction.",
            "expected_dimensions": ["transitional", "boundary"]
        },
        {
            "name": "Emergent Reasoning Problem",
            "problem": "In a network of interacting agents, explain how collective behavior emerges from individual interactions and identify phase transitions.",
            "expected_dimensions": ["emergent", "causal"]
        },
        {
            "name": "Logical Structure Problem",
            "problem": "Prove that if all humans are mortal and Socrates is human, then Socrates is mortal using formal logic.",
            "expected_dimensions": ["logical_structure", "positive"]
        },
        {
            "name": "Causal Reasoning Problem",
            "problem": "Analyze the causal relationship between smoking and lung cancer, considering confounding variables and counterfactual scenarios.",
            "expected_dimensions": ["causal", "negative"]
        },
        {
            "name": "Pattern Recognition Problem",
            "problem": "Identify the pattern in the sequence 2, 6, 12, 20, 30, ... and find an analogous pattern in nature.",
            "expected_dimensions": ["pattern", "positive"]
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_problems, 1):
        print(f"\n🔍 Test {i}: {test_case['name']}")
        print("-" * 40)
        print(f"Problem: {test_case['problem']}")
        print(f"Expected dimensions: {test_case['expected_dimensions']}")
        
        try:
            # Run reasoning
            start_time = time.time()
            result = engine.reason(test_case['problem'], enable_logging=False)
            reasoning_time = time.time() - start_time
            
            # Extract key information
            expert_analyses = result.get('expert_analyses', [])
            reasoning_paths = result.get('reasoning_paths', [])
            
            print(f"\n⏱️  Reasoning time: {reasoning_time:.2f} seconds")
            print(f"🎯 Detected domain: {result.get('detected_domain', 'unknown')} (confidence: {result.get('domain_confidence', 0):.2f})")
            
            # Analyze expert perspectives
            print(f"\n👥 Expert Analyses ({len(expert_analyses)} experts):")
            for expert in expert_analyses:
                dimension = expert.get('reasoning_dimension', 'unknown')
                confidence = expert.get('confidence', 0)
                insights = expert.get('insights', [])
                principles = expert.get('foundational_principles', [])
                
                print(f"  • {expert['expert']} ({dimension}): {confidence:.2f}")
                if insights:
                    print(f"    Insights: {insights[0] if insights else 'None'}")
                if principles:
                    print(f"    Principles: {principles[0] if principles else 'None'}")
            
            # Analyze reasoning paths
            if reasoning_paths:
                path = reasoning_paths[0]  # Primary path
                print(f"\n🛤️  Primary Reasoning Path:")
                print(f"  • Types: {', '.join(path.get('reasoning_types', []))}")
                print(f"  • Confidence: {path.get('confidence', 0):.2f}")
                print(f"  • Complexity: {path.get('complexity', 0)}")
                
                # Show granular reasoning steps
                steps = path.get('reasoning_steps', [])
                if steps:
                    print(f"  • Granular Steps ({len(steps)}):")
                    for step in steps[:3]:  # Show first 3 steps
                        print(f"    - {step['step_type'].upper()}: {step['description'][:60]}...")
                
                # Show enhanced explanations
                print(f"\n📝 Enhanced Explanations:")
                explanations = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
                for exp_type in explanations:
                    explanation = path.get(exp_type, '')
                    if explanation:
                        print(f"  • {exp_type.upper()}: {explanation[:80]}...")
            
            # Check if expected dimensions were detected
            detected_dimensions = [expert.get('reasoning_dimension', '') for expert in expert_analyses]
            expected = test_case['expected_dimensions']
            matches = [dim for dim in expected if any(exp_dim in dim for exp_dim in detected_dimensions)]
            
            print(f"\n✅ Dimension Match: {len(matches)}/{len(expected)} expected dimensions detected")
            if matches:
                print(f"   Matched: {', '.join(matches)}")
            
            results.append({
                'test_name': test_case['name'],
                'success': True,
                'reasoning_time': reasoning_time,
                'expert_count': len(expert_analyses),
                'dimension_matches': len(matches),
                'expected_dimensions': expected,
                'detected_dimensions': detected_dimensions
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'test_name': test_case['name'],
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 60)
    
    successful_tests = [r for r in results if r['success']]
    if successful_tests:
        avg_time = sum(r['reasoning_time'] for r in successful_tests) / len(successful_tests)
        avg_experts = sum(r['expert_count'] for r in successful_tests) / len(successful_tests)
        avg_dimensions = sum(r['dimension_matches'] for r in successful_tests) / len(successful_tests)
        
        print(f"✅ Successful tests: {len(successful_tests)}/{len(test_problems)}")
        print(f"⏱️  Average reasoning time: {avg_time:.2f} seconds")
        print(f"👥 Average experts involved: {avg_experts:.1f}")
        print(f"🎯 Average dimension matches: {avg_dimensions:.1f}")
        
        # Show best performing test
        best_test = max(successful_tests, key=lambda x: x['dimension_matches'])
        print(f"🏆 Best test: {best_test['test_name']} ({best_test['dimension_matches']} dimensions matched)")
    
    failed_tests = [r for r in results if not r['success']]
    if failed_tests:
        print(f"❌ Failed tests: {len(failed_tests)}")
        for test in failed_tests:
            print(f"   • {test['test_name']}: {test['error']}")
    
    return results

def demonstrate_mathematical_foundations():
    """Demonstrate the mathematical foundation translation framework"""
    
    print(f"\n🔬 Mathematical Foundation Translation Framework")
    print("=" * 60)
    
    # Show how mathematical concepts translate to reasoning processes
    mathematical_concepts = {
        "Set Theory": {
            "concept": "A ⊆ B (subset relation)",
            "reasoning_translation": "Constraint reasoning - what elements must be included/excluded",
            "expert_dimension": "negative",
            "foundational_principle": "Axiom of Extensionality"
        },
        "Order Theory": {
            "concept": "≤ (partial order)",
            "reasoning_translation": "Hierarchical reasoning - establishing precedence and relationships",
            "expert_dimension": "boundary",
            "foundational_principle": "Transitivity of relations"
        },
        "Group Theory": {
            "concept": "(G, *) with closure, associativity, identity, inverse",
            "reasoning_translation": "Constructive reasoning - building solutions from basic operations",
            "expert_dimension": "positive",
            "foundational_principle": "Closure under operation"
        },
        "Measure Theory": {
            "concept": "μ: Σ → [0,∞] with countable additivity",
            "reasoning_translation": "Quantitative reasoning - measuring and aggregating information",
            "expert_dimension": "boundary",
            "foundational_principle": "Countable additivity"
        },
        "Dynamical Systems": {
            "concept": "dx/dt = f(x,t)",
            "reasoning_translation": "Process reasoning - understanding how systems evolve over time",
            "expert_dimension": "transitional",
            "foundational_principle": "Differential equations"
        },
        "Complexity Theory": {
            "concept": "Emergence: properties not reducible to components",
            "reasoning_translation": "Emergent reasoning - understanding system-level properties",
            "expert_dimension": "emergent",
            "foundational_principle": "Emergence principle"
        }
    }
    
    for concept_name, details in mathematical_concepts.items():
        print(f"\n📐 {concept_name}")
        print(f"   Mathematical Concept: {details['concept']}")
        print(f"   → Reasoning Translation: {details['reasoning_translation']}")
        print(f"   → Expert Dimension: {details['expert_dimension']}")
        print(f"   → Foundational Principle: {details['foundational_principle']}")

def demonstrate_granular_reasoning():
    """Demonstrate the granular reasoning framework"""
    
    print(f"\n🔍 Granular Reasoning Framework (7 Dimensions)")
    print("=" * 60)
    
    reasoning_dimensions = {
        "WHAT": {
            "description": "What is being reasoned about and what needs to be solved",
            "mathematical_basis": "Problem formulation: P(x) → find x such that P(x)",
            "example": "Understanding the structure of a mathematical proof"
        },
        "HOW": {
            "description": "How the reasoning process works and what methods are applied",
            "mathematical_basis": "Algorithmic approach: systematic step-by-step process",
            "example": "Applying induction to prove a theorem"
        },
        "WHY": {
            "description": "Why this reasoning is valid and what principles justify it",
            "mathematical_basis": "Mathematical and logical foundations",
            "example": "Justifying induction using the well-ordering principle"
        },
        "WHEN": {
            "description": "When this reasoning applies and under what conditions",
            "mathematical_basis": "Domain of validity and applicability conditions",
            "example": "Induction applies to well-ordered sets"
        },
        "WHERE": {
            "description": "Where this reasoning is valid (spatial/contextual boundaries)",
            "mathematical_basis": "Spatial and contextual boundaries",
            "example": "Valid in mathematical domains with appropriate axioms"
        },
        "WHO": {
            "description": "Who/what entities are involved in the reasoning",
            "mathematical_basis": "Entity modeling and agent identification",
            "example": "Mathematical objects, variables, and logical operators"
        },
        "WHICH": {
            "description": "Which alternatives exist and how to select among them",
            "mathematical_basis": "Decision theory and optimization",
            "example": "Choosing between direct proof, contradiction, or induction"
        }
    }
    
    for dimension, details in reasoning_dimensions.items():
        print(f"\n❓ {dimension}")
        print(f"   Description: {details['description']}")
        print(f"   Mathematical Basis: {details['mathematical_basis']}")
        print(f"   Example: {details['example']}")

if __name__ == "__main__":
    # Run comprehensive tests
    test_results = test_improved_reasoning_engine()
    
    # Demonstrate mathematical foundations
    demonstrate_mathematical_foundations()
    
    # Demonstrate granular reasoning
    demonstrate_granular_reasoning()
    
    print(f"\n🎉 Improved Unified Reasoning Engine Test Complete!")
    print("The engine now features:")
    print("• 8 foundational reasoning experts covering different dimensions")
    print("• Mathematical foundation translation framework")
    print("• 7-dimensional granular reasoning (WHAT/HOW/WHY/WHEN/WHERE/WHO/WHICH)")
    print("• Enhanced reasoning paths with step-by-step granularity")