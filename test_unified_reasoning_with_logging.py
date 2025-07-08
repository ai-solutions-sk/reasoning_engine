#!/usr/bin/env python3
"""
Test script for the Unified Reasoning Engine with comprehensive logging
Demonstrates the complete reasoning process with mathematical rigor and schema compliance
"""

import json
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from unified_reasoning_engine import unified_engine

def test_mathematical_problem():
    """Test with a mathematical optimization problem"""
    problem = """
    A factory produces two products A and B. Product A requires 2 hours of labor and 3 units of material, 
    yielding a profit of $5. Product B requires 1 hour of labor and 2 units of material, yielding a profit of $3.
    The factory has 100 hours of labor and 150 units of material available daily.
    How many units of each product should be produced to maximize profit?
    """
    
    print("🔍 Testing Mathematical Optimization Problem...")
    print(f"Problem: {problem.strip()}")
    print("\n" + "="*80 + "\n")
    
    # Run the unified reasoning engine with logging
    result = unified_engine.reason(problem, enable_logging=True)
    
    print("📊 RESULTS:")
    print(f"Domain Detected: {result['detected_domain']} (confidence: {result['domain_confidence']:.2f})")
    print(f"Solution: {result['solution'][:200]}...")
    print(f"Expert Count: {result['performance']['expert_count']}")
    print(f"Reasoning Paths: {result['performance']['path_count']}")
    print(f"Processing Time: {result['performance']['latency_seconds']:.2f} seconds")
    
    if result['session_id']:
        print(f"Session ID: {result['session_id']}")
        print(f"Detailed log saved automatically")
    
    return result

def test_logical_reasoning():
    """Test with a logical reasoning problem"""
    problem = """
    If all cats are mammals, and all mammals are animals, and Fluffy is a cat,
    what can we conclude about Fluffy? Explain the logical reasoning process.
    """
    
    print("\n🧠 Testing Logical Reasoning Problem...")
    print(f"Problem: {problem.strip()}")
    print("\n" + "="*80 + "\n")
    
    result = unified_engine.reason(problem, enable_logging=True)
    
    print("📊 RESULTS:")
    print(f"Domain Detected: {result['detected_domain']} (confidence: {result['domain_confidence']:.2f})")
    print(f"Solution: {result['solution'][:200]}...")
    print(f"Expert Analyses:")
    for expert in result['expert_analyses']:
        print(f"  - {expert['expert']}: {expert['confidence']:.2f} confidence")
        if expert['insights']:
            print(f"    Insights: {expert['insights'][0]}")
    
    return result

def test_complex_interdisciplinary():
    """Test with a complex interdisciplinary problem"""
    problem = """
    A pharmaceutical company is developing a new drug delivery system using nanotechnology.
    The drug follows first-order kinetics with a half-life of 4 hours. The nanoparticles
    have a spherical shape with radius r, and the drug release rate is proportional to
    the surface area. If the therapeutic window requires maintaining drug concentration
    between 50-200 ng/mL for 12 hours, and the initial dose is 1000 ng, calculate the
    optimal nanoparticle radius and dosing schedule. Consider both pharmacokinetic
    modeling and optimization constraints.
    """
    
    print("\n🔬 Testing Complex Interdisciplinary Problem...")
    print(f"Problem: {problem.strip()}")
    print("\n" + "="*80 + "\n")
    
    result = unified_engine.reason(problem, enable_logging=True)
    
    print("📊 RESULTS:")
    print(f"Domain Detected: {result['detected_domain']} (confidence: {result['domain_confidence']:.2f})")
    print(f"Solution: {result['solution'][:300]}...")
    print(f"Problem Metrics:")
    print(f"  - Entropy: {result['problem_metrics']['entropy']:.2f}")
    print(f"  - Complexity: {result['problem_metrics']['complexity']:.2f}")
    print(f"  - Token Count: {result['problem_metrics']['token_count']}")
    
    print(f"\nExpert Confidence Scores:")
    for expert in result['expert_analyses']:
        if expert['confidence'] > 0.5:
            print(f"  - {expert['expert']}: {expert['confidence']:.2f}")
            if expert['patterns']:
                print(f"    Hidden Patterns: {expert['patterns'][:2]}")
    
    return result

def demonstrate_schema_compliance():
    """Demonstrate that the logging follows the comprehensive schema"""
    print("\n📋 Demonstrating Schema Compliance...")
    
    # Run a simple test
    result = unified_engine.reason("What is 2 + 2?", enable_logging=True)
    
    # Load the most recent log file
    log_dir = Path("reasoning_logs")
    if log_dir.exists():
        log_files = list(log_dir.glob("reasoning_session_*.json"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            
            print(f"Loading log file: {latest_log}")
            
            with open(latest_log, 'r') as f:
                log_data = json.load(f)
            
            print("\n🏗️  Schema Structure Validation:")
            required_sections = [
                "session_metadata",
                "input_analysis", 
                "reasoning_process",
                "mathematical_foundations",
                "output_synthesis",
                "performance_metrics",
                "validation_metrics"
            ]
            
            for section in required_sections:
                if section in log_data:
                    print(f"  ✅ {section}")
                    if section == "reasoning_process":
                        print(f"     - Expert analyses: {len(log_data[section].get('expert_analyses', []))}")
                        print(f"     - Reasoning paths: {len(log_data[section].get('reasoning_paths', []))}")
                    elif section == "performance_metrics":
                        latency = log_data[section]['latency_metrics']['total_latency_seconds']
                        print(f"     - Total latency: {latency:.3f}s")
                else:
                    print(f"  ❌ {section} (missing)")
            
            # Show sample of mathematical foundations
            math_found = log_data.get("mathematical_foundations", {})
            print(f"\n🔢 Mathematical Foundations Sample:")
            for component, data in math_found.items():
                if data:
                    print(f"  - {component}: {type(data).__name__}")
            
            # Show validation metrics
            validation = log_data.get("validation_metrics", {})
            print(f"\n✅ Validation Metrics:")
            for metric_type, metrics in validation.items():
                print(f"  - {metric_type}: {len(metrics) if isinstance(metrics, dict) else 'N/A'} metrics")
            
            return log_data
    
    print("No log files found.")
    return None

def main():
    """Main test function"""
    print("🚀 UNIFIED REASONING ENGINE - COMPREHENSIVE LOGGING TEST")
    print("="*80)
    
    try:
        # Test 1: Mathematical Problem
        math_result = test_mathematical_problem()
        
        # Test 2: Logical Reasoning  
        logic_result = test_logical_reasoning()
        
        # Test 3: Complex Interdisciplinary
        complex_result = test_complex_interdisciplinary()
        
        # Test 4: Schema Compliance
        log_data = demonstrate_schema_compliance()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\n📁 Log files saved in 'reasoning_logs/' directory")
        print("📊 Each log contains comprehensive reasoning traces following the unified schema")
        print("🔍 Review the JSON files to see the complete mathematical and logical foundations")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 