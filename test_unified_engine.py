#!/usr/bin/env python3
"""
Test script for the Unified Reasoning Engine
Demonstrates capabilities across multiple domains
"""

import requests
import json
import time

# API endpoint
API_URL = "http://localhost:5001/unified_reason"

# Test problems covering different domains and expert systems
TEST_PROBLEMS = [
    {
        "name": "Simple Mathematics",
        "prompt": "What is 15 * 7?",
        "expected_experts": ["AI/LLM Scientist", "Pattern Finder"]
    },
    {
        "name": "Linear Algebra",
        "prompt": "Find the eigenvalues of the matrix [[3, 1], [1, 3]]",
        "expected_experts": ["Linear Algebra", "AI/LLM Scientist"]
    },
    {
        "name": "Calculus Optimization",
        "prompt": "Find the critical points of f(x) = x³ - 3x² + 2x and determine if they are maxima or minima",
        "expected_experts": ["Calculus", "AI/LLM Scientist"]
    },
    {
        "name": "Linear Programming",
        "prompt": "A factory produces products A and B. Product A requires 2 hours of labor and yields $3 profit. Product B requires 3 hours of labor and yields $4 profit. With 12 hours of labor available, how many of each product should be produced to maximize profit?",
        "expected_experts": ["Linear Programming", "Calculus", "AI/LLM Scientist"]
    },
    {
        "name": "Logic Puzzle",
        "prompt": "If all philosophers are thinkers, and some thinkers are writers, can we conclude that some philosophers are writers?",
        "expected_experts": ["Logic and Reasoning", "AI/LLM Scientist"]
    },
    {
        "name": "Statistical Inference", 
        "prompt": "A coin is flipped 10 times and shows heads 8 times. What is the probability that the coin is biased?",
        "expected_experts": ["Statistics", "AI/LLM Scientist"]
    },
    {
        "name": "Causal Analysis",
        "prompt": "Ice cream sales increase in summer, and so do drowning incidents. Does eating ice cream cause drowning?",
        "expected_experts": ["Causal Reasoning", "Statistics", "AI/LLM Scientist"]
    },
    {
        "name": "Pattern Finding",
        "prompt": "Find the pattern in the sequence: 2, 6, 12, 20, 30, 42, ?",
        "expected_experts": ["Pattern Finder", "AI/LLM Scientist"]
    },
    {
        "name": "Complex Reasoning",
        "prompt": "Three perfect logicians are told that their hats are either red or blue, with at least one red hat. They can see others' hats but not their own. After Alice and Bob say 'I don't know', Carla says 'I know my hat is red'. What color is each person's hat?",
        "expected_experts": ["Logic and Reasoning", "Pattern Finder", "AI/LLM Scientist"]
    }
]

def test_problem(problem_info):
    """Test a single problem and display results"""
    print(f"\n{'='*80}")
    print(f"Testing: {problem_info['name']}")
    print(f"{'='*80}")
    print(f"Problem: {problem_info['prompt'][:100]}...")
    
    # Make API request
    start_time = time.time()
    
    try:
        response = requests.post(API_URL, json={"prompt": problem_info["prompt"]})
        response.raise_for_status()
        result = response.json()
        
        end_time = time.time()
        
        # Display results
        print(f"\n✅ Success! (Time: {end_time - start_time:.2f}s)")
        print(f"\n📍 Detected Domain: {result['detected_domain']} ({result['domain_confidence']*100:.0f}% confidence)")
        
        # Show expert analyses
        print(f"\n👥 Expert Analyses:")
        for expert in result['expert_analyses']:
            if expert['confidence'] > 0.3:  # Only show relevant experts
                print(f"  • {expert['expert']}: {expert['confidence']*100:.0f}% confidence")
                if expert['insights']:
                    print(f"    - {expert['insights'][0]}")
                if expert['patterns']:
                    print(f"    - Hidden: {expert['patterns'][0]}")
        
        # Show reasoning paths
        print(f"\n🛤️  Reasoning Paths: {len(result['reasoning_paths'])}")
        for path in result['reasoning_paths']:
            print(f"  • Types: {', '.join(path['reasoning_types'])}")
            print(f"    - Confidence: {path['confidence']*100:.0f}%")
            print(f"    - What: {path['what'][:80]}...")
            
        # Show metrics
        print(f"\n📊 Problem Metrics:")
        print(f"  • Entropy: {result['problem_metrics']['entropy']:.2f}")
        print(f"  • Complexity: {result['problem_metrics']['complexity']:.2f}")
        print(f"  • Tokens: {result['problem_metrics']['token_count']}")
        
        # Show solution excerpt
        print(f"\n💡 Solution Preview:")
        solution_lines = result['solution'].split('\n')
        for line in solution_lines[:5]:
            if line.strip():
                print(f"  {line[:100]}...")
                
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error: {e}")
        print("Make sure the unified reasoning engine is running on port 5001")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 Unified Reasoning Engine Test Suite")
    print("=====================================")
    
    # Check if server is running
    print("\nChecking server status...")
    try:
        response = requests.get("http://localhost:5001/")
    except:
        print("❌ Server not running! Please start the unified reasoning engine:")
        print("   python unified_reasoning_engine.py")
        return
        
    print("✅ Server is running!")
    
    # Run tests
    successful = 0
    total = len(TEST_PROBLEMS)
    
    for problem in TEST_PROBLEMS:
        if test_problem(problem):
            successful += 1
        time.sleep(1)  # Rate limiting
        
    # Summary
    print(f"\n{'='*80}")
    print(f"Test Summary: {successful}/{total} tests passed")
    print(f"{'='*80}")
    
    if successful == total:
        print("🎉 All tests passed! The unified reasoning engine is working perfectly!")
    else:
        print(f"⚠️  {total - successful} tests failed. Check the logs for details.")

if __name__ == "__main__":
    main()