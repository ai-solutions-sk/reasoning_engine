import time
import random
import os
import re
import google.generativeai as genai
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS for cross-origin requests
import json # Import json for parsing LLM output

# --- API Key Configuration ---
# IMPORTANT: Replace "YOUR_GEMINI_API_KEY_HERE" with your actual Gemini API key.
# For production, it's highly recommended to load this from an environment variable.
API_KEY = "AIzaSyBLXXuiqpx9BfDxGi28Ci8szlsb3qAm9Dw" 

# Configure the Gemini API globally
genai.configure(api_key=API_KEY)

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing frontend to connect

# --- Global instance of the simulator for persistent state ---
current_observer_state = {
    "perceptionThreshold": 0.2,
    "learningRate": 0.1,
    "noiseTolerance": 0.0
}

# --- Helper for Embedding Generation ---
class EmbeddingHelper:
    """Helper class to generate embeddings using text-embedding-004 model."""
    def __init__(self):
        self.model = genai.GenerativeModel('text-embedding-004')

    def get_embedding(self, text: str) -> list[float]:
        """Generates an embedding for the given text."""
        try:
            result = self.model.embed_content(model='text-embedding-004', content=text)
            return result['embedding']
        except Exception as e:
            print(f"Embedding API Error: {e}")
            return [] # Return empty list on error

embedding_helper = EmbeddingHelper() # Instantiate globally to avoid re-initializing model

# --- 1. SignalGenerator Class (The "Process") ---
class SignalGenerator:
    def __init__(self, task_types=None, num_tasks_per_type=3):
        self.task_types = task_types if task_types is not None else ["math_problem", "logical_puzzle", "summarization", "factual_question", "optimization_problem", "regulatory_compliance"]
        self.num_tasks_per_type = num_tasks_per_type
        self.tasks = self._generate_tasks()
        self._compute_task_embeddings()

    def _generate_tasks(self):
        all_tasks = []
        predefined_tasks_data = [
            {
                "prompt": "What is 15 * 7?",
                "type": "math_problem",
                "complexity": 0.1, # Low complexity
                "expectedIntuitive": "105",
                "expectedReasoning": "To multiply 15 by 7, you can decompose it: (10 * 7) + (5 * 7). This simplifies to 70 + 35, which results in 105. So the answer is 105."
            },
            {
                "prompt": "Summarize the key idea of photosynthesis in one sentence.",
                "type": "summarization",
                "complexity": 0.3, # Medium complexity
                "expectedIntuitive": "Plants convert light energy into chemical energy.",
                "expectedReasoning": "Photosynthesis is a complex process. In simple terms, plants, algae, and cyanobacteria convert light energy from the sun into chemical energy, stored in sugars, using carbon dioxide and water, and releasing oxygen. This is fundamental for most life on Earth."
            },
            {
                "prompt": "If all dogs are mammals, and Fido is a dog, is Fido a mammal?",
                "type": "logical_puzzle",
                "complexity": 0.2, # Medium complexity
                "expectedIntuitive": "Yes.",
                "expectedReasoning": "Given the premises: 1) All dogs are mammals. 2) Fido is a dog. By applying deductive reasoning (specifically, modus ponens), if Fido belongs to the category of dogs, and all dogs are mammals, it logically follows that Fido must also be a mammal. Therefore, the conclusion is yes."
            },
            {
                "prompt": "What is 100 / 2?",
                "type": "math_problem",
                "complexity": 0.05, # Very low complexity
                "expectedIntuitive": "50",
                "expectedReasoning": "To divide 100 by 2, you split 100 into two equal parts. Each part is 50. This is a basic division operation. So the answer is 50."
            },
            {
                "prompt": "Explain the concept of quantum entanglement simply.",
                "type": "summarization",
                "complexity": 0.4, # High complexity
                "expectedIntuitive": "Two particles linked, share fate instantly.",
                "expectedReasoning": "Quantum entanglement is a phenomenon where two or more particles become linked such that they share the same fate, regardless of distance. Measuring one instantly determines the other's property. This 'spooky action at a distance' is a cornerstone of quantum computing and challenges classical physics."
            },
            {
                "prompt": "What is the capital of USA?",
                "type": "factual_question",
                "complexity": 0.05, # Very low complexity, should be intuitive
                "expectedIntuitive": "Washington D.C.",
                "expectedReasoning": "The capital city of the United States of America is Washington, D.C. This is a factual recall. It is a federal district, not part of any state, established by the Constitution as the seat of the U.S. federal government."
            },
            {
                "prompt": "Can a dog fly?",
                "type": "factual_question",
                "complexity": 0.05, # Very low complexity, should be intuitive
                "expectedIntuitive": "No.",
                "expectedReasoning": "Dogs are mammals and do not possess wings or the biological structures necessary for flight. Therefore, based on biological facts, a dog cannot fly."
            },
            {
                "prompt": "Three perfect logicians — Alice, Bob, and Carla — are each given a hat that is either red or blue. They can see the other two hats, but not their own. The game master tells them: 'At least one of your hats is red.' 'You will take turns, starting with Alice, then Bob, then Carla, and each of you must either say 'I know my hat's color' or 'I don't know.' The game begins. They respond as follows: Alice: 'I don't know.' Bob: 'I don't know.' Carla: 'I know my hat is red.' Question: What are the colors of each person's hat, and how did Carla reason it out?",
                "type": "logical_puzzle",
                "complexity": 0.5, # High complexity, should be reasoning
                "expectedIntuitive": "Alice: Red, Bob: Red, Carla: Red.",
                "expectedReasoning": """
This is a classic logic puzzle. Here's how Carla reasons it out step-by-step:

1.  **Initial State & Common Knowledge:** Everyone knows there's at least one red hat. The possible combinations (R=Red, B=Blue) are RRR, RRB, RBR, BRR, RBB, BRB, BBR.
2.  **Alice's Turn (Alice says 'I don't know'):**
    * If Alice saw two blue hats (BB) on Bob and Carla, she would immediately know her hat was red (because of the 'at least one red hat' rule).
    * Since she *doesn't* know, it means she did *not* see two blue hats. Therefore, the combination BBR (Bob Blue, Carla Blue, Alice Red) is eliminated. Everyone (Bob and Carla) hears Alice say 'I don't know' and understands this implication.
3.  **Bob's Turn (Bob says 'I don't know'):**
    * Bob knows Alice didn't see BB. Now Bob considers his own situation.
    * If Bob saw a blue hat on Carla (B), and he himself had a blue hat, then Alice would have seen two blue hats (BB), which she ruled out. So, if Carla's hat was blue, Bob would immediately know his own hat was red.
    * Since Bob says 'I don't know', it means he did *not* see a blue hat on Carla. If he had, he would have known his own hat was red.
    * Therefore, Carla's hat must be Red. Everyone (Carla) hears Bob say 'I don't know' and understands this implication.
4.  **Carla's Turn (Carla says 'I know my hat is red'):**
    * Carla heard Alice say 'I don't know' (meaning not BBR).
    * Carla heard Bob say 'I don't know'. Carla deduces from Bob's statement that if her hat was Blue, Bob would have known his was Red. Since Bob *didn't* know, Carla's hat *must* be Red.
    * This unique deduction allows Carla to definitively state her hat color.

**Conclusion:** All three logicians must be wearing **Red** hats (R, R, R).
"""
            },
            # --- New Optimization Problem Task ---
            {
                "prompt": "You are a factory manager producing two products, X and Y. Each unit of X requires 2 hours on Machine A and 1 hour on Machine B. Each unit of Y requires 1 hour on Machine A and 3 hours on Machine B. You have a maximum of 10 hours available on Machine A and 15 hours on Machine B. Each unit of X sells for $3 profit, and each unit of Y sells for $2 profit. How many units of X and Y should you produce to maximize total profit? Also, what is the maximum profit? Assume you can only produce whole units.",
                "type": "optimization_problem",
                "complexity": 0.7, # High complexity, definitely requires reasoning/tool use
                "expectedIntuitive": "Produce 3 units of X and 4 units of Y for a maximum profit of $17.",
                "expectedReasoning": """
To solve this, we can formulate it as an Integer Linear Programming (ILP) problem:

**Variables:**
x = number of units of Product X
y = number of units of Product Y

**Objective Function (Maximize Profit):**
Maximize P = 3x + 2y

**Constraints:**
1. Machine A: 2x + 1y <= 10
2. Machine B: 1x + 3y <= 15
3. Non-negativity: x >= 0, y >= 0
4. Integer constraint: x, y must be integers (since we produce whole units)

**Solving Steps (Conceptual for LLM, Actual for Tool):**
1.  **Identify Variables, Objective, and Constraints:** The problem variables are X and Y units. The objective is to maximize profit. The constraints are machine hours.
2.  **Formulate as ILP:** Translate to mathematical inequalities and objective.
3.  **Choose a Solver:** This is an optimization problem requiring a solver (e.g., PuLP, SciPy, OR-Tools).
4.  **Generate Code:** Write Python code using the chosen library to set up and solve the ILP.
    ```python
    from pulp import *

    # Define the problem
    prob = LpProblem("Factory_Production", LpMaximize)

    # Define variables
    x = LpVariable("x", 0, None, LpInteger) # x >= 0, integer
    y = LpVariable("y", 0, None, LpInteger) # y >= 0, integer

    # Objective function
    prob += 3*x + 2*y, "Total Profit"

    # Constraints
    prob += 2*x + y <= 10, "Machine_A_Constraint"
    prob += x + 3*y <= 15, "Machine_B_Constraint"

    # Solve the problem
    prob.solve()

    # Get results
    production_x = value(x)
    production_y = value(y)
    max_profit = value(prob.objective)

    print(f"Produce X: {production_x}")
    print(f"Produce Y: {production_y}")
    print(f"Max Profit: {max_profit}")
    ```
5.  **Execute Code and Interpret Output:** The LLM would interpret the output of this code.
6.  **Verify Feasibility:** Check if the calculated (x, y) satisfy the constraints (e.g., 2*3 + 1*4 = 10 <= 10; 1*3 + 3*4 = 15 <= 15. Both feasible).
7.  **State Conclusion:** Present the optimal production plan and profit.

**Optimal Solution:**
Produce X = 3 units
Produce Y = 4 units
Maximum Profit = $17
"""
            },
            {
                "prompt": "FinnGrant is an AI system for automating loan approvals for small business applicants. Evaluate its high-risk classification under the EU AI Act, specifically regarding creditworthiness assessment.",
                "type": "regulatory_compliance",
                "complexity": 0.8,
                "expectedIntuitive": "FinnGrant is generally NOT high-risk under EU AI Act Annex III for small businesses.",
                "expectedReasoning": """
To evaluate FinnGrant's classification under the EU AI Act, we must apply a precise understanding of Annex III, Section 5(b) regarding creditworthiness assessment.

**1. Identify the AI System:** FinnGrant automates loan approvals for **small business applicants**. This is the crucial context.

**2. Review EU AI Act Annex III, Section 5(b):**
    * Annex III, Section 5(b) states that AI systems intended to be used for "evaluating the creditworthiness of natural persons or establishing their credit score, with the exception of AI systems used for the purpose of detecting financial fraud" are classified as high-risk.
    * **Crucial Nuance:** The key phrase here is "natural persons." The EU AI Act generally distinguishes between natural persons (individuals) and legal persons (businesses, including most SMEs).

**3. Apply Contextual Factor: Entity Type:**
    * FinnGrant is for "small business applicants." While some small businesses might be sole traders (natural persons), the general context of "small business" often refers to legal entities.
    * If FinnGrant is primarily assessing the creditworthiness of legal entities (e.g., limited companies, partnerships) as opposed to the *natural persons* behind them, then Annex III, Section 5(b) *does not directly apply* to classify it as high-risk *solely* on the basis of credit scoring.

**4. Conclusion on Risk Classification (Specific to 5b):**
    * Based on the strict interpretation of Annex III, Section 5(b), FinnGrant, when used for assessing the creditworthiness of *small businesses as legal entities*, is generally **NOT** classified as high-risk under this specific clause.
    * **Important Caveat:** It could still be classified as high-risk if it falls under *other* high-risk categories in Annex III (e.g., if it's used for access to essential private services that are not credit, or if it's a safety component of a product). However, based *solely* on the creditworthiness aspect for small businesses (legal persons), it's not high-risk.

**5. Compliance Implications:**
    * If FinnGrant is indeed *not* high-risk under 5(b) for small businesses (legal entities), then the stringent requirements of Articles 9-14 (risk management, data governance, human oversight, conformity assessment) would not apply *unless* another high-risk classification is met.
    * Compliance would still be necessary for general AI Act provisions (e.g., transparency for general-purpose AI, prohibited practices) and other relevant regulations like GDPR.

**Final Conclusion:** FinnGrant, as an AI system for automating loan approvals for **small business applicants (understood as legal entities)**, is generally **NOT** considered a high-risk AI system under Annex III, Section 5(b) of the EU AI Act. Its compliance requirements would depend on whether it meets *any other* high-risk criteria or if it processes personal data of natural persons within the business context that would trigger other high-risk classifications.
"""
            }
        ]
        return predefined_tasks_data

    def _compute_task_embeddings(self):
        """Computes and stores embeddings for all predefined tasks."""
        print("[Backend SignalGenerator] Computing embeddings for predefined tasks...")
        for task in self.tasks:
            task['embedding'] = embedding_helper.get_embedding(task['prompt']) # Corrected call to use global instance
            if not task['embedding']:
                print(f"Warning: Could not get embedding for task: {task['prompt'][:50]}... Using zero vector.")
                task['embedding'] = np.zeros(768).tolist() # Use a zero vector as fallback
        print("[Backend SignalGenerator] Embeddings computed.")

    def get_task_data_by_prompt(self, user_prompt: str):
        user_prompt_embedding = embedding_helper.get_embedding(user_prompt) # Corrected call to use global instance
        if not user_prompt_embedding:
            print("[Backend SignalGenerator] Could not get embedding for user prompt. Falling back to default.")
            return {
                "prompt": user_prompt,
                "type": "unknown",
                "complexity": 0.08,
                "expectedIntuitive": "I processed your request.",
                "expectedReasoning": "I processed your request with detailed consideration."
            }

        best_match = None
        highest_similarity = -1
        SIMILARITY_THRESHOLD_FOR_MATCH = 0.75 # Adjust this threshold as needed

        for task in self.tasks:
            if task.get('embedding'):
                sim = cosine_similarity(np.array(user_prompt_embedding).reshape(1, -1), 
                                        np.array(task['embedding']).reshape(1, -1))[0][0]
                
                if sim > highest_similarity:
                    highest_similarity = sim
                    best_match = task
        
        if best_match and highest_similarity >= SIMILARITY_THRESHOLD_FOR_MATCH:
            print(f"[Backend SignalGenerator] Matched (Embedding) to: {best_match['prompt'][:50]}... (Similarity: {highest_similarity:.2f}), Type: {best_match['type']}, Complexity: {best_match['complexity']}")
            return best_match
        else:
            default_task = {
                "prompt": user_prompt,
                "type": "unknown",
                "complexity": 0.08, # Default low complexity, more likely to be intuitive
                "expectedIntuitive": "I processed your request.",
                "expectedReasoning": "I processed your request with detailed consideration."
            }
            print(f"[Backend SignalGenerator] No strong embedding match (Highest Sim: {highest_similarity:.2f}). Using default task. Type: {default_task['type']}, Complexity: {default_task['complexity']}")
            return default_task

# --- 2. Observer Class (The Perception Logic) ---
class Observer:
    def __init__(self, perception_threshold: float = 0.2, learning_rate: float = 0.1, noise_tolerance: float = 0.0):
        self.perception_threshold = perception_threshold 
        self.learning_rate = learning_rate
        self.noise_tolerance = noise_tolerance 

    def determine_mode(self, task_complexity: float) -> str:
        if task_complexity < (self.perception_threshold + self.noise_tolerance):
            return "intuitive"
        else:
            return "reasoning"

    def adapt_threshold(self, task_complexity: float, performance_metric: float, mode_used: str):
        pass # No adaptation when mode is user-controlled

# --- New: DomainReasoningBuilder Class ---
class DomainReasoningBuilder:
    """
    Dynamically builds a domain-specific reasoning framework using a meta-LLM call.
    Caches frameworks to avoid redundant LLM calls.
    """
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.cache = {} # Cache for storing generated frameworks

    def _call_gemini_for_framework(self, prompt: str) -> str:
        """Helper to call Gemini for framework generation."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=0.4, # Lower temperature for more structured output
                    max_output_tokens=2048 # Sufficient tokens for the framework
                )
            )
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            return ""
        except Exception as e:
            print(f"Meta-LLM Framework Generation Error: {e}")
            return ""

    def get_reasoning_framework(self, domain: str, sub_domain: str = "") -> dict:
        """
        Generates or retrieves a domain-specific reasoning framework.
        """
        cache_key = f"{domain.lower()}_{sub_domain.lower()}"
        if cache_key in self.cache:
            print(f"[DomainReasoningBuilder] Returning cached framework for {domain}/{sub_domain}")
            return self.cache[cache_key]

        print(f"[DomainReasoningBuilder] Generating new framework for {domain}/{sub_domain}...")
        
        # --- ENHANCED META-PROMPT ---
        meta_prompt = (
            f"You are an AI Cognitive Architect and expert in prompt engineering. "
            f"Your task is to define a detailed reasoning framework for an LLM agent operating in the "
            f"'{domain}' domain" + (f", specifically within the '{sub_domain}' sub-domain." if sub_domain else ".") +
            f"\n\nThis framework should guide the LLM to perform highly nuanced and context-sensitive reasoning. "
            f"Explicitly define conditional rules or key distinctions that the LLM must apply based on specific contextual factors within this domain. "
            f"For example, if discussing regulations, specify how to differentiate applicability based on entity type (e.g., natural persons vs. businesses), scale, or specific conditions. "
            f"The 'mathematical_logic_framework' should briefly describe any relevant conditional logic, rule-based inference, multi-criteria decision analysis, or other formal methods needed to handle these nuances."
            f"\n\nProvide the following in strict JSON format. Ensure all keys are present, even if their values are null or empty lists. "
            f"The 'weights_override' should sum to 1.0 if provided, otherwise use null. "
            f"The 'reasoning_keywords' should be a list of 10-15 relevant terms, including terms related to contextual analysis, nuance, and conditional application."
            f"The 'conditional_reasoning_rules' should be a list of objects, each with 'condition' (string) and 'implication' (string) fields."
            f"\n\nJSON Schema Example:\n"
            f"{{\n"
            f"    \"persona\": \"<string>\",\n"
            f"    \"reasoning_guidelines\": \"<string>\",\n"
            f"    \"reasoning_keywords\": [\"<string>\", ...],\n"
            f"    \"weights_override\": {{\"accuracy\": <float>, \"semantic_similarity\": <float>, \"reasoning_process_quality\": <float>}} | null,\n"
            f"    \"mathematical_logic_framework\": \"<string>\",\n"
            f"    \"conditional_reasoning_rules\": [{{\"condition\": \"<string>\", \"implication\": \"<string>\"}}, ...]\n"
            f"}}\n\n"
            f"Generate the JSON framework now:"
        )

        llm_response_text = self._call_gemini_for_framework(meta_prompt)
        
        try:
            # Clean the response to ensure it's valid JSON
            # Sometimes LLMs add markdown code blocks, remove them
            if llm_response_text.startswith("```json"):
                llm_response_text = llm_response_text[len("```json"):].strip()
            if llm_response_text.endswith("```"):
                llm_response_text = llm_response_text[:-len("```")].strip()

            framework = json.loads(llm_response_text)
            # Ensure default values if LLM misses something
            framework.setdefault("persona", f"expert in {domain}")
            framework.setdefault("reasoning_guidelines", "Follow a logical, step-by-step process.")
            framework.setdefault("reasoning_keywords", [])
            framework.setdefault("weights_override", None)
            framework.setdefault("mathematical_logic_framework", "General Logic")
            framework.setdefault("conditional_reasoning_rules", []) # Ensure this is always present

            self.cache[cache_key] = framework
            return framework
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from meta-LLM: {e}")
            print(f"Raw LLM response: {llm_response_text}")
            # Fallback to a default framework if parsing fails
            return {
                "persona": f"general {domain} expert",
                "reasoning_guidelines": "Adopt a structured reasoning approach. Break down the problem, analyze components, and derive conclusions logically, considering context and nuances.",
                "reasoning_keywords": ["analyze", "structure", "component", "derive", "conclusion", "logic", "breakdown", "approach", "method", "principle", "context", "nuance", "exception", "applicability", "distinction"],
                "weights_override": None, # Use default weights
                "mathematical_logic_framework": "Conditional Logic and Contextual Analysis",
                "conditional_reasoning_rules": [] # Default to empty list of rules
            }

# --- 3. LLMAgent Class (Gemini 1.5 Pro Wrapper) ---
class LLMAgent:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    def _call_gemini(self, prompt: str, temperature: float = 0.7, max_output_tokens: int = 8192) -> tuple[str, int, int]:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=temperature, 
                    max_output_tokens=max_output_tokens 
                )
            )
            response_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            if response.candidates and response.candidates[0].content.parts:
                response_text = response.candidates[0].content.parts[0].text
            
            if response.usage_metadata:
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count
                
            return response_text, prompt_tokens, completion_tokens
        except Exception as e:
            print(f"LLM API Error: {e}")
            return f"Error: {e}", 0, 0

    def get_cognition(self, task_prompt: str, mode: str, reasoning_framework: dict) -> tuple[str, float, int, int]:
        start_time = time.time()
        prompt_template = ""
        
        if mode == "reasoning":
            persona = reasoning_framework.get("persona", "expert problem solver")
            guidelines = reasoning_framework.get("reasoning_guidelines", "Take step-by-step reasoning to solve the following problem. Explain your thought process clearly and then state the final answer.")
            conditional_rules = reasoning_framework.get("conditional_reasoning_rules", [])

            # Format conditional rules for injection into the prompt
            rules_str = ""
            if conditional_rules:
                rules_str = "\n\nApply the following specific conditional reasoning rules:\n"
                for i, rule in enumerate(conditional_rules):
                    rules_str += f"{i+1}. IF: {rule.get('condition', 'N/A')}\n   THEN: {rule.get('implication', 'N/A')}\n"
                rules_str += "\nEnsure your reasoning explicitly checks for and applies these conditions where relevant.\n"

            # Special handling for optimization problems within the dynamic framework
            if "optimization_problem" in task_prompt.lower() or "maximize profit" in task_prompt.lower() or "linear programming" in task_prompt.lower():
                prompt_template = (
                    f"You are an expert in mathematical optimization. "
                    f"First, outline your strategy for solving this problem, including how you'll decompose it. "
                    f"Then, formulate the problem as a Linear Programming (LP) or Integer Linear Programming (ILP) problem. "
                    f"Next, generate Python code using a standard library like 'pulp' or 'scipy.optimize' to solve it. "
                    f"Finally, interpret the results, verify constraints, and state the optimal solution clearly. "
                    f"Consider any potential pitfalls or alternative approaches you might have evaluated. Problem: {task_prompt}"
                )
            else:
                # General reasoning prompt incorporating dynamic persona, guidelines, AND conditional rules
                prompt_template = (
                    f"You are a {persona}. "
                    f"{guidelines} "
                    f"{rules_str}" # Inject the conditional rules here
                    f"Problem: {task_prompt}"
                )
        elif mode == "intuitive":
            prompt_template = (
                f"You are an intuitive genius. Give your best intuitive insight or the final concise answer "
                f"to the following problem. Respond with only the most essential information, no steps. Problem: {task_prompt}"
            )
        else:
            return "Invalid mode", 0.0, 0, 0
        
        response_text, prompt_tokens, completion_tokens = self._call_gemini(prompt_template)
        end_time = time.time()
        latency = end_time - start_time
        
        return response_text, latency, prompt_tokens, completion_tokens

# --- 4. CognitiveScoringEngine Class ---
class CognitiveScoringEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        # Default weights - these can be overridden by dynamic framework
        self.default_weights = {
            "math_problem": {"accuracy": 0.5, "semantic_similarity": 0.3, "reasoning_process_quality": 0.2},
            "logical_puzzle": {"accuracy": 0.4, "semantic_similarity": 0.4, "reasoning_process_quality": 0.2},
            "summarization": {"accuracy": 0.3, "semantic_similarity": 0.5, "reasoning_process_quality": 0.2},
            "factual_question": {"accuracy": 0.6, "semantic_similarity": 0.3, "reasoning_process_quality": 0.1},
            "optimization_problem": {"accuracy": 0.5, "semantic_similarity": 0.3, "reasoning_process_quality": 0.2},
            "regulatory_compliance": {"accuracy": 0.7, "semantic_similarity": 0.2, "reasoning_process_quality": 0.1}, # Added for the new task type
            "unknown": {"accuracy": 0.4, "semantic_similarity": 0.4, "reasoning_process_quality": 0.2}
        }
        # Default reasoning keywords - can be overridden by dynamic framework
        self.default_reasoning_keywords = [
            "step-by-step", "steps", "first", "next", "then", "finally", "therefore", "conclude", 
            "premises", "conclusion", "deduce", "infer", "logical", "formulate", "variables", 
            "objective", "constraints", "solution", "algorithm", "plan", "strategy", 
            "decompose", "break down", "consider", "reflect", "verify", "eliminate possibilities", 
            "scenario", "modus ponens", "modus tollens", "deduction", "induction",
            "contextual", "nuance", "exception", "applicability", "distinction", "conditional", "rule" # Added nuance terms
        ]

    def calculate_similarity(self, text1: str, text2: str) -> float:
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        try:
            vectors = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity
        except ValueError as e:
            if "empty vocabulary" in str(e):
                return 1.0 if text1 == text2 else 0.0
            raise e

    def calculate_reasoning_process_quality(self, llm_response: str, reasoning_keywords: list[str]) -> float:
        """
        Calculates a score based on the presence of reasoning-indicative keywords and structural elements.
        Uses dynamically provided reasoning_keywords.
        """
        response_lower = llm_response.lower()
        
        # Keyword presence
        keyword_count = sum(1 for keyword in reasoning_keywords if keyword in response_lower)
        max_keywords_possible = len(reasoning_keywords)
        keyword_score = keyword_count / max_keywords_possible if max_keywords_possible > 0 else 0.0

        # Structural elements (e.g., numbered lists, bullet points)
        has_numbered_list = bool(re.search(r'\d+\.\s', llm_response))
        has_bullet_points = bool(re.search(r'[\*-]\s', llm_response))
        
        structure_score = 0.0
        if has_numbered_list or has_bullet_points:
            structure_score = 0.5 # Give a base score if structured elements are present

        process_quality = (keyword_score * 0.7) + (structure_score * 0.3)
        return min(process_quality, 1.0)

    def score_response(self, llm_response: str, expected_reasoning: str, expected_intuitive: str, 
                       prompted_mode: str, task_type: str, reasoning_framework: dict) -> tuple[float, float, float, float]:
        
        accuracy = 0.0
        semantic_similarity = 0.0
        reasoning_process_quality = 0.0

        actual_output_style = prompted_mode 

        # Determine weights and keywords based on dynamic framework or defaults
        weights_for_type = reasoning_framework.get("weights_override")
        if weights_for_type is None:
            weights_for_type = self.default_weights.get(task_type, self.default_weights["unknown"])
        
        current_reasoning_keywords = reasoning_framework.get("reasoning_keywords")
        if not current_reasoning_keywords: # Fallback if dynamic keywords are empty
            current_reasoning_keywords = self.default_reasoning_keywords


        if task_type == "math_problem":
            llm_numbers = re.findall(r'\d+', llm_response)
            expected_intuitive_num = str(expected_intuitive).strip()
            if expected_intuitive_num in llm_numbers:
                accuracy = 1.0
            else:
                cleaned_llm_response = re.sub(r'[^\d\s]', '', llm_response).lower().strip()
                cleaned_expected_intuitive = re.sub(r'[^\d\s]', '', expected_intuitive).lower().strip()
                if cleaned_expected_intuitive in cleaned_llm_response:
                    accuracy = 1.0

        elif task_type in ["logical_puzzle", "summarization", "factual_question", "unknown", "regulatory_compliance"]: # Added regulatory_compliance
            if actual_output_style == "intuitive":
                sim_to_intuitive = self.calculate_similarity(llm_response, expected_intuitive)
                if sim_to_intuitive > 0.7:
                    accuracy = 1.0
            elif actual_output_style == "reasoning":
                sim_to_reasoning = self.calculate_similarity(llm_response, expected_reasoning)
                if sim_to_reasoning > 0.7:
                    accuracy = 1.0
        
        elif task_type == "optimization_problem":
            optimal_intuitive_parts = re.findall(r'\d+', expected_intuitive)
            llm_response_numbers = re.findall(r'\d+', llm_response)

            found_all_optimal_parts = True
            for part in optimal_intuitive_parts:
                if part not in llm_response_numbers:
                    found_all_optimal_parts = False
                    break
            if found_all_optimal_parts:
                accuracy = 1.0

            semantic_similarity = self.calculate_similarity(llm_response, expected_reasoning)
            if semantic_similarity > 0.7:
                accuracy = max(accuracy, 1.0)

        # Calculate semantic similarity and reasoning process quality based on the prompted mode
        if actual_output_style == "reasoning":
            semantic_similarity = self.calculate_similarity(llm_response, expected_reasoning)
            reasoning_process_quality = self.calculate_reasoning_process_quality(llm_response, current_reasoning_keywords)
        elif actual_output_style == "intuitive":
            semantic_similarity = self.calculate_similarity(llm_response, expected_intuitive)
            reasoning_process_quality = 0.0 # Not applicable for intuitive mode
        
        combined_score = (weights_for_type["accuracy"] * accuracy) + \
                         (weights_for_type["semantic_similarity"] * semantic_similarity) + \
                         (weights_for_type["reasoning_process_quality"] * reasoning_process_quality)
        
        return accuracy, semantic_similarity, reasoning_process_quality, combined_score

# --- Flask API Endpoint ---
@app.route('/process_task', methods=['POST'])
def process_task():
    data = request.get_json()
    user_prompt = data.get('prompt')
    selected_mode = data.get('selectedMode') 
    domain = data.get('domain', 'general') # Default to 'general'
    sub_domain = data.get('subDomain', '') # Default to empty string

    observer_instance = Observer() 
    signal_generator = SignalGenerator()
    llm_agent = LLMAgent()
    scoring_engine = CognitiveScoringEngine()
    domain_builder = DomainReasoningBuilder() # Instantiate the new builder

    task_data = signal_generator.get_task_data_by_prompt(user_prompt)
    task_complexity = task_data['complexity']
    
    cognitive_mode = selected_mode 

    reasoning_framework = {}
    if cognitive_mode == "reasoning":
        reasoning_framework = domain_builder.get_reasoning_framework(domain, sub_domain)
        # If the LLM generates empty keywords, ensure a fallback for scoring
        if not reasoning_framework.get("reasoning_keywords"):
            reasoning_framework["reasoning_keywords"] = scoring_engine.default_reasoning_keywords
        # If weights_override is None from LLM, ensure a fallback
        if reasoning_framework.get("weights_override") is None:
             reasoning_framework["weights_override"] = scoring_engine.default_weights.get(task_data['type'], scoring_engine.default_weights["unknown"])
    else:
        # For intuitive mode, provide a minimal default framework
        reasoning_framework = {
            "persona": "intuitive genius",
            "reasoning_guidelines": "Respond concisely.",
            "reasoning_keywords": [], # No keywords for intuitive
            "weights_override": scoring_engine.default_weights.get(task_data['type'], scoring_engine.default_weights["unknown"]),
            "mathematical_logic_framework": "Intuition",
            "conditional_reasoning_rules": [] # No rules for intuitive
        }


    llm_response, latency, prompt_tokens, completion_tokens = llm_agent.get_cognition(
        task_data['prompt'], 
        cognitive_mode,
        reasoning_framework # Pass the dynamic framework to LLMAgent
    )
    
    accuracy, semantic_similarity, reasoning_process_quality, combined_score = scoring_engine.score_response(
        llm_response,
        task_data['expectedReasoning'],
        task_data['expectedIntuitive'],
        cognitive_mode, 
        task_data['type'],
        reasoning_framework # Pass the dynamic framework to CognitiveScoringEngine
    )

    return jsonify({
        "cognitiveMode": cognitive_mode,
        "llmResponse": llm_response,
        "latencySeconds": latency,
        "promptTokens": prompt_tokens,
        "completionTokens": completion_tokens,
        "totalTokens": prompt_tokens + completion_tokens,
        "accuracy": accuracy,
        "semanticSimilarity": semantic_similarity,
        "reasoningProcessQuality": reasoning_process_quality, 
        "combinedScore": combined_score, 
        "taskType": task_data['type'],
        "taskComplexity": task_complexity,
        "reasoningFramework": reasoning_framework # Return the generated framework for display
    })

# To run this Flask app:
# 1. Save it as e.g., `app.py`
# 2. Install Flask and Flask-CORS: `pip install Flask Flask-CORS`
# 3. Run from your terminal: `flask run`
#    (By default, it runs on http://127.0.0.1:5000/)
