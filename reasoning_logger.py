import json
import time
import uuid
import hashlib
import psutil
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import numpy as np
import networkx as nx
from pathlib import Path

class UnifiedReasoningLogger:
    """
    Comprehensive logging system for the unified reasoning engine
    Based on AI system design principles and mathematical foundations
    """
    
    def __init__(self, log_directory: str = "reasoning_logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        self.current_session = None
        self.start_time = None
        self.performance_tracker = {}
        
    def start_session(self, prompt: str, engine_version: str = "1.0.0", 
                     model_config: Dict[str, Any] = None) -> str:
        """Start a new reasoning session with comprehensive metadata"""
        
        session_id = str(uuid.uuid4())
        self.start_time = time.time()
        
        # Get system information
        process = psutil.Process()
        memory_info = process.memory_info()
        
        self.current_session = {
            "session_metadata": {
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "engine_version": engine_version,
                "model_configuration": model_config or {
                    "primary_model": "gemini-1.5-pro",
                    "temperature": 0.2,
                    "max_tokens": 8192,
                    "top_p": 0.9
                },
                "environment": {
                    "python_version": f"{psutil.version_info[0]}.{psutil.version_info[1]}.{psutil.version_info[2]}",
                    "os": psutil.os.name,
                    "memory_available_gb": psutil.virtual_memory().available / (1024**3)
                }
            },
            "input_analysis": {
                "raw_prompt": prompt,
                "preprocessed_prompt": self._preprocess_prompt(prompt),
                "structural_analysis": {},
                "domain_detection": {},
                "complexity_metrics": {}
            },
            "reasoning_process": {
                "expert_analyses": [],
                "reasoning_paths": [],
                "cross_validation": {
                    "consistency_metrics": {},
                    "contradiction_analysis": []
                }
            },
            "mathematical_foundations": {
                "linear_algebra_components": {},
                "calculus_components": {},
                "statistical_components": {},
                "information_theory": {}
            },
            "output_synthesis": {
                "final_answer": "",
                "synthesis_process": {},
                "confidence_assessment": {},
                "hidden_insights": [],
                "alternative_solutions": []
            },
            "performance_metrics": {
                "latency_metrics": {},
                "computational_metrics": {},
                "quality_metrics": {}
            },
            "validation_metrics": {
                "mathematical_validation": {},
                "logical_validation": {},
                "empirical_validation": {},
                "meta_validation": {}
            },
            "traceability": {
                "decision_tree": {},
                "lineage": []
            },
            "metadata": {
                "version_info": {
                    "schema_version": "1.0.0",
                    "engine_version": engine_version,
                    "dependencies": self._get_dependencies()
                },
                "debug_info": {
                    "warnings": [],
                    "errors": [],
                    "debug_traces": []
                },
                "reproducibility": {
                    "random_seed": np.random.get_state()[1][0],
                    "deterministic_flags": {},
                    "environment_hash": self._generate_environment_hash()
                }
            }
        }
        
        return session_id
    
    def log_input_analysis(self, structural_analysis: Dict[str, Any], 
                          domain_detection: Dict[str, Any],
                          complexity_metrics: Dict[str, Any]):
        """Log comprehensive input analysis"""
        if not self.current_session:
            raise ValueError("No active session. Call start_session first.")
            
        self.current_session["input_analysis"]["structural_analysis"] = structural_analysis
        self.current_session["input_analysis"]["domain_detection"] = domain_detection
        self.current_session["input_analysis"]["complexity_metrics"] = complexity_metrics
        
        # Calculate information theory metrics
        prompt = self.current_session["input_analysis"]["raw_prompt"]
        self.current_session["mathematical_foundations"]["information_theory"] = {
            "entropy_measures": {
                "shannon_entropy": self._calculate_shannon_entropy(prompt),
                "conditional_entropy": 0.0,  # Placeholder
                "mutual_information": 0.0,   # Placeholder
                "kl_divergence": 0.0        # Placeholder
            },
            "compression_analysis": self._analyze_compression(prompt)
        }
    
    def log_expert_analysis(self, expert_type: str, analysis_results: Dict[str, Any], 
                           confidence: float, what_analysis: str, 
                           how_analysis: str, why_analysis: str):
        """Log individual expert analysis with mathematical rigor"""
        if not self.current_session:
            raise ValueError("No active session. Call start_session first.")
            
        expert_log = {
            "expert_type": expert_type,
            "confidence_score": confidence,
            "analysis_results": analysis_results,
            "what_analysis": what_analysis,
            "how_analysis": how_analysis,
            "why_analysis": why_analysis
        }
        
        self.current_session["reasoning_process"]["expert_analyses"].append(expert_log)
        
        # Update mathematical foundations based on expert type
        self._update_mathematical_foundations(expert_type, analysis_results)
    
    def log_reasoning_path(self, path_data: Dict[str, Any]):
        """Log a complete reasoning path with step-by-step process"""
        if not self.current_session:
            raise ValueError("No active session. Call start_session first.")
            
        # Ensure step-by-step process is included
        if "step_by_step_process" not in path_data:
            path_data["step_by_step_process"] = []
            
        self.current_session["reasoning_process"]["reasoning_paths"].append(path_data)
        
        # Update traceability
        self._update_traceability(path_data)
    
    def log_cross_validation(self, consistency_metrics: Dict[str, Any], 
                           contradictions: List[Dict[str, Any]]):
        """Log cross-validation between reasoning approaches"""
        if not self.current_session:
            raise ValueError("No active session. Call start_session first.")
            
        self.current_session["reasoning_process"]["cross_validation"] = {
            "consistency_metrics": consistency_metrics,
            "contradiction_analysis": contradictions
        }
    
    def log_synthesis(self, final_answer: str, synthesis_process: Dict[str, Any],
                     confidence_assessment: Dict[str, Any],
                     hidden_insights: List[Dict[str, Any]] = None,
                     alternative_solutions: List[Dict[str, Any]] = None):
        """Log the final synthesis with complete reasoning chain"""
        if not self.current_session:
            raise ValueError("No active session. Call start_session first.")
            
        self.current_session["output_synthesis"] = {
            "final_answer": final_answer,
            "synthesis_process": synthesis_process,
            "confidence_assessment": confidence_assessment,
            "hidden_insights": hidden_insights or [],
            "alternative_solutions": alternative_solutions or []
        }
    
    def log_performance_metrics(self, additional_metrics: Dict[str, Any] = None):
        """Log comprehensive performance metrics"""
        if not self.current_session:
            raise ValueError("No active session. Call start_session first.")
            
        current_time = time.time()
        total_latency = current_time - self.start_time if self.start_time else 0
        
        # Get current system metrics
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        self.current_session["performance_metrics"] = {
            "latency_metrics": {
                "total_latency_seconds": total_latency,
                "input_analysis_time": self.performance_tracker.get("input_analysis", 0),
                "expert_analysis_time": self.performance_tracker.get("expert_analysis", 0),
                "reasoning_path_time": self.performance_tracker.get("reasoning_paths", 0),
                "synthesis_time": self.performance_tracker.get("synthesis", 0),
                "llm_call_latencies": self.performance_tracker.get("llm_calls", [])
            },
            "computational_metrics": {
                "memory_peak_mb": memory_info.rss / (1024 * 1024),
                "cpu_usage_percent": cpu_percent,
                "token_consumption": self.performance_tracker.get("tokens", {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }),
                "expert_utilization": self._calculate_expert_utilization()
            },
            "quality_metrics": {
                "reasoning_depth": self._calculate_reasoning_depth(),
                "mathematical_complexity": self._calculate_mathematical_complexity(),
                "cross_validation_score": self._calculate_cross_validation_score(),
                "novelty_score": self._calculate_novelty_score()
            }
        }
        
        if additional_metrics:
            self.current_session["performance_metrics"].update(additional_metrics)
    
    def log_validation_metrics(self, mathematical_validation: Dict[str, Any],
                              logical_validation: Dict[str, Any],
                              empirical_validation: Dict[str, Any],
                              meta_validation: Dict[str, Any]):
        """Log comprehensive validation metrics"""
        if not self.current_session:
            raise ValueError("No active session. Call start_session first.")
            
        self.current_session["validation_metrics"] = {
            "mathematical_validation": mathematical_validation,
            "logical_validation": logical_validation,
            "empirical_validation": empirical_validation,
            "meta_validation": meta_validation
        }
    
    def add_warning(self, warning: str):
        """Add a warning to the debug info"""
        if self.current_session:
            self.current_session["metadata"]["debug_info"]["warnings"].append(warning)
    
    def add_error(self, error: str):
        """Add an error to the debug info"""
        if self.current_session:
            self.current_session["metadata"]["debug_info"]["errors"].append(error)
            self.current_session["metadata"]["debug_info"]["debug_traces"].append(traceback.format_exc())
    
    def track_performance(self, operation: str, duration: float):
        """Track performance for specific operations"""
        if operation not in self.performance_tracker:
            self.performance_tracker[operation] = []
        self.performance_tracker[operation].append(duration)
    
    def save_session(self, filename: str = None) -> str:
        """Save the current session to a JSON file"""
        if not self.current_session:
            raise ValueError("No active session to save.")
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = self.current_session["session_metadata"]["session_id"][:8]
            filename = f"reasoning_session_{timestamp}_{session_id}.json"
        
        filepath = self.log_directory / filename
        
        # Make the session serializable
        serializable_session = self._make_serializable(self.current_session)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_session, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def load_session(self, filepath: str) -> Dict[str, Any]:
        """Load a session from a JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _preprocess_prompt(self, prompt: str) -> str:
        """Preprocess the prompt for analysis"""
        # Basic cleaning and normalization
        cleaned = prompt.strip()
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        return cleaned
    
    def _calculate_shannon_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        total_chars = len(text)
        entropy = 0.0
        for count in char_counts.values():
            p = count / total_chars
            entropy -= p * np.log2(p)
        
        return entropy
    
    def _analyze_compression(self, text: str) -> Dict[str, Any]:
        """Analyze compression properties of text"""
        import zlib
        
        if not text:
            return {"original_size": 0, "compressed_size": 0, "compression_ratio": 0}
        
        original_bytes = text.encode('utf-8')
        compressed_bytes = zlib.compress(original_bytes)
        
        return {
            "original_size": len(original_bytes),
            "compressed_size": len(compressed_bytes),
            "compression_ratio": len(compressed_bytes) / len(original_bytes)
        }
    
    def _update_mathematical_foundations(self, expert_type: str, analysis_results: Dict[str, Any]):
        """Update mathematical foundations based on expert analysis"""
        if expert_type == "linear_algebra":
            self.current_session["mathematical_foundations"]["linear_algebra_components"] = {
                "vector_spaces": analysis_results.get("vector_spaces", []),
                "matrices": analysis_results.get("matrices", []),
                "transformations": analysis_results.get("transformations", [])
            }
        elif expert_type == "calculus":
            self.current_session["mathematical_foundations"]["calculus_components"] = {
                "derivatives": analysis_results.get("derivatives", []),
                "integrals": analysis_results.get("integrals", []),
                "limits": analysis_results.get("limits", [])
            }
        elif expert_type == "statistics":
            self.current_session["mathematical_foundations"]["statistical_components"] = {
                "probability_distributions": analysis_results.get("distributions", []),
                "statistical_tests": analysis_results.get("tests", []),
                "confidence_intervals": analysis_results.get("confidence_intervals", [])
            }
    
    def _update_traceability(self, path_data: Dict[str, Any]):
        """Update traceability information"""
        lineage_entry = {
            "output_component": path_data.get("path_id", "unknown"),
            "source_inputs": [self.current_session["input_analysis"]["raw_prompt"]],
            "transformation_path": [step.get("operation", "") for step in path_data.get("step_by_step_process", [])],
            "confidence_propagation": [step.get("confidence", 0) for step in path_data.get("step_by_step_process", [])]
        }
        
        self.current_session["traceability"]["lineage"].append(lineage_entry)
    
    def _calculate_expert_utilization(self) -> Dict[str, bool]:
        """Calculate which experts were utilized"""
        expert_types = set()
        for analysis in self.current_session["reasoning_process"]["expert_analyses"]:
            expert_types.add(analysis["expert_type"])
        
        all_experts = ["linear_algebra", "calculus", "statistics", "linear_programming", 
                      "logic", "causal", "pattern", "ai_scientist"]
        
        return {expert: expert in expert_types for expert in all_experts}
    
    def _calculate_reasoning_depth(self) -> int:
        """Calculate the depth of reasoning"""
        max_depth = 0
        for path in self.current_session["reasoning_process"]["reasoning_paths"]:
            depth = len(path.get("step_by_step_process", []))
            max_depth = max(max_depth, depth)
        return max_depth
    
    def _calculate_mathematical_complexity(self) -> float:
        """Calculate mathematical complexity score"""
        complexity_score = 0.0
        
        # Count mathematical expressions
        math_expressions = self.current_session["input_analysis"]["structural_analysis"].get("math_expressions", [])
        complexity_score += len(math_expressions) * 0.2
        
        # Add complexity from expert analyses
        for analysis in self.current_session["reasoning_process"]["expert_analyses"]:
            if analysis["expert_type"] in ["linear_algebra", "calculus", "statistics"]:
                complexity_score += analysis["confidence_score"] * 0.3
        
        return min(complexity_score, 10.0)  # Cap at 10
    
    def _calculate_cross_validation_score(self) -> float:
        """Calculate cross-validation score"""
        consistency_metrics = self.current_session["reasoning_process"]["cross_validation"]["consistency_metrics"]
        return consistency_metrics.get("expert_agreement_score", 0.0)
    
    def _calculate_novelty_score(self) -> float:
        """Calculate novelty score based on hidden insights"""
        hidden_insights = self.current_session["output_synthesis"]["hidden_insights"]
        return min(len(hidden_insights) * 0.1, 1.0)  # Cap at 1.0
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get version information for key dependencies"""
        dependencies = {}
        try:
            import numpy
            dependencies["numpy"] = numpy.__version__
        except ImportError:
            pass
        
        try:
            import sympy
            dependencies["sympy"] = sympy.__version__
        except ImportError:
            pass
        
        try:
            import sklearn
            dependencies["scikit-learn"] = sklearn.__version__
        except ImportError:
            pass
        
        return dependencies
    
    def _generate_environment_hash(self) -> str:
        """Generate a hash of the environment for reproducibility"""
        env_string = f"{psutil.os.name}_{psutil.version_info}_{self._get_dependencies()}"
        return hashlib.md5(env_string.encode()).hexdigest()
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object serializable for JSON"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, nx.Graph) or isinstance(obj, nx.DiGraph):
            return {
                "nodes": list(obj.nodes()),
                "edges": [{"source": u, "target": v, "weight": d.get("weight", 1.0)} 
                         for u, v, d in obj.edges(data=True)]
            }
        else:
            return obj

# Example usage and integration helper
class ReasoningSessionManager:
    """Manager class to integrate logging with the unified reasoning engine"""
    
    def __init__(self, log_directory: str = "reasoning_logs"):
        self.logger = UnifiedReasoningLogger(log_directory)
        self.current_session_id = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_session_id:
            try:
                self.logger.log_performance_metrics()
                filepath = self.logger.save_session()
                print(f"Session saved to: {filepath}")
            except Exception as e:
                print(f"Error saving session: {e}")
    
    def start_reasoning_session(self, prompt: str, **kwargs) -> str:
        """Start a new reasoning session"""
        self.current_session_id = self.logger.start_session(prompt, **kwargs)
        return self.current_session_id
    
    def log_step(self, step_type: str, data: Dict[str, Any]):
        """Log a step in the reasoning process"""
        if step_type == "input_analysis":
            self.logger.log_input_analysis(**data)
        elif step_type == "expert_analysis":
            self.logger.log_expert_analysis(**data)
        elif step_type == "reasoning_path":
            self.logger.log_reasoning_path(data)
        elif step_type == "synthesis":
            self.logger.log_synthesis(**data)
        # Add more step types as needed
    
    def add_debug_info(self, info_type: str, message: str):
        """Add debug information"""
        if info_type == "warning":
            self.logger.add_warning(message)
        elif info_type == "error":
            self.logger.add_error(message) 