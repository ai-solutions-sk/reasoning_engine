import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

class ConversationHistoryManager:
    """Manages conversation history for distinct sessions, storing each in a separate file."""

    def __init__(self, session_id: str, history_dir: str = "conversation_history", max_entries: int = 20):
        if not session_id or not isinstance(session_id, str):
            raise ValueError("A valid session_id string must be provided.")
            
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        self.history_path = self.history_dir / f"history_{session_id}.json"
        self.max_entries = max_entries
        self._history: List[Dict[str, Any]] = []
        self._load()

    def save_entry(self, prompt: str, what_problem: str, answer: str) -> None:
        """Append a new conversation turn and persist to disk for the current session."""
        self._history.append({
            "prompt": prompt,
            "what_problem": what_problem,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        self._history = self._history[-self.max_entries:]
        self._persist()

    def get_recent_history(self, max_entries: int = 5) -> List[Dict[str, Any]]:
        """Return the *max_entries* most recent conversation items for the current session."""
        return self._history[-max_entries:]

    def get_context_string(self, max_entries: int = 5) -> str:
        """Return a condensed text representation of the session's history."""
        items = self.get_recent_history(max_entries)
        if not items:
            return ""
        lines = ["PREVIOUS CONVERSATION TURNS:"]
        for idx, item in enumerate(items, 1):
            lines.append(f"{idx}. USER PROMPT: {item['prompt']}")
            lines.append(f"   YOUR ANSWER: {item['answer'][:200]}...")
        return "\n".join(lines)

    def _load(self):
        """Loads history from the session-specific file."""
        if self.history_path.exists():
            try:
                self._history = json.loads(self.history_path.read_text())
            except Exception:
                self._history = []
        else:
            self._history = []

    def _persist(self):
        """Saves history to the session-specific file."""
        try:
            self.history_path.write_text(json.dumps(self._history, ensure_ascii=False, indent=2))
        except Exception:
            pass 