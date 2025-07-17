from datetime import datetime
import os
import json

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.memory = []
        self.log_dir = "data"
        self.log_path = os.path.join(self.log_dir, "agent_logs.json")

        os.makedirs(self.log_dir, exist_ok=True)
        self._load_memory()

    def _save_memory(self):
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"❌ Failed to save memory: {e}")

    def _load_memory(self):
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    self.memory = json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load memory: {e}")

    def add_memory(self, role: str, text: str):
        self.memory.append({
            "role": role,
            "text": text,
            "timestamp": datetime.now().isoformat()
        })
        self._save_memory()

    def think(self, input_text: str) -> str:
        raise NotImplementedError("think method must be implemented by subclass")
