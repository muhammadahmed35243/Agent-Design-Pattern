# src/planning_agent.py
import os
import json
from src.base_agent import BaseAgent
from src.call_groq import chat_with_grok

class PlanningAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.plan_path = os.path.join(base_dir, "..", "data", "planning_steps.json")

    def format_prompt(self, input_text: str) -> str:
        return (
            f"You are a planning agent. Given the input, break it into 3 or more actionable steps.\n\n"
            f"Input:\n{input_text}\n\n"
            f"Respond with exactly 3 clear, numbered steps to complete the task."
        )

    def think(self, input_text: str) -> str:
        prompt = self.format_prompt(input_text)
        messages = [{"role": "user", "content": prompt}]
        response = chat_with_grok(
            model="llama3-70b-8192",
            messages=messages
        )

        plan = response["choices"][0]["message"]["content"]
        self.memory.append({"role": "user", "timestamp": self.get_timestamp(), "text": input_text})
        self.memory.append({"role": "agent", "timestamp": self.get_timestamp(), "text": plan})
        self._save_plan(input_text, plan)
        self.save_logs()
        return plan

    def _save_plan(self, task: str, plan: str):
        os.makedirs(os.path.dirname(self.plan_path), exist_ok=True)

        data = []
        if os.path.exists(self.plan_path):
            try:
                with open(self.plan_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass

        data.append({"task": task, "plan": plan})

        with open(self.plan_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_logs(self):
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "agent_logs.json")

        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"❌ Failed to save logs: {e}")

    def get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()
