import os
import json
from src.base_agent import BaseAgent
from src.call_groq import chat_with_grok

class ReflectionAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.log_path = "logs/agent_logs.json"
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def format_prompt(self, last_thoughts: str, input_text: str) -> str:
        return (
            f"You are a reflective AI agent.\n"
            f"Earlier we talked about:\n{last_thoughts}\n\n"
            f"Current Input:\n{input_text}\n\n"
            f"If past talk is related to current input, reflect based on past thoughts. "
            f"Otherwise, give a response based on current input."
        )

    def think(self, input_text: str) -> str:
        self.add_memory("user", input_text)

        last_thoughts = "\n".join(
            [m["text"] for m in self.memory if m["role"] == "user"][-3:]
        )
        prompt = self.format_prompt(last_thoughts, input_text)

        messages = [{"role": "user", "content": prompt}]
        response = chat_with_grok(model="llama3-70b-8192", messages=messages)
        reply = response["choices"][0]["message"]["content"]

        self.add_memory("agent", reply)
        self._save_log(input_text, reply)

        return reply

    def _save_log(self, user_input: str, agent_response: str):
        log_entry = {
            "agent": self.name,
            "input": user_input,
            "response": agent_response,
        }

        data = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass

        data.append(log_entry)

        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
