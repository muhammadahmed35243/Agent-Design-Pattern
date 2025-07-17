from src.base_agent import BaseAgent
from tools.calculator import calculate
from tools.search import search
import re

TOOL_REGISTRY = {
    "calculator": calculate,
    "search": search
}

class ToolUserAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)

    def think(self, input_text: str) -> str:
        self.add_memory("user", input_text)

        tool_call = self.extract_tool_call(input_text)

        if tool_call:
            tool_name, tool_input = tool_call
        else:
            # fallback to using search tool for any unknown input
            tool_name = "search"
            tool_input = input_text

        tool_func = TOOL_REGISTRY.get(tool_name)

        if tool_func:
            try:
                result = tool_func(tool_input)
                response = f"[Tool:{tool_name}] Result: {result}"
            except Exception as e:
                response = f"[Error] Tool '{tool_name}' failed: {str(e)}"
        else:
            response = f"[Error] Tool '{tool_name}' not recognized."

        self.add_memory("agent", response)
        return response

    def extract_tool_call(self, text: str):
        pattern = r"\[TOOL:(\w+)\]\s*(.+)"
        match = re.search(pattern, text.strip())
        if match:
            tool_name = match.group(1)
            tool_input = match.group(2)
            return tool_name.lower(), tool_input
        return None
