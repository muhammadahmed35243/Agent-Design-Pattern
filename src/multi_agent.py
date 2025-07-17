from src.reflection_agent import ReflectionAgent
from src.planning_agent import PlanningAgent
from src.tool_user_agent import ToolUserAgent
from datetime import datetime
import joblib
import os
from typing import Dict, List, Any


class CoordinatorAgent:
    """
    Coordinator agent that routes user inputs to appropriate specialized agents
    using ML-based classification with keyword fallback.
    """
    
    def __init__(self, model_path: str = './model/simple_agent_classifier.pkl'):
        """
        Initialize the coordinator with specialized agents and ML model.
        
        Args:
            model_path: Path to the trained classification model
        """
        # Initialize specialized agents
        self.reflector = ReflectionAgent("Reflector")
        self.planner = PlanningAgent("Planner")
        self.tooler = ToolUserAgent("Tooler")
        
        # Initialize memory for conversation history
        self.memory: List[Dict[str, Any]] = []
        
        # Load ML model for agent classification
        self.model = None
        self.model_path = model_path
        self._load_model()
        
        # Configuration
        self.confidence_threshold = 0.6
        
    def _load_model(self) -> None:
        """Load the ML classification model if available."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"✅ ML model loaded from {self.model_path}")
            else:
                print(f"⚠️ Model file not found: {self.model_path}")
                print("Using keyword-based fallback method only.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Using keyword-based fallback method only.")
            self.model = None
    
    def decide_agent(self, user_input: str) -> str:
        """
        Decide which agent to use based on user input using ML model
        with keyword fallback for low confidence or model unavailability.
        
        Args:
            user_input: The user's input text
            
        Returns:
            str: Agent type name (PlanningAgent, ReflectionAgent, or ToolUserAgent)
        """
        # Try ML prediction first if model is available
        if self.model is not None:
            try:
                prediction = self.model.predict([user_input])[0]
                probabilities = self.model.predict_proba([user_input])[0]
                confidence = probabilities.max()
                
                # Use ML prediction if confident enough
                if confidence >= self.confidence_threshold:
                    return prediction
                else:
                    print(f"⚠️ Low confidence ({confidence:.3f}), using keyword fallback")
                    
            except Exception as e:
                print(f"❌ ML prediction failed: {e}")
        
        # Fall back to keyword method
        return self._keyword_fallback(user_input)
    
    def _keyword_fallback(self, user_input: str) -> str:
        """
        Keyword-based agent classification as fallback method.
        
        Args:
            user_input: The user's input text
            
        Returns:
            str: Agent type name
        """
        input_lower = user_input.lower()

        # Define keyword categories
        planning_keywords = [
            "plan", "steps", "organize", "task", "roadmap", "schedule", 
            "structure", "workflow", "timeline", "strategy", "approach"
        ]
        
        tool_keywords = [
            "calculate", "search", "weather", "time", "news", "current", 
            "what", "who", "when", "where", "convert", "find", "lookup"
        ]
        
        reflection_keywords = [
            "think", "reflect", "why", "reason", "feeling", "analyze", 
            "thoughts", "opinion", "assess", "evaluate", "review"
        ]

        # Check for keyword matches
        if any(word in input_lower for word in planning_keywords):
            return "PlanningAgent"
        elif any(word in input_lower for word in tool_keywords):
            return "ToolUserAgent"
        elif any(word in input_lower for word in reflection_keywords):
            return "ReflectionAgent"
        else:
            # Default fallback
            return "ReflectionAgent"
    
    def think(self, user_input: str) -> str:
        """
        Process user input by routing to appropriate agent and storing conversation.
        
        Args:
            user_input: The user's input text
            
        Returns:
            str: Agent's response
        """
        # Decide which agent to use
        selected_agent = self.decide_agent(user_input)
        
        # Route to appropriate agent
        try:
            if selected_agent == "PlanningAgent":
                result = self.planner.think(user_input)
            elif selected_agent == "ToolUserAgent":
                result = self.tooler.think(user_input)
            else:  # ReflectionAgent
                result = self.reflector.think(user_input)
                
        except Exception as e:
            print(f"❌ Error with {selected_agent}: {e}")
            # Fallback to reflection agent if selected agent fails
            result = self.reflector.think(user_input)
            selected_agent = "ReflectionAgent (fallback)"

        # Store conversation in memory
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.memory.append({
            "role": "user",
            "text": user_input,
            "timestamp": timestamp,
            "agent_selected": selected_agent
        })
        
        self.memory.append({
            "role": "agent",
            "text": result,
            "timestamp": timestamp,
            "agent_type": selected_agent
        })

        return result
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        
        Returns:
            List of conversation entries
        """
        return self.memory.copy()
    
    def clear_memory(self) -> None:
        """Clear conversation history."""
        self.memory.clear()
        print("🧹 Conversation history cleared")
    
    def get_agent_stats(self) -> Dict[str, int]:
        """
        Get statistics about which agents were used.
        
        Returns:
            Dictionary with agent usage counts
        """
        stats = {"PlanningAgent": 0, "ReflectionAgent": 0, "ToolUserAgent": 0}
        
        for entry in self.memory:
            if entry["role"] == "agent":
                agent_type = entry.get("agent_type", "").split(" ")[0]  # Remove "(fallback)" if present
                if agent_type in stats:
                    stats[agent_type] += 1
        
        return stats
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Set the confidence threshold for ML predictions.
        
        Args:
            threshold: Confidence threshold (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            print(f"✅ Confidence threshold set to {threshold}")
        else:
            print("❌ Threshold must be between 0.0 and 1.0")
    
    def __str__(self) -> str:
        """String representation of the coordinator."""
        model_status = "✅ Loaded" if self.model is not None else "❌ Not loaded"
        return f"CoordinatorAgent(model={model_status}, conversations={len(self.memory)//2})"


# Example usage and testing
if __name__ == "__main__":
    # Initialize coordinator
    coordinator = CoordinatorAgent()
    
    # Test different types of inputs
    test_inputs = [
        "How should I organize my project timeline?",
        "What went wrong with my approach?",
        "Calculate 25 multiplied by 48",
        "Help me plan my day",
        "What are your thoughts on this?",
        "What's the weather like today?"
    ]
    
    print("🧪 Testing Coordinator Agent:")
    print("=" * 50)
    
    for inp in test_inputs:
        print(f"\nInput: '{inp}'")
        response = coordinator.think(inp)
        print(f"Response: {response[:100]}...")  # Show first 100 chars
    
    # Show statistics
    print(f"\n📊 Agent Usage Stats: {coordinator.get_agent_stats()}")
    print(f"🤖 Coordinator Status: {coordinator}")