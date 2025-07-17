import streamlit as st
from src.multi_agent import CoordinatorAgent

st.set_page_config(page_title="AgentX", layout="centered")
st.title("🧠 AgentX — Multi-Agent Streamlit App")

# Initialize agent
if "agent" not in st.session_state:
    st.session_state.agent = CoordinatorAgent()

# Reset button
if st.sidebar.button("🔄 Reset Multi-Agent"):
    st.session_state.agent = CoordinatorAgent()
    st.rerun()

# Prompt
st.markdown("### 💬 Enter your message and let the system route it automatically:")
user_input = st.text_input("Your Message", key="input")

# Handle interaction
if st.button("Send") and user_input.strip():
    agent = st.session_state.agent
    with st.spinner("🤖 Thinking..."):
        response = agent.think(user_input)

    st.markdown("### 📡 Response")
    st.success(response)

    st.markdown("### 🧠 Memory Log")
    for i, entry in enumerate(agent.memory, 1):
        with st.expander(f"{i}. {entry.get('role', '').capitalize()} @ {entry.get('timestamp', '')}"):
            st.write(entry.get("text", ""))
