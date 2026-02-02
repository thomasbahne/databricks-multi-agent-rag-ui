"""
Multi-Agent RAG Chat Interface
Gradio app deployed via Databricks Apps
"""

import os
import gradio as gr
from databricks.sdk import WorkspaceClient

# Agent endpoint mapping: display name -> serving endpoint name
AGENTS = {
    "Agent A - Product Docs": "agent-a-rag",
    "Agent B - Technical Specs": "agent-b-rag",
    "Agent C - Support KB": "agent-c-rag",
}

# Initialize Databricks client (uses app service principal)
w = WorkspaceClient()


def get_openai_client(endpoint_name: str):
    """Get OpenAI-compatible client for a serving endpoint."""
    return w.serving_endpoints.get_open_ai_client()


def chat(message: str, history: list, agent_name: str):
    """Send message to selected agent endpoint."""
    if not message.strip():
        return history, ""

    endpoint_name = AGENTS.get(agent_name)
    if not endpoint_name:
        history.append((message, "Error: No agent selected"))
        return history, ""

    # Build messages from history
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    try:
        client = get_openai_client(endpoint_name)
        response = client.chat.completions.create(
            model=endpoint_name,
            messages=messages,
            max_tokens=1024,
        )
        assistant_response = response.choices[0].message.content
    except Exception as e:
        assistant_response = f"Error: {e}"

    history.append((message, assistant_response))
    return history, ""


def clear_chat():
    """Clear chat history."""
    return [], ""


# Custom CSS for polished dark theme
css = """
:root {
    --bg-primary: #0f0f12;
    --bg-secondary: #1a1a22;
    --bg-tertiary: #252530;
    --accent: #6366f1;
    --accent-hover: #818cf8;
    --text-primary: #f4f4f5;
    --text-secondary: #a1a1aa;
    --border: #333340;
}

.gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'JetBrains Mono', 'SF Mono', monospace !important;
}

.main-header {
    text-align: center;
    padding: 2rem 0 1rem;
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}

.main-header h1 {
    color: var(--text-primary);
    font-size: 1.75rem;
    font-weight: 600;
    margin: 0;
    letter-spacing: -0.02em;
}

.main-header p {
    color: var(--text-secondary);
    font-size: 0.875rem;
    margin: 0.5rem 0 0;
}

#chatbot {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    min-height: 500px !important;
}

#chatbot .message {
    border-radius: 8px !important;
    padding: 12px 16px !important;
}

#chatbot .user {
    background: var(--accent) !important;
    color: white !important;
}

#chatbot .bot {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
}

.agent-selector {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    margin-bottom: 1rem !important;
}

.agent-selector label {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

button.primary {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: background 0.2s ease !important;
}

button.primary:hover {
    background: var(--accent-hover) !important;
}

button.secondary {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
}

textarea, input {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

textarea:focus, input:focus {
    border-color: var(--accent) !important;
    outline: none !important;
}
"""

with gr.Blocks(css=css, title="RAG Chat") as app:
    gr.HTML("""
        <div class="main-header">
            <h1>Multi-Agent RAG Chat</h1>
            <p>Select an agent and start chatting</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            agent_dropdown = gr.Dropdown(
                choices=list(AGENTS.keys()),
                value=list(AGENTS.keys())[0] if AGENTS else None,
                label="Select Agent",
                elem_classes=["agent-selector"],
            )
            clear_btn = gr.Button("Clear Chat", variant="secondary")

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                height=500,
                show_label=False,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type your message...",
                    show_label=False,
                    scale=9,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

    # Event handlers
    msg_input.submit(
        chat,
        inputs=[msg_input, chatbot, agent_dropdown],
        outputs=[chatbot, msg_input],
    )
    send_btn.click(
        chat,
        inputs=[msg_input, chatbot, agent_dropdown],
        outputs=[chatbot, msg_input],
    )
    clear_btn.click(clear_chat, outputs=[chatbot, msg_input])

    # Clear chat when agent changes
    agent_dropdown.change(clear_chat, outputs=[chatbot, msg_input])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8000)
