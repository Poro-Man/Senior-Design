# src/ui.py
"""
Premium Gradio Chat UI for Mini LLaMA.
Connects to the FastAPI backend at /generate.
"""

import requests
import gradio as gr

API_URL = "http://127.0.0.1:8000/generate"

# ── Custom dark theme ──
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#0f0f13",
    body_background_fill_dark="#0f0f13",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
    block_background_fill="#1a1a24",
    block_background_fill_dark="#1a1a24",
    block_border_color="#2a2a3a",
    block_border_color_dark="#2a2a3a",
    block_label_text_color="#94a3b8",
    block_label_text_color_dark="#94a3b8",
    block_title_text_color="#e2e8f0",
    block_title_text_color_dark="#e2e8f0",
    input_background_fill="#12121a",
    input_background_fill_dark="#12121a",
    input_border_color="#2a2a3a",
    input_border_color_dark="#2a2a3a",
    button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
    button_primary_background_fill_dark="linear-gradient(135deg, #6366f1, #8b5cf6)",
    button_primary_background_fill_hover="linear-gradient(135deg, #818cf8, #a78bfa)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #818cf8, #a78bfa)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#1e1e2e",
    button_secondary_background_fill_dark="#1e1e2e",
    button_secondary_text_color="#94a3b8",
    button_secondary_text_color_dark="#94a3b8",
    border_color_primary="#6366f1",
    border_color_primary_dark="#6366f1",
)

# ── Custom CSS ──
custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}
#title-text {
    text-align: center;
    padding: 1.2rem 0 0.2rem;
}
#title-text h1 {
    background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
#subtitle-text {
    text-align: center;
    padding-bottom: 0.8rem;
}
#subtitle-text p {
    color: #64748b;
    font-size: 0.9rem;
}
#footer-text {
    text-align: center;
    padding: 0.5rem 0;
}
#footer-text p {
    color: #475569;
    font-size: 0.75rem;
}
textarea {
    border-radius: 12px !important;
    font-size: 0.95rem !important;
}
button, textarea, input {
    transition: all 0.2s ease !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f0f13; }
::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3a3a4a; }
"""


# ── API call ──
def call_api(prompt, max_new_tokens, temperature, top_p):
    try:
        r = requests.post(
            API_URL,
            json={
                "prompt": prompt,
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
            },
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["text"]
    except requests.ConnectionError:
        return "Could not connect to the server. Make sure the FastAPI backend is running on port 8000."
    except Exception as e:
        return f"Error: {e}"


def build_prompt(history, user_msg):
    lines = []
    for m in history or []:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    lines.append(f"User: {user_msg}")
    lines.append("Assistant:")
    return "\n".join(lines)


def user_submit(user_msg, history, max_tokens, temperature, top_p):
    """Called when user sends a message. Returns updated history."""
    if not user_msg.strip():
        return history, ""
    history = history or []
    history.append({"role": "user", "content": user_msg})
    prompt = build_prompt(history[:-1], user_msg)
    response = call_api(prompt, max_tokens, temperature, top_p)
    history.append({"role": "assistant", "content": response})
    return history, ""


# ── Build UI with Blocks ──
with gr.Blocks(title="Mini LLaMA Chat") as demo:

    gr.Markdown("# Mini LLaMA", elem_id="title-text")
    gr.Markdown("A lightweight LLaMA-style language model &mdash; 244M parameters, built from scratch", elem_id="subtitle-text")

    chatbot = gr.Chatbot(
        height=480,
        elem_id="chatbot",
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your message here...",
            scale=7,
            container=False,
            elem_id="msg-input",
        )
        send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

    with gr.Accordion("Generation Settings", open=False):
        with gr.Row():
            max_tokens = gr.Slider(
                minimum=1, maximum=512, value=128, step=1,
                label="Max Tokens",
                info="Maximum number of tokens to generate",
            )
            temperature = gr.Slider(
                minimum=0.0, maximum=2.0, value=0.8, step=0.05,
                label="Temperature",
                info="Higher = more creative, lower = more focused",
            )
            top_p = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.95, step=0.01,
                label="Top-p",
                info="Nucleus sampling threshold",
            )

    with gr.Row():
        clear_btn = gr.Button("Clear Chat", variant="secondary", size="sm")

    gr.Markdown("Mini LLaMA &bull; 244M parameters &bull; Powered by PyTorch & Gradio", elem_id="footer-text")

    # ── Event handlers ──
    submit_inputs = [msg, chatbot, max_tokens, temperature, top_p]
    submit_outputs = [chatbot, msg]

    msg.submit(user_submit, inputs=submit_inputs, outputs=submit_outputs)
    send_btn.click(user_submit, inputs=submit_inputs, outputs=submit_outputs)
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

demo.launch(theme=theme, css=custom_css)
