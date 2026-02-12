import requests
import gradio as gr

API_URL = "http://127.0.0.1:8000/generate"

def call_api(prompt, max_new_tokens, temperature, top_p):
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

def build_prompt(history, user_msg):
    """
    history is a list of message dicts:
      [{"role": "user"/"assistant", "content": "...", ...}, ...]
    """
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

def respond(user_msg, history, max_new_tokens, temperature, top_p):
    # IMPORTANT: return ONLY the assistant message (string)
    prompt = build_prompt(history, user_msg)
    out = call_api(prompt, max_new_tokens, temperature, top_p)
    return out

demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Slider(1, 512, value=128, step=1, label="max_new_tokens"),
        gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="temperature"),
        gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="top_p"),
    ],
    title="Mini LLM Chat (Gradio → FastAPI → Model)",
)

demo.launch()
