import gradio as gr
import requests
import uuid
import json

API_URL = "https://2ek47g919h.execute-api.us-east-1.amazonaws.com/dev/handleTravelPlan"
session_id = str(uuid.uuid4())
user_id = f"user-{uuid.uuid4()}"

conversation_history = []


def chat_with_lambda(user_message, history):
    payload = {
        "message": user_message,
        "session_id": session_id,
        "user_id": user_id
    }
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            reply = data.get("reply", "No response from server.")

            travel_plan = data.get("getPlanResult")
            if travel_plan:
                reply += "\n\n\U0001F4E6 Travel Plan:\n" + json.dumps(travel_plan, indent=2)

        else:
            reply = f"❌ Server error: {response.status_code}"
    except Exception as e:
        reply = f"❌ Error: {str(e)}"

    history.append((user_message, reply))
    return "", history


def reset():
    global session_id, user_id, conversation_history
    session_id = str(uuid.uuid4())
    user_id = f"user-{uuid.uuid4()}"
    conversation_history = []
    return [], ""


with gr.Blocks() as demo:
    gr.Markdown("## 👛 Travel Assistant - Powered by AWS Lambda & Claude via Bedrock")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your message")
    clear = gr.Button("🔄 New Session")

    msg.submit(chat_with_lambda, [msg, chatbot], [msg, chatbot])
    clear.click(reset, outputs=[chatbot, msg])


if __name__ == "__main__":
    demo.launch()
