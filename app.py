import gradio as gr

def respond(message, history):
    return "Hello! I am your RAG Complaint Chatbot. How can I help you today?"

demo = gr.ChatInterface(fn=respond, title="RAG Complaint Chatbot")

if __name__ == "__main__":
    demo.launch()
