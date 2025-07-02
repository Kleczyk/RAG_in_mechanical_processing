# File: app.py
import os
import gradio as gr
from proces_chain import list_sample_images, process_image, followup_plan, SAMPLE_FOLDER


def start_convo(upload, sample, history):
    csv_str, initial = process_image(upload, sample)
    history = history or []
    history.append(('Assistant', initial))
    return history, history


def chat_turn(msg, history):
    history = history or []
    history.append((msg, None))
    initial = history[0][1]
    updated = followup_plan(initial, msg)
    history[-1] = (msg, updated)
    return history, history


def build_interface():
    samples = list_sample_images()
    with gr.Blocks() as demo:
        gr.Markdown('## Machining RAG Chat')
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.Image(type='filepath', label='Upload Image')
                sample = gr.Dropdown([''] + samples, label='Or choose sample')
                # Preview of selected sample
                sample_preview = gr.Image(type='filepath', label='Sample Preview')
                # Update preview when dropdown changes
                sample.change(
                    fn=lambda fn: os.path.join(SAMPLE_FOLDER, fn) if fn else None,
                    inputs=[sample],
                    outputs=[sample_preview]
                )
                start = gr.Button('Start Chat')
                msg_in = gr.Textbox(placeholder='Adjust the plan...', label='Your message')
                send = gr.Button('Send')
            with gr.Column(scale=2):
                chat = gr.Chatbot()
            state = gr.State([])

        start.click(start_convo, inputs=[upload, sample, state], outputs=[chat, state])
        send.click(chat_turn, inputs=[msg_in, state], outputs=[chat, state])
    return demo

if __name__ == '__main__':
    build_interface().launch(share=True)

