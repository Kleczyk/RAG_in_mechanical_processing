#!/usr/bin/env python3
"""
Gradio interface for the machining RAG pipeline defined in chain_process.py.
"""
import gradio as gr
from proces_chain import process_image, list_sample_images, SAMPLE_IMAGES_FOLDER


def build_interface() -> gr.Blocks:
    """
    Construct and return the Gradio Blocks UI.
    """
    sample_images = list_sample_images(SAMPLE_IMAGES_FOLDER)

    with gr.Blocks() as demo:
        gr.Markdown("## Machining RAG Pipeline")
        with gr.Row():
            with gr.Column(scale=1):
                upload_input = gr.Image(
                    type="filepath",
                    label="Upload Image"
                )
                select_input = gr.Dropdown(
                    choices=[""] + sample_images,
                    value="",
                    label="Select Sample Image",
                    interactive=True
                )
                process_btn = gr.Button("Process")

            with gr.Column(scale=2):
                csv_output_box = gr.Textbox(
                    label="Retrieved CSV Tables",
                    lines=10
                )
                gpt_response_md = gr.Markdown(
                    label="GPT-4 Vision Output"
                )

        process_btn.click(
            fn=process_image,
            inputs=[upload_input, select_input],
            outputs=[csv_output_box, gpt_response_md]
        )

    return demo


if __name__ == "__main__":
    interface = build_interface()
    interface.launch(share=True)
