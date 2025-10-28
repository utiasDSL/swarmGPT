"""GUI module for the gradio web app."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, List

import gradio as gr

if TYPE_CHECKING:
    from swarm_gpt.core import AppBackend


def padding_column():
    """Create a column with a hidden textbox to add padding to the UI."""
    with gr.Column():
        gr.Textbox(visible=False)


def centered_markdown(text: str) -> gr.Markdown:
    """Create a centered markdown element.

    Args:
        text: The text to display.

    Returns:
        A markdown element formatted to be centered.
    """
    md = f'<div align="center"> <font size = "10"> <span style="color:grey">{text}</span>'
    return gr.Markdown(md, visible=False)


def update_visibility(visible_flags: List[bool]) -> Callable:
    """Update the visibility of the UI elements.

    We return a function that returns the gradio updates since gradio expects a function instead of
    plain update values.

    Args:
        visible_flags: A list of booleans indicating whether the UI elements should be visible.

    Returns:
        A function that returns the list of gradio updates for the UI elements.
    """

    def gradio_ui_update() -> List[dict]:
        return [gr.update(visible=x) for x in visible_flags]

    return gradio_ui_update


def run_with_bar(backend: AppBackend, progress: gr.Progress = gr.Progress(track_tqdm=True)) -> str:
    """Run the simulation with a progress bar."""
    # Get the generator from your simulation code
    for key, data, total in backend.simulate():
        if key == "progress":
            if data != total:
                percent = int(data / total)
                progress(percent, desc="Simulation Loading...", total=100)
            else:
                progress(100, desc="Simulation Playing", total=100)
        else:
            return "Simulation Playing!"


def create_ui(backend: AppBackend) -> gr.Blocks:
    """Create the gradio web app.

    Args:
        backend: The app backend. This is used to connect the UI to the simulator, AMSwarm and the
            ROS nodes that execute the choreography.

    Returns:
        The UI.
    """
    # Ignore gradio renaming warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="api_name")
    # Define the UI
    with gr.Blocks(theme=gr.themes.Monochrome()) as ui:
        gr.Markdown(
            """ <div align="center"> <font size = "50"> SwarmGPT-Primitive""", elem_id="swarmgpt"
        )
        # Initial window with song selection
        with gr.Row():
            padding_column()
            with gr.Column():
                song_input = gr.Dropdown(
                    choices=backend.songs + backend.presets, label="Select song"
                )
            with gr.Column():
                prompt_choices = list(backend.choreographer.prompts.keys())
                gr.Dropdown(
                    choices=prompt_choices,
                    label="Enter prompt type:",
                    visible=False,
                    interactive=True,
                )
        # Interface during data processing and simulation
        with gr.Row():
            with gr.Column():
                replay_msg = centered_markdown("Replaying simulation")
                sim_msg = centered_markdown("Simulating safe choreography")
                choreo_msg = centered_markdown("LLM is generating choreography")
        # Chatbot and message display
        chatbot = gr.Chatbot(visible=False, type="messages")
        message = gr.Textbox(label="Enter prompt:", visible=False)

        with gr.Row():
            with gr.Column():
                progress_bar = gr.Textbox("Progress", visible=False)

        with gr.Row():
            padding_column()
            with gr.Column():
                replay_sim_button = gr.Button("Replay simulation", visible=False)
                sim_button = gr.Button("Simulate", visible=False)
            with gr.Column():
                alter_button = gr.Button("Refine/Modify the choreography", visible=False)
            padding_column()

        with gr.Row():
            padding_column()
            with gr.Column():
                select_song_button = gr.Button("Choose another song", visible=False)
            padding_column()

        with gr.Row():
            padding_column()
            with gr.Column():
                start_button = gr.Button("Send selections to LLM", visible=False)
                deploy_button = gr.Button("Let the Crazyflies dance", visible=False)
                save_preset_button = gr.Button("Save preset", visible=False)
                show_output = gr.Checkbox(
                    label="Display conversation with LLM",
                    visible=False,
                    value=False,
                    container=True,
                    interactive=True,
                )
            padding_column()

        # Define the UI control flow when the user interacts with the UI elements
        # Song selection flow. On select, the start button and the show output checkbox appear.
        song_input.select(update_visibility([True, True]), [], [start_button, show_output])
        # Start button flow. On click, the song input and start button disappear
        # The choreo message appears
        start_button_flow = start_button.click(
            update_visibility([False, False, True]), [], [song_input, start_button, choreo_msg]
        )
        # The song is handed to the backend start function, and the output of `start` is piped into
        # the chatbot.
        start_button_flow = start_button_flow.success(backend.initial_prompt, song_input, chatbot)
        # The choreo message disappears and the simulate, modify and select song buttons appear
        start_button_flow = start_button_flow.success(
            update_visibility([False, True, True, True, True, True]),
            [],
            [
                choreo_msg,
                sim_button,
                alter_button,
                select_song_button,
                deploy_button,
                save_preset_button,
            ],
        )

        # Alter waypoints flow
        alter_button_flow = alter_button.click(
            lambda: gr.update(visible=True, value=None), [], [message]
        )
        alter_button_flow = alter_button_flow.success(
            update_visibility([False, False, False, False, True]),
            [],
            [alter_button, deploy_button, replay_sim_button, sim_button, chatbot],
        )

        # Show output of the LLM if the checkbox is checked
        def on_select(evt: gr.SelectData) -> dict:
            return gr.update(visible=evt.value)

        show_output.select(on_select, [], [chatbot])  # Toggle chatbot visibility

        # Message flow
        message_flow = message.submit(
            update_visibility([False, False, True]), [], [sim_msg, replay_msg, choreo_msg]
        )
        message_flow = message_flow.success(backend.reprompt, [message], [chatbot])
        message_flow = message_flow.success(
            update_visibility([False, False, True, True, False]),
            [],
            outputs=[alter_button, choreo_msg, sim_button, deploy_button, replay_sim_button],
        )
        message_flow = message_flow.success(
            lambda: gr.update(visible=True, value=None), [], message
        )

        # Sim button flow. On click, the sim message appears and all other messages disappear.
        sim_button_flow = sim_button.click(
            update_visibility([False, False, True, True]),
            [],
            [replay_msg, choreo_msg, sim_msg, progress_bar],
        )
        # AMSwarm is launched and the resulting trajectories are simulated
        sim_button_flow = sim_button_flow.success(
            lambda: run_with_bar(backend), outputs=progress_bar
        )

        # The buttons reappear and the sim message disappears
        sim_button_flow = sim_button_flow.success(
            update_visibility([False, False, True, True, True, True, False]),
            [],
            [
                sim_msg,
                sim_button,
                replay_sim_button,
                alter_button,
                deploy_button,
                select_song_button,
                progress_bar,
            ],
        )
        # Deploy button flow
        deploy_button.click(backend.deploy, [], chatbot)

        # Save preset button flow
        save_preset_button.click(backend.save_preset, [], [])

        # Replay sim button flow
        replay_sim_flow = replay_sim_button.click(
            update_visibility([False, True]), [], [sim_msg, replay_msg]
        )

        replay_sim_flow = replay_sim_flow.success(
            lambda: run_with_bar(backend), outputs=progress_bar
        )

        replay_sim_flow = replay_sim_flow.success(
            update_visibility([False, True]), [], [replay_msg, select_song_button]
        )
        select_song_button.click(None, js="window.location.reload()")
    return ui
