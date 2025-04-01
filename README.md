# ICLight Custom Node for ComfyUI

This repository contains a custom node for ComfyUI that integrates with [FAL.ai](https://fal.ai/)'s ICLight v2 model. It allows you to relight images within your ComfyUI workflows using text prompts via the `fal-client` library.

## Features

*   Relight images using FAL.ai's ICLight v2 model.
*   Integrates directly into ComfyUI workflows.
*   Supports text prompts, negative prompts, and various generation parameters.
*   Optional masking for targeted relighting.
*   Uses the official `fal-client` for API interaction.

## Installation

1.  **Clone the Repository:**
    Navigate to your ComfyUI `custom_nodes` directory and clone this repository:
    ```bash
    cd /path/to/ComfyUI/custom_nodes/
    git clone https://github.com/Synthetic-Media-Lab/ICLight_CustomNode.git
    ```

2.  **Install Dependencies:**
    This node requires the `fal-client` library. Install it using pip:
    ```bash
    pip install fal-client requests Pillow numpy torch
    # Or activate your ComfyUI virtual environment first, then pip install
    ```
    *(Note: `requests`, `Pillow`, `numpy`, and `torch` are usually already present in a standard ComfyUI setup, but are included for completeness).*

3.  **Restart ComfyUI:** Ensure ComfyUI recognizes the new node.

## Configuration

This node requires an API key from FAL.ai to function.

1.  **Get API Key:** Obtain your API key from your [FAL.ai account](https://fal.ai/).
2.  **Set Environment Variable:** Set the `FAL_KEY` environment variable before launching ComfyUI. How you do this depends on your operating system and how you run ComfyUI:
    *   **Linux/macOS (Terminal):**
        ```bash
        export FAL_KEY="your_fal_api_key_here"
        python main.py # or however you launch ComfyUI
        ```
    *   **Windows (Command Prompt):**
        ```cmd
        set FAL_KEY=your_fal_api_key_here
        python main.py
        ```
    *   **Windows (PowerShell):**
        ```powershell
        $env:FAL_KEY = "your_fal_api_key_here"
        python main.py
        ```
    *   **(Alternative) `.env` file:** You might also use a `.env` file in your ComfyUI root directory if your setup supports it (e.g., using `python-dotenv`). Add the line `FAL_KEY=your_fal_api_key_here` to the `.env` file.

## Usage

1.  Open ComfyUI.
2.  Right-click on the canvas, select "Add Node".
3.  Navigate to the `external/generation` category.
4.  Select the `FAL.ai ICLight v2` node.
5.  Connect the required inputs and adjust parameters as needed.

## Node Inputs & Outputs

### Inputs

*   `image`: The input image to be relit (IMAGE tensor).
*   `prompt`: (STRING) Text description of the desired lighting and scene.
*   `negative_prompt`: (STRING) Text description of elements to avoid.
*   `num_inference_steps`: (INT) Number of diffusion steps (Default: 28).
*   `guidance_scale`: (FLOAT) How strictly the model should follow the prompt (Default: 5.0).
*   `seed`: (INT) Random seed for generation. -1 uses a random seed (Default: -1).
*   `initial_latent`: (STRING) Optional lighting condition preset ("None", "Left", "Right", "Top", "Bottom"). (Default: "None").
*   `output_format`: (STRING) Format for the output image ("jpeg" or "png"). (Default: "jpeg").
*   `mask` (Optional): An optional mask image to restrict relighting to specific areas (IMAGE tensor).

### Outputs

*   `image`: The relit image (IMAGE tensor).

## Dependencies

*   [fal-client](https://github.com/fal-ai/fal-client-python)
*   requests
*   Pillow
*   NumPy
*   PyTorch (usually included with ComfyUI)

## License

(Consider adding a license file, e.g., MIT or Apache 2.0)

---

*This README provides a starting point. Feel free to add examples, screenshots, or further details.*
