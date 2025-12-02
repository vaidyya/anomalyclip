# AnomalyCLIP: Real-Time Video Anomaly Detection

AnomalyCLIP is a powerful tool for real-time video anomaly detection, leveraging the capabilities of OpenAI's CLIP model and large language models (LLMs) to identify and summarize unusual events in video streams.

![Architecture](media/architecture.png)

## Features

- **Real-time Anomaly Detection**: Identifies anomalous events in video streams with low latency.
- **LLM-Powered Summaries**: Uses local large language models via Ollama to generate summaries of detected anomalies.
- **Web-Based Dashboard**: An intuitive web interface for video upload, live analysis, and viewing results.
- **Cross-Platform**: Supports macOS, Linux, and Windows.
- **GPU Acceleration**: Supports MPS for Apple Silicon and can be extended for CUDA.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- **Conda**: The project uses Conda for environment management. Please [install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) before proceeding.
- **Ollama**: For anomaly summarization, you need to have [Ollama](https://ollama.com/) installed and running.

### Ollama Setup

1.  **Install and Run Ollama**:
    - **macOS/Linux**:
        ```bash
        brew install ollama && ollama serve
        ```
    - **Windows**: Download and run the installer from the [Ollama website](https://ollama.com/).

2.  **Download Required Models**:
    You need the `phi` and `moondream` models for the AI summarization feature. Run the following commands:

    ```bash
    ollama pull phi
    ollama pull moondream
    ```

## Installation

1.  **Clone the Repository**:

    ```bash
    git clone https://github.com/your-username/AnomalyCLIP-main.git
    cd AnomalyCLIP-main
    ```

2.  **Create the Conda Environment**:
    This will create a Conda environment named `myenv` with all the necessary dependencies.

    ```bash
    conda env create -f environment.yaml
    conda activate myenv
    ```

3.  **Download Checkpoints and Labels**:
    The model requires pre-trained checkpoints and label files.

    - Download the model checkpoints and place them in the `checkpoints/` directory. For example, `checkpoints/ucfcrime/last.ckpt`.
    - Download the label files and place them in the `data/` directory. For example, `data/ucf_labels.csv`.

## Running the Dashboard

Once the installation is complete, you can start the AnomalyCLIP dashboard.

-   **On macOS and Linux**:

    ```bash
    ./start_dashboard.sh
    ```

-   **On Windows**:

    ```bat
    .\start_dashboard.bat
    ```

The script will perform checks, set up the environment, and start the FastAPI server.

## Usage

1.  Open your web browser and navigate to [http://localhost:8000](http://localhost:8000).
2.  Upload a video file using the web interface.
3.  The video will be processed in real-time, and the dashboard will display:
    - The video stream.
    - A timeline with anomaly scores.
    - AI-generated summaries of detected anomalies (if Ollama is running).

## Configuration

The behavior of the server can be configured through environment variables, which are set in the `start_dashboard.sh` and `start_dashboard.bat` scripts: `ANOMALYCLIP_DEVICE`: The device to run the model on (`cpu`, `mps`, `cuda`). `SERVE_INFER_INTERVAL`: The frame interval for running inference. `SERVE_MAX_EMIT_FPS`: The maximum FPS for emitting results to the client. `SERVE_MAX_WIDTH`: The maximum width of the video frames to be processed.