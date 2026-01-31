# TensorRT-LLM with Triton Inference Server Setup

This guide walks you through setting up inference for the **Mistral 7B (7 billion parameter)** model using **TensorRT-LLM** and **Triton Inference Server** on a VM or bare metal with GPU.

## Prerequisites

- GPU-enabled VM or bare metal server with GPU
- NVIDIA GPU drivers installed
- Docker and NVIDIA Container Toolkit installed

## Setup Steps

### Step 1: Initialize Environment

Run the initialization script to install all necessary dependencies and clone the TensorRT-LLM repository:

```bash
bash init.sh
```

This script will:
- Update system packages
- Install Python virtual environment
- Install HuggingFace Hub CLI
- Install Git LFS
- Clone the TensorRT-LLM repository to `~/TensorRT-LLM`

### Step 2: Download Model from HuggingFace

Download the Mistral-7B-Instruct-v0.3 model and store it locally:

```bash
hf download mistralai/Mistral-7B-Instruct-v0.3 --local-dir ./Mistral-7B-Instruct-v0.3
```

**Note:** Adjust the path (`./Mistral-7B-Instruct-v0.3`) to your desired location. The model will be mounted into the container in the next step.

### Step 3: Start Triton Server Container

Start the Triton Inference Server container with the TensorRT-LLM backend and model mounted:

```bash
docker run --rm -it --net host --shm-size=2g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus all \
  -v "./TensorRT-LLM":/opt/tritonserver/tensorrtllm_backend \
  -v "/home/shadeform/Mistral-7B-Instruct-v0.3":/models/Mistral-7B-Instruct-v0.3:ro \
  -v "$(pwd)/tensorrt-llm-with-triton-server":/workspace \
  nvcr.io/nvidia/tritonserver:25.12-trtllm-python-py3 bash
```

**Important:** 
- Adjust the volume mount paths to match your actual paths:
  - `./TensorRT-LLM` should point to your cloned TensorRT-LLM repository
  - `/home/shadeform/Mistral-7B-Instruct-v0.3` should point to your downloaded model directory
  - `$(pwd)/tensorrt-llm-with-triton-server` mounts the setup scripts directory
- The `:ro` flag mounts the model directory as read-only
- The container uses `--net host` for network access

Once the container starts, you'll be inside the container's bash shell and can proceed with building engines and configuring Triton.

### Step 4: Setup Triton Server for Inferencing

Inside the container, copy and run the `triton-server-init.sh` script to set up the Triton server:

```bash
# Copy the script to a convenient location
cp /workspace/triton-server-init.sh /opt/tritonserver/

# Make it executable
chmod +x /opt/tritonserver/triton-server-init.sh

# Run the setup script (this will build engines and start Triton)
bash /opt/tritonserver/triton-server-init.sh --start
```

This script will:
- Convert the HuggingFace checkpoint to TensorRT-LLM format
- Build optimized TensorRT engines for your GPU
- Create and configure the Triton model repository
- Start the Triton Inference Server

**Note:** The engine building step can take 5-30 minutes depending on your GPU. The script will automatically start the Triton server once setup is complete.

### Step 5: Test Inference

Once the Triton server is running (you'll see it start in the container), open **another terminal** on your host machine and test the inference endpoint:

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{
  "text_input": "Howdy how life ?",
  "parameters": {
    "max_tokens": 500,
    "bad_words": [""],
    "stop_words": [""]
  }
}'
```

**Expected response:** You should receive a JSON response with the generated text from the model.

**Note:** 
- The server must be fully started before testing (wait for "Uvicorn running on..." or similar messages)
- If you're testing from a remote machine, replace `localhost` with the server's IP address
- The endpoint uses the `ensemble` model which routes through preprocessing → inference → postprocessing
