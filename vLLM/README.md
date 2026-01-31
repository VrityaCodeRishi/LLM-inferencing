# LLM Inferencing: vLLM (Hello World on 1× L40S)

This repo is a **step-by-step, copy/paste** starter for running **vLLM** on a single NVIDIA **L40S** GPU instance.

## 0) What you’ll build

- A local **OpenAI-compatible** chat/completions API served by vLLM on your GPU
- A **curl** test
- A tiny **Python client** test

---

## 1) Prereqs on the GPU instance

Assumes **Ubuntu 22.04/24.04** with an NVIDIA driver installed.

### 1.1 Verify the GPU is visible

```bash
nvidia-smi
```

If this fails, fix your NVIDIA driver install first (this is outside vLLM).

### 1.2 Install Docker + NVIDIA Container Toolkit

Install Docker Engine and NVIDIA Container Toolkit for your distro.

- Docker docs: `https://docs.docker.com/engine/install/`
- NVIDIA Container Toolkit docs: `https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html`

After installation, verify containers can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## 2) Choose a model (recommended first model)

For a smooth first run, use an **ungated** model:

- Recommended: `mistralai/Mistral-7B-Instruct-v0.2`

If you want Llama models, you may need to accept model terms and use a Hugging Face token.

---

## 3) Start vLLM (OpenAI-compatible server) with Docker

### 3.1 (Optional) Hugging Face token

If your model requires auth, set:

```bash
export HF_TOKEN="YOUR_HF_TOKEN"
```

### 3.1.1 (Optional) If you prefer docker-compose

This repo includes `docker-compose.vllm.yml`:

```bash
export HF_TOKEN="YOUR_HF_TOKEN"   # optional
docker compose -f docker-compose.vllm.yml up
```

### 3.2 Run the server

From anywhere on your instance:

```bash
docker run --rm --gpus all --ipc=host \
  -p 8000:8000 \
  -e HF_TOKEN \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu121 \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --dtype bfloat16 \
  --max-model-len 8192
```

**Note:** Using `:cu121` (CUDA 12.1) instead of `:latest` for better driver compatibility. If you still get CUDA errors, see troubleshooting section 6.4 below.

Notes:
- `-v ~/.cache/huggingface:...` persists model downloads across restarts.
- `--ipc=host` avoids shared-memory issues on some setups.
- If you hit OOM, reduce `--max-model-len` (try 4096) and/or add `--gpu-memory-utilization 0.90`.

---

## 4) “Hello world” test with curl

In a second terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "messages": [{"role":"user","content":"Say hello world in one short sentence."}],
    "temperature": 0
  }'
```

You should get JSON back containing `choices[0].message.content`.

Tip: there’s also a tiny script in this repo:

```bash
chmod +x smoke_test.sh
./smoke_test.sh
```

---

## 5) “Hello world” test with Python

On your laptop or the same instance, install the Python deps and run the client in this repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python client_vllm_openai.py
```

If your server is on a remote instance, set:

```bash
export VLLM_BASE_URL="http://YOUR_INSTANCE_IP:8000/v1"
```

---

## 6) Common issues (quick fixes)

### 6.1 Port not reachable from your laptop

- Ensure your cloud security group / firewall allows inbound TCP **8000**.
- For a quick local-only test, run curl on the instance.

### 6.2 “401 / gated repo” / model download fails

- Use an ungated model (recommended for first run), or
- Set `HF_TOKEN` and ensure you have access to the model on Hugging Face.

### 6.3 CUDA OOM

Try these (in order):

- Reduce max context: `--max-model-len 4096`
- Reduce memory utilization: `--gpu-memory-utilization 0.85`
- Use a smaller model (7B before 13B/70B)

### 6.4 CUDA Driver Compatibility Error (Error 803)

If you see: `Error 803: system has unsupported display driver / cuda driver combination`

**Step 1: Check your host driver version**
```bash
nvidia-smi
```
Look for the "Driver Version" line (e.g., `535.xx`, `550.xx`, `600.xx`).

**Step 2: Match container CUDA to your driver**

- **Driver 535.x or older**: Use CUDA 11.8 image
  ```bash
  docker run ... vllm/vllm-openai:cu118 ...
  ```

- **Driver 550.x or newer**: Use CUDA 12.1 image (default in this repo)
  ```bash
  docker run ... vllm/vllm-openai:cu121 ...
  ```

- **Driver 600.x or newer**: Try CUDA 12.4 image
  ```bash
  docker run ... vllm/vllm-openai:cu124 ...
  ```

**Step 3: Update docker-compose.yml**

If using docker-compose, change the image tag in `docker-compose.vllm.yml`:
```yaml
image: vllm/vllm-openai:cu118  # or cu121, cu124 based on your driver
```

**Step 4: Verify container can see GPU**
```bash
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```
If this fails, your NVIDIA Container Toolkit may need reinstall/restart.

---

## 7) Next steps (once Hello World works)

- Enable streaming responses
- Add a system prompt + multi-turn chat
- Try quantized weights (where appropriate)
- Deploy behind a reverse proxy (nginx/caddy) and add auth

