# Deploy the Backend on a Dedicated RTX 3060 PC

This profile runs the current direct-chat product on an Ubuntu 24.04 LTS PC.
FastAPI and Ollama listen on localhost only. The existing named Cloudflare
Tunnel publishes `https://api.mywsite.online` without opening either local API
port to the Internet.

## Persistent server lifecycle

Do not run `IELTS_Chatbot_BE.ipynb` on this machine. That notebook is only for
ephemeral Colab runtimes. A dedicated server installs each component once:

- Ollama and the IELTS model stay under persistent system storage.
- The Git repository stays at `/opt/ielts-chatbot`.
- Python packages stay inside `/opt/ielts-chatbot/backend/.venv`.
- The API token stays in the protected `backend/.env` file.
- The Cloudflare connector and token stay in the operating-system service.

After the one-time setup, a normal reboot downloads nothing. `systemd` starts
Ollama, FastAPI and Cloudflare automatically. Ollama reads the existing model
from disk and loads it into VRAM on warmup or the first chat request.

Only run download/update commands deliberately:

- `git pull` when deploying a new backend commit.
- `pip install` when Python requirements change.
- `ollama pull` when installing or changing the LLM model.
- Cloudflared/Ollama installers when intentionally upgrading those programs.

## 1. Recommended host setup

- Install Ubuntu 24.04 LTS directly on the machine. Avoid WSL or a desktop
  Docker GPU stack for this first production deployment.
- Prefer at least 16 GB system RAM; 32 GB is safer. The RTX 3060 provides
  12 GB VRAM.
- Use wired Ethernet, disable automatic sleep/hibernate and configure the BIOS
  to power on after AC power is restored.
- The host needs outbound HTTPS and SSH access. Do not expose ports 8765 or
  11434 on the router or firewall.

Check the machine before installation:

```bash
cat /etc/os-release
uname -m
nvidia-smi
free -h
df -h /
```

If `nvidia-smi` is unavailable, install the Ubuntu-recommended compute driver:

```bash
sudo apt update
sudo apt upgrade -y
sudo ubuntu-drivers install --gpgpu
sudo reboot
```

After reboot, `nvidia-smi` must list the RTX 3060 without errors.

Disable host sleep only after confirming this PC is dedicated to the service:

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

This can be reverted with the same command using `unmask`.

## 2. Install and tune Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
ollama pull hf.co/Zkare/Chatbot_Ielts_Assistant_v2:Q4_K_M
```

Verify Ollama:

```bash
systemctl status ollama --no-pager
curl -s http://127.0.0.1:11434/api/tags
ollama ps
nvidia-smi
```

## 3. Install the direct-chat backend

Create a service account and clone the repository:

```bash
sudo apt install -y git curl python3-venv
sudo useradd --system --create-home --shell /usr/sbin/nologin ielts-chatbot
sudo git clone https://github.com/phamtu2x5/ielts-chatbot.git /opt/ielts-chatbot
sudo chown -R ielts-chatbot:ielts-chatbot /opt/ielts-chatbot
```

Install the dedicated-GPU Ollama override from the repository:

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo cp /opt/ielts-chatbot/deploy/ollama.override.conf \
  /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

The override keeps the single production model loaded, uses one inference
request at a time to protect 12 GB VRAM, enables Flash Attention and uses a
q8 K/V cache. FastAPI may accept two chat requests; Ollama queues the second
instead of allocating a second 4K context concurrently.

Create the Python environment with only direct-chat dependencies:

```bash
sudo -u ielts-chatbot python3 -m venv /opt/ielts-chatbot/backend/.venv
sudo -u ielts-chatbot /opt/ielts-chatbot/backend/.venv/bin/pip install --upgrade pip
sudo -u ielts-chatbot /opt/ielts-chatbot/backend/.venv/bin/pip install --no-cache-dir \
  -r /opt/ielts-chatbot/backend/requirements-direct.txt
sudo -u ielts-chatbot cp /opt/ielts-chatbot/backend/.env.server.example \
  /opt/ielts-chatbot/backend/.env
sudo chmod 600 /opt/ielts-chatbot/backend/.env
```

Edit `/opt/ielts-chatbot/backend/.env` and replace `API_AUTH_TOKEN` with the
same secret used by the trusted frontend proxy. Do not commit this file.

Install and start FastAPI:

```bash
sudo cp /opt/ielts-chatbot/deploy/ielts-chatbot.service \
  /etc/systemd/system/ielts-chatbot.service
sudo systemctl daemon-reload
sudo systemctl enable --now ielts-chatbot
sudo systemctl status ielts-chatbot --no-pager
```

Verify locally:

```bash
curl -s http://127.0.0.1:8765/health
curl -s -X POST http://127.0.0.1:8765/warmup \
  -H "Authorization: Bearer YOUR_IELTS_API_TOKEN"
ollama ps
nvidia-smi
```

`ollama ps` should show the model on the GPU. The warmup result should mark the
LLM ready and embedding, OCR and layout as skipped.

## 4. Move the existing Cloudflare Tunnel

In Cloudflare Zero Trust, open the existing `ielts-chatbot-colab` tunnel and
copy its Linux installation command. Install `cloudflared` and the existing
tunnel token on this PC as instructed by the dashboard. Keep the published
application route unchanged:

```text
api.mywsite.online -> http://localhost:8765
```

The connector should run as `cloudflared.service`:

```bash
sudo systemctl enable --now cloudflared
sudo systemctl status cloudflared --no-pager
curl -s https://api.mywsite.online/health
```

Run a frontend chat through the public API before stopping Colab. Do not leave
both connectors active permanently because Cloudflare may send traffic to
either origin.

## 5. Reboot and operations check

Reboot once before considering the migration complete:

```bash
sudo reboot
```

After reboot:

```bash
systemctl is-active ollama ielts-chatbot cloudflared
curl -s http://127.0.0.1:8765/health
curl -s https://api.mywsite.online/health
ollama ps
nvidia-smi
```

Useful logs:

```bash
journalctl -u ollama -f
journalctl -u ielts-chatbot -f
journalctl -u cloudflared -f
```

To update the backend later:

```bash
sudo -u ielts-chatbot git -C /opt/ielts-chatbot pull --ff-only
sudo systemctl restart ielts-chatbot
```

Run `pip install --no-cache-dir -r requirements-direct.txt` during an update
only when a pulled commit changes a requirements file. A routine restart or
server reboot must not reinstall packages or pull the model again.

## 6. Optional document stack

The production profile does not install or load BGE-M3, RapidOCR,
DocLayout-YOLO, PyTorch, PyMuPDF or OpenCV. Their code remains available for a
future upload release. To restore it, install
`backend/requirements-documents.txt`, enable the document environment settings
and test GPU memory before exposing uploads again.
