# Running the Full-Trace Experiments

After reserving a node and connecting via SSH:

## 1. Set up the node and model

```bash
git clone https://github.com/salviromain/mccviahat
cd mccviahat
git switch main
bash scripts/node/bootstrap_node.sh
newgrp docker
export HF_TOKEN="YOUR_HF_TOKEN"
bash scripts/model/model_fetch.sh 70b
bash scripts/model/llama_build.sh
```

Replace `YOUR_HF_TOKEN` with your Hugging Face access token. The runner starts the model server automatically.

### Use Llama 3.1 405B instead

Use the updated model scripts and a large-memory node. The
[Q4_K_M model](https://huggingface.co/bullerwins/Meta-Llama-3.1-405B-Instruct-GGUF/tree/main)
has six shards totaling about 246 GB. For this CPU setup, budget 384 GB RAM and
at least 300 GB free disk; these are planning recommendations, not measured
requirements from a completed 405B experiment.

```bash
free -h
df -h "$HOME"
bash scripts/model/model_fetch.sh 405b
bash scripts/model/llama_build.sh
```

The fetch script downloads all six shards and configures the first shard for
llama.cpp, which loads the others automatically. It keeps the same context size,
CPU affinity, baseline, sampling interval, and generated-token count as 70B;
startup and request timeouts are increased. The CPU affinity still assumes CPUs
`0-61` are available. Set `MCC_MODEL_DIR` before fetching if the weights must go
on a different disk.

Run 405B with its own output directory and unique run labels:

```bash
python3 scripts/run/run_prompts_fulltrace.py \
  --json prompts/20base/independentE.json \
  --label emotional --run_label 232_E_1 \
  --output-root data/raw/fulltrace/405b
```

Reboot between experiments as described below, SSH back in, and run:

```bash
cd ~/mccviahat
python3 scripts/run/run_prompts_fulltrace.py \
  --json prompts/20base/independentN.json \
  --label neutral --run_label 232_N_1 \
  --output-root data/raw/fulltrace/405b
```

Replace `232` with your node ID. The saved metadata records `model_size: 405b`.
Existing 70B data is kept separate. To switch back, run
`bash scripts/model/model_fetch.sh 70b` before starting the next experiment.

## 2. Run an experiment

**Emotional:**

```bash
python3 scripts/run/run_prompts_fulltrace.py \
  --json prompts/20base/independentE.json \
  --label emotional --run_label 232_E
```

**Neutral:**

```bash
python3 scripts/run/run_prompts_fulltrace.py \
  --json prompts/20base/independentN.json \
  --label neutral --run_label 232_N
```

**Replace `232` with your node ID. Use lowercase condition labels and a unique run label for every recording, such as `232_E_2` for another emotional run.**

## 3. Reboot between experiments on the same node

**After each experiment finishes, reboot the node before starting the next experiment on that same node:**

```bash
sudo reboot
```

This disconnects your SSH session. Wait for the node to come back online, then SSH back in from your own device:

```bash
ssh YOUR_CLOUDLAB_USERNAME@clnode232.clemson.cloudlab.us
```

Replace the username and hostname with your CloudLab details. Return to the repository and run the next experiment, for example:

```bash
cd ~/mccviahat
python3 scripts/run/run_prompts_fulltrace.py \
  --json prompts/20base/independentN.json \
  --label neutral --run_label 232_N
```

Choose the prompt file, condition, and unique run label for the next experiment. You do not need to clone the repository, download the model, or rebuild the Docker image again.

## 4. Find the recorded data

Each command records all prompts in the selected JSON as one continuous trace. Raw files are saved in:

```text
data/raw/fulltrace/incoming/<run_label>/
```

To choose a different parent folder, add `--output-root data/raw/fulltrace/<new_batch>` to the run command.

## 5. Pull the data onto your own device

Run this command from a terminal on your own device, outside the node's SSH session:

```bash
rsync -avP "rsalvi@clnode184.clemson.cloudlab.us:~/mccviahat/data/raw/fulltrace/incoming/" \
  "$HOME/Desktop/mccviahat/data/raw/fulltrace/"
```

Change:

- `rsalvi` to your CloudLab username.
- `clnode184.clemson.cloudlab.us` to your reserved node's hostname.
- "$HOME/Desktop/mccviahat/data/raw/fulltrace/" to the destination folder on your own device.

Keep the trailing `/` on the source directory to copy its contents into the destination folder.
