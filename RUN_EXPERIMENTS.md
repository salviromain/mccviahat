# Running the Full-Trace Experiments

After reserving a node and connecting via SSH:

## 1. Set up the node and model

```bash
git clone https://github.com/salviromain/mccviahat
cd mccviahat
git switch newfeatures
bash scripts/node/bootstrap_node.sh
newgrp docker
export HF_TOKEN="YOUR_HF_TOKEN"
bash scripts/model/model_fetch.sh 70b
bash scripts/model/llama_build.sh
```

Replace `YOUR_HF_TOKEN` with your Hugging Face access token. The runner starts the model server automatically.

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

Replace `232` with your node ID. Use lowercase condition labels and a unique run label for every recording, such as `232_E_2` for another emotional run.

## 3. Find the recorded data

Each command records all prompts in the selected JSON as one continuous trace. Raw files are saved in:

```text
data/raw/fulltrace/incoming/<run_label>/
```

To choose a different parent folder, add `--output-root data/raw/fulltrace/<new_batch>` to the run command.
