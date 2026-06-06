<a name="readme-top"></a>

<div align="center">
  <h1 align="center">Optical Reasoning: Rethinking Images as an Expressive Reasoning Medium Beyond Text</h1>
</div>

<div align="center">
  <a href="" style="text-decoration: none; display: inline-block; line-height: 0;">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv" alt="Paper">
  </a>
  <a href="https://huggingface.co/datasets/ModalityDance/Optical-Reasoning" style="text-decoration: none; display: inline-block; line-height: 0;">
    <img src="https://img.shields.io/badge/Dataset-Available-4c1?style=for-the-badge" alt="Dataset">
  </a>
</div>


---

Welcome to **Optical Reasoning**! 👋 This repository accompanies *"Optical Reasoning: Rethinking Images as an Expressive Reasoning Medium Beyond Text"*, a framework that treats images as a standalone reasoning medium. It supports typographic-based optical reasoning for compact rationale rendering and graphical-based optical reasoning for structured visual rationales. The repository also provides scripts for preparing visual rationales and reproducing experimental results.

<img src="./assets/intro.png" alt="vision"  style="max-width: 70%; height: auto;">

### 🪐 Key Features <span id="key-features"></span>

🖨️ **Typographic-Based Optical Reasoning** T-OR renders the interleaved-modal rationale sequence into a compact typographic image with XeLaTeX.

🎨 **Graphical-Based Optical Reasoning** G-OR transforms the interleaved-modal rationale sequence into a unified image-based rationale that organizes reasoning with text, graphical elements, and spatial layouts.

## 🔥 News <span id="news"></span>

<div style="max-height: 240px; overflow-y: auto;">

* **[2026.06]** Initial release of Optical Reasoning.

</div>

## 📑 Table of Contents <span id="table-of-contents"></span>

* [🚀 Quick Start](#quick-start)
    * [1. Installation](#installation)
        * [Create environment](#install-env)
        * [Install XeLaTeX](#install-xelatex)
        * [Model profiles](#install-profiles)
    * [2. Rationales](#rationales)
        * [Prepare Your Own Rationales](#rationales-prepare)
        * [T-OR Rationales](#rationales-tor)
        * [G-OR Rationales](#rationales-gor)
    * [3. Inference](#inference)
        * [Optical reasoning](#inference-optical)
        * [Text baselines](#inference-text)
* [✨ How It Works](#how-it-works)
* [🪐 Key Features](#key-features)
* [🔥 News](#news)
* [🗂️ Project Structure](#project-structure)
* [🌱 Acknowledgements](#acknowledgements)
* [📚 Citation](#citation)

## 🚀 Quick Start <span id="quick-start"></span>

### 1. Installation <span id="installation"></span>

#### Create environment <span id="install-env"></span>

```bash
conda create -n optical-reasoning python=3.11 -y
conda activate optical-reasoning

pip install -U pip
pip install -r requirements.txt

```

#### Install XeLaTeX <span id="install-xelatex"></span>

The renderer is implemented with XeLaTeX.

```bash
# Ubuntu / Debian
apt-get update
apt-get install -y texlive-xetex texlive-latex-extra texlive-fonts-recommended

# macOS
brew install --cask mactex-no-gui

```

Check the installation:

```bash
xelatex --version

```

#### Model profiles <span id="install-profiles"></span>

```bash
cp src/configs/profiles_example.yaml src/configs/profiles.yaml

```

```yaml
models:
  gpt5.1:
    api_key: ""
    base_url: ""
    model: "gpt-5.1-2025-11-13"
    temperature: 0.0
  llmjudge:
    api_key: ""
    base_url: ""
    model: "deepseek-chat"
    temperature: 0.0
  nano-banana-pro:
    api_key: ""
    base_url: ""
    model: "nano-banana-pro"
    temperature: 0.0

```

---

### 2. Rationales <span id="rationales"></span>

> [!TIP]
> The rationales used in the paper can be downloaded from the [Optical-Reasoning dataset](https://huggingface.co/datasets/ModalityDance/Optical-Reasoning).

#### Prepare Your Own Rationales <span id="rationales-prepare"></span>

If you want to build visual rationales from textual rationales, start from the JSONL format below.

```json
{
  "id": "sample-001",
  "problem": "Question text.",
  "solution": "Reasoning rationale.",
  "answer": "A",
  "reasoning_token": 512
}

```

The fields are:

* `id`: unique example identifier.
* `problem`: input question or problem statement.
* `solution`: textual rationale to be rendered.
* `answer`: ground-truth answer.
* `reasoning_token`: token count of the textual rationale in `solution`.

The generated T-OR and G-OR rationales follow this folder structure:

```plaintext
data/
  └── <dataset>/
      ├── <dataset>.jsonl
      ├── T-OR/
      │   ├── output.jsonl
      │   └── images/
      └── G-OR/
          ├── output.jsonl
          └── images/

```

#### T-OR Rationales <span id="rationales-tor"></span>

T-OR renders the rationale into a compact typographic image while preserving the original order of the reasoning content.

```bash
DATASET=aqua_rat \
INPUT_JSONL=data/aqua_rat/aqua_rat.jsonl \
OUTPUT_DIR=data/aqua_rat/T-OR \
OUTPUT_JSONL=data/aqua_rat/T-OR/output.jsonl \
bash scripts/render_typographic.sh

```

> [!IMPORTANT]
> For T-OR rendering, the textual rationale in the `solution` field must be LaTeX text without syntax errors.

* Reads rationales from the `solution` field.
* Searches for a compact and readable typographic layout under the `reasoning_token` budget.

#### G-OR Rationales <span id="rationales-gor"></span>

G-OR generates a structured visual rationale by composing reasoning steps into graphical panels.

```bash
DATASET=aqua_rat \
INPUT_JSONL=data/aqua_rat/aqua_rat.jsonl \
OUTPUT_BASE=data/aqua_rat/G-OR \
OUTPUT_JSONL=data/aqua_rat/G-OR/output.jsonl \
PROFILE=nano-banana-pro \
bash scripts/render_graphical.sh

```

* Uses the configured generation profile, such as `PROFILE=nano-banana-pro`.
* Converts the problem, rationale, and optional visual inputs into a step-aligned graphical rationale.

---

### 3. Inference <span id="inference"></span>

#### Optical reasoning <span id="inference-optical"></span>

For optical reasoning, T-OR takes the problem text together with the rendered typographic rationale image, while G-OR takes the problem text together with the generated graphical rationale image.

Run inference on T-OR:

```bash
PROFILE=gpt5.1 \
INPUT_JSONL=data/aqua_rat/T-OR/output.jsonl \
OUTPUT_DIR=outputs/aqua_rat/T-OR \
OUTPUT_JSONL=outputs/aqua_rat/T-OR/infer_gpt5.1.jsonl \
bash scripts/infer_typographic.sh

```

Run inference on G-OR:

```bash
PROFILE=gpt5.1 \
INPUT_JSONL=data/aqua_rat/G-OR/output.jsonl \
OUTPUT_DIR=outputs/aqua_rat/G-OR \
OUTPUT_JSONL=outputs/aqua_rat/G-OR/infer_gpt5.1.jsonl \
bash scripts/infer_graphical.sh

```

#### Text baselines <span id="inference-text"></span>

Text reasoning receives the problem followed by the rationale, and free reasoning asks the model to solve the problem step by step.

```bash
python src/run.py infer \
  --data data/<dataset>/<dataset>.jsonl \
  --output outputs/<dataset>/text_reasoning/infer_<model>.jsonl \
  --profile <model> \
  --task-type text_reasoning

```

Use `--task-type no_reasoning` or `--task-type free_reasoning` for the other text baselines.

## ✨ How It Works <span id="how-it-works"></span>

Optical Reasoning explores the bold idea of using images as a standalone reasoning medium for both language and multimodal tasks.  

- **Optical Reasoning:** formulates a unified interleaved-modal rationale sequence and maps it into an image, allowing the model to derive final answers directly from visual reasoning tokens rather than textual ones.  
- **Typographic-Based (T-OR):** optimizes visual layouts by searching over text width and font size to render rationales into compact, high-density typographic images under a strictly controllable reasoning-token budget.  
- **Graphical-Based (G-OR):** decomposes rationales into distinct reasoning steps and assigns them to specific visual panels, creating a step-aligned composition that naturally unifies textual rationales, graphical elements, and spatial layouts.  

---

## 🗂️ Project Structure <span id="project-structure"></span>

```plaintext
├── scripts/
│   ├── render_typographic.sh
│   ├── render_graphical.sh
│   ├── infer_typographic.sh
│   └── infer_graphical.sh
│
└── src/
    ├── run.py
    ├── configs/
    │   └── profiles_example.yaml
    ├── inference/
    │   ├── predictor.py
    │   └── evaluation.py
    ├── render/
    │   ├── typographic_render.py
    │   └── graphical_render.py
    └── utils/

```

## 🌱 **Acknowledgements** <span id="acknowledgements"></span>

[![XeTex](https://img.shields.io/badge/Renderer-XeTex-blue?style=flat)](https://xetex.sourceforge.net/)
[![Gsm8k](https://img.shields.io/badge/Dataset-Gsm8k-blue?style=flat&logo=huggingface)](https://huggingface.co/datasets/openai/gsm8k)
[![AquaRat](https://img.shields.io/badge/Dataset-AquaRat-blue?style=flat&logo=huggingface)](https://huggingface.co/datasets/deepmind/aqua_rat)
[![ScienceQA](https://img.shields.io/badge/Dataset-ScienceQA-blue?style=flat&logo=github)](https://github.com/lupantech/ScienceQA)
[![Zebra-CoT](https://img.shields.io/badge/Dataset-Zebra--CoT-blue?style=flat&logo=huggingface)](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)

This project is licensed under the **MIT License**. Please refer to the [LICENSE](https://www.google.com/search?q=./LICENSE) file for more details.

## 📚 **Citation** <span id="citation"></span>

```bibtex
@misc{opticalreasoning2026,
  title        = {Optical Reasoning: Rethinking Images as an Expressive Reasoning Medium Beyond Text},
  year         = {2026}
}

```

---
