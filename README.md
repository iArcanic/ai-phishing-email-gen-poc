# ai-phishing-email-gen-poc

![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Proof of concept for generating synthetic phishing emails using a fine-tuned GPT-2 model.

> [!WARNING]
>
> **This project is intended for educational purposes only.**
>
> **Misuse of this code is to generate phishing emails or any other malicious activities is strictly prohibited.**
>
> iArcanic not liable for any misuse or damages resulting from the use of this code. By using this code, you agree to use it responsibly and ethically.

## 1 TL;DR / Quickstart

Via Docker Compose:

```bash
docker-compose up --build
```

Via Docker CLI:

```bash
docker build -t ai-phishing-email-gen .
docker run ai-phishing-email-gen
```

## 2 Prerequisites

### 2.1 Docker

Ensure the Docker engine is installed on your system with version **18.06.0** or higher.

You can download and install the Docker engine from the [official Docker website](https://www.docker.com/get-started/).

> [!NOTE]
>
> - Especially on Linux, make sure your user has the [required permissions](https://docs.docker.com/engine/install/linux-postinstall/) to interact with the Docker daemon.
> - If you are unable to do this, either append `sudo` in front of each `docker` command or switch to root using `sudo -s`.

### 2.2 Docker Compose

Ensure that Docker Compose is installed on your system with **version 1.28.0** or higher.

You can download and install Docker Compose from the [official Docker website](https://docs.docker.com/compose/install/).

### 2.3 Hardware

#### 2.3.1 Minimum requirements

- **CPU**: Multi-core processor (4 cores or more)
- **RAM**: At least 16GB of RAM
- **Disk space**: Minimum of 50GB of free disk space
- **Graphics**: Integrated graphics (for non-GPU training)

#### 2.3.2 Recommended requirements

- **CPU**: High-performance multi-core processor (8 cores or more)
- **RAM**: 32GB or more of RAM
- **Disk space**: Minimum if 100GB of free disk space
- **GPU**: NVIDIA GPU with CUDA support (e.g., NVIDIA RTX 2080 or better) for faster training
- **Graphics**: Dedicated GPU for significant performance improvement

> [!NOTE]
>
> Training on a CPU will be significantly slower than on a GPU. It is highly recommended to use a machine with a compatible NVIDIA GPU for training large models.
>
> Ensure sufficient disk space to store the model checkpoints and datasets.
>
> If running in a virtualized environment or container, allocate resources according to the above recommendations to ensure optimal performance.

## 3 Usage

1. Clone the repository to your local machine.

```bash
git clone https://github.com/iArcanic/ai-phishing-email-gen-poc
```

2. Navigate to the project's root directory.

```bash
cd ai-phishing-email-gen-poc
```

3. Build and run the Docker container.

```bash
docker-compose up --build
```

> [!NOTE]
> With Docker Compose, you can also optionally use the following:
>
> - If you want to build the images each time (or changed a Dockerfile), use:
>
> ```bash
> docker-compose up --build
> ```
>
> - If you want to run all the services in the background, use:
>
> ```bash
> docker-compose up -d
> ```
>
> After, you can optionally view Docker images, status of containers, and interact with running containers using the Docker Desktop application.

4. View the Docker Container's logs

The following logs should be displayed upon running the container:

```plain
Loading the data...
Preprocessing the data...
Splitting datasets...
Initialising GPT2 Model...
GPT2 model initialised.
Tokenizing the train dataset...
Map: 100%|██████████| 4323/4323 [00:02<00:00, 2010.93 examples/s]
Train dataset tokenized.
Tokenizing the evaluation dataset...
Map: 100%|██████████| 1081/1081 [00:00<00:00, 2470.80 examples/s]
Evaluation dataset tokenized.
Setting up training arguments...
Training arguments set.
Initialising the trainer...
Trainer initialised.
Training the GPT2 model...
  0%|          | 10/3243 [00:47<4:06:58,  4.58s/it]
{'loss': 8.376, 'grad_norm': 196.69766235351562, 'learning_rate': 5.000000000000001e-07, 'epoch': 0.0}
```

> [!NOTE]
>
> The training of the GPT-2 model can take a **very** long time without a GPU.
>
> Testing this on a Ubuntu Server 16-core machine with 32 GB memory (no GPU) resulted in a time of **2 days**.

Upon successful model training, the following logs should be displayed:

```plain
100%|██████████| 3243/3243 [32:26:09<00:00, 36.01s/it] ning_rate': 5.468465184104995e-08, 'epoch': 3.0}
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
{'eval_loss': 0.7192820906639099, 'eval_runtime': 313.6607, 'eval_samples_per_second': 3.446, 'eval_steps_per_second': 0.864, 'epoch': 3.0}
{'train_runtime': 116769.8894, 'train_samples_per_second': 0.111, 'train_steps_per_second': 0.028, 'train_loss': 0.7957250362849698, 'epoch': 3.0}
Training completed successfully.
Saving the final model and tokenizer...
Model and tokenizer saved successfully.
Loading the trained model and tokenizer...
Model and tokenizer loaded successfully.
Generating text with prompt: 'Dear customer, we have noticed unusual activity in your bank account.'...
Text generation completed successfully.
Generated text 1:
Dear customer, we have noticed unusual activity in your bank account. we are trying to contact you.
please verify your account and regain full access to your å£350 weekly sae. call 0871871850 now!
```

> [!NOTE]
>
> The prompt can be customised via the `prompt` variable in [src/main.py](https://github.com/iArcanic/ai-phishing-email-gen-poc/blob/main/src/main.py).

5. Destroy the Docker Container after use

```bash
docker-compose down
```

## 4 Acknowledgements

- Datasets in the [`data`](https://github.com/iArcanic/ai-phishing-email-gen-poc/tree/main/data) folder are taken from [TanusreeSharma/phishingdata-Analysis](https://github.com/TanusreeSharma/phishingdata-Analysis).
- GPT-2 model from [Hugging Face](https://huggingface.co/openai-community/gpt2).
