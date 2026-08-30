# MLOps PyTorch Pipeline

A containerized PyTorch image-classification pipeline demonstrating the deployment lifecycle from model development to Docker-based training and Kubernetes-based training orchestration.

This project was developed as part of the **MLOps & Infrastructure for Machine Learning** assignment.

## Project Overview

The project implements an image classification workload using **Fashion-MNIST** and PyTorch.

The pipeline currently covers:

* **Part A:** Repository setup and Git workflow
* **Part B:** PyTorch CNN model and training pipeline
* **Part C:** Docker containerization
* **Part D:** Kubernetes training Job

The Kubernetes training workload is deployed using a local **kind** cluster.

### Current Architecture

```text
                    ┌─────────────────────┐
                    │   Fashion-MNIST     │
                    │      Dataset        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   PyTorch CNN       │
                    │   Training Script   │
                    └──────────┬──────────┘
                               │
                         Docker Image
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Kubernetes Job    │
                    │   mlops-training    │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
       ConfigMap                       PersistentVolumeClaim
    training_config.yaml                  /app/data
                                            /app/checkpoints
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                    classifier_v1.pt
```

---

# Project Structure

```text
mlops-pytorch-pipeline/
│
├── README.md
├── .gitignore
│
├── .github/
│   └── workflows/
│
├── src/
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   └── serve.py
│
├── configs/
│   └── training_config.yaml
│
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
│
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── pvc.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
│
├── requirements/
│   ├── train.txt
│   └── serve.txt
│
└── tests/
    └── test_model.py
```

---

# Part A — Repository Setup

The project follows a Git-based development workflow with separate branches for individual features.

The main development branch is:

```text
main
```

Feature work is performed on dedicated branches, for example:

```text
k8s-deployment
part-e-model-serving
```

The Kubernetes training implementation was developed separately and then merged into the main branch.

Git commits use descriptive messages such as:

```text
Implement Part D Kubernetes training pipeline
```

A `.gitignore` file is used to prevent generated files and local development artifacts such as datasets, checkpoints, virtual environments and Python cache files from being committed.

---

# Part B — PyTorch Model

## Dataset

The project uses the **Fashion-MNIST** dataset.

Fashion-MNIST contains grayscale images of size:

```text
28 × 28
```

with:

```text
10 classes
```

The dataset is downloaded automatically using `torchvision.datasets.FashionMNIST`.

Training data uses:

* Random horizontal flip
* Conversion to tensor
* Normalization

Validation data uses:

* Conversion to tensor
* Normalization

The Fashion-MNIST normalization values used are:

```text
mean = 0.2860
std  = 0.3530
```

## CNN Model

The final model used for the Kubernetes training workload is a lightweight CNN instead of ResNet-18.

The model accepts:

```text
1 × 28 × 28
```

grayscale Fashion-MNIST images and produces predictions for:

```text
10 classes
```

The architecture is selected through the configuration file:

```yaml
model:
  architecture: "cnn"
  num_classes: 10
```

This makes the model architecture configurable without changing the training script.

## Training

The training implementation is located in:

```text
src/train.py
```

The training script:

1. Loads the YAML configuration.
2. Selects CPU or CUDA automatically.
3. Creates the CNN model.
4. Loads the Fashion-MNIST dataset.
5. Creates the optimizer and loss function.
6. Trains the model.
7. Evaluates it on the validation dataset.
8. Prints metrics as JSON lines.
9. Saves the best checkpoint.
10. Supports early stopping.

Example training output:

```json
{"epoch": 1, "train_loss": 0.4497, "train_accuracy": 0.839, "val_loss": 0.3399, "val_accuracy": 0.8766}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"event": "training_complete", "best_val_loss": 0.3399}
```

---

# Part C — Docker Containerization

## Training Image

The training workload is packaged using:

```text
docker/Dockerfile.train
```

The Docker image contains the Python runtime, PyTorch dependencies and project source code.

Build the training image using:

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v2 .
```

The resulting image was loaded into the kind Kubernetes cluster using:

```bash
kind load docker-image mlops-train:v2 --name ml-training
```

The image was then used by the Kubernetes training Job.

## Serving Image

A separate serving Dockerfile is provided:

```text
docker/Dockerfile.serve
```

The serving image uses a slim Python base image and contains only inference-related dependencies.

The serving application is implemented using FastAPI and exposes:

```text
GET  /health
POST /predict
```

The serving implementation is prepared for the later Kubernetes model-serving stage.

---

# Part D — Kubernetes Training Job

The Kubernetes training workload is deployed as a `Job`.

The Kubernetes resources are:

```text
k8s/namespace.yaml
k8s/configmap.yaml
k8s/pvc.yaml
k8s/training-job.yaml
```

## Kubernetes Cluster

The workload was tested using a local **kind** cluster.

The active Kubernetes context was:

```bash
kubectl config current-context
```

which returned:

```text
kind-ml-training
```

---

## Namespace

The project uses a dedicated namespace:

```text
ml-training
```

This keeps the project's Kubernetes resources isolated from other workloads.

Apply the namespace:

```bash
kubectl apply -f k8s/namespace.yaml
```

---

## ConfigMap

Training parameters are stored in:

```text
k8s/configmap.yaml
```

The configuration contains:

```yaml
model:
  architecture: cnn
  num_classes: 10

training:
  epochs: 1
  batch_size: 64
  learning_rate: 0.001
  early_stopping_patience: 3

data:
  dataset: fashion_mnist
  data_dir: /app/data

output:
  checkpoint_dir: /app/checkpoints
  model_name: classifier_v1.pt
```

The ConfigMap is mounted inside the training container at:

```text
/app/configs
```

The training script reads:

```text
/app/configs/training_config.yaml
```

This separates configuration from the Docker image.

---

## Persistent Storage

A PersistentVolumeClaim is used to provide persistent storage for:

```text
/app/data
/app/checkpoints
```

This is important because the trained model checkpoint should survive the lifecycle of the training container.

The trained model is saved as:

```text
/app/checkpoints/classifier_v1.pt
```

---

## Kubernetes Training Job

The Job manifest is:

```text
k8s/training-job.yaml
```

The container uses:

```yaml
image: mlops-train:v2
```

The Job requests:

```yaml
requests:
  cpu: "2"
  memory: "4Gi"
```

and limits:

```yaml
limits:
  cpu: "2"
  memory: "4Gi"
```

The ConfigMap is mounted at:

```text
/app/configs
```

and the PVC is mounted for:

```text
/app/data
/app/checkpoints
```

---

# Deploying the Training Workload

Apply the Kubernetes resources:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/training-job.yaml
```

Check the Job:

```bash
kubectl get jobs -n ml-training
```

Check the Pod:

```bash
kubectl get pods -n ml-training
```

Example:

```text
NAME                       READY   STATUS    RESTARTS   AGE
mlops-training-job-zvzqp   0/1     Completed 0          4m
```

---

# Verifying the Configuration

The mounted configuration can be checked using:

```bash
kubectl exec -it <pod-name> -n ml-training -- \
cat /app/configs/training_config.yaml
```

The deployed image can be verified using:

```bash
kubectl get pod <pod-name> -n ml-training \
-o jsonpath="{.spec.containers[0].image}"
```

Expected:

```text
mlops-train:v2
```

The training process can be verified using:

```bash
kubectl exec -it <pod-name> -n ml-training -- \
sh -c "cat /proc/1/cmdline | tr '\0' ' '; echo"
```

Expected:

```text
python src/train.py
```

---

# Training Validation

Training logs can be viewed using:

```bash
kubectl logs <pod-name> -n ml-training
```

Successful execution produced:

```json
{"epoch": 1, "train_loss": 0.4497, "train_accuracy": 0.839, "val_loss": 0.3399, "val_accuracy": 0.8766}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"event": "training_complete", "best_val_loss": 0.3399}
```

The trained checkpoint can be verified using:

```bash
kubectl exec -it <pod-name> -n ml-training -- \
ls -lh /app/checkpoints
```

---
## Part D Validation

Kubernetes training was successfully validated using:
- Kubernetes Pod status
- Kubernetes Job completion
- Training logs
- ConfigMap verification
- Model checkpoint verification

Validation screenshots and challenges faced are available in the `docs/` directory.

# Technologies Used

| Technology            | Purpose                               |
| --------------------- | ------------------------------------- |
| Python                | Programming language                  |
| PyTorch               | Deep learning framework               |
| torchvision           | Dataset and image transformations     |
| Fashion-MNIST         | Image classification dataset          |
| Docker                | Containerization                      |
| Kubernetes            | Workload orchestration                |
| kind                  | Local Kubernetes cluster              |
| kubectl               | Kubernetes management                 |
| ConfigMap             | External training configuration       |
| PersistentVolumeClaim | Persistent dataset/checkpoint storage |
| FastAPI               | Model-serving API                     |
| Git/GitHub            | Version control                       |

---


The current README documents the implementation through **Part D**. Part E and Part F will be documented after the serving and end-to-end validation stages are completed.
