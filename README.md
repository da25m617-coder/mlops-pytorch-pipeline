# MLOps PyTorch Pipeline

A production-style MLOps pipeline for training and serving a PyTorch image classification model using Docker and Kubernetes.

The project covers the complete workflow from model development and containerization to Kubernetes-based training and model serving.

## Project Overview

This project implements a PyTorch image classification pipeline using the Fashion-MNIST dataset.

The workflow consists of:

1. PyTorch model development
2. Dataset preparation and training
3. Docker containerization
4. Kubernetes training Job
5. Persistent model checkpoint storage
6. Kubernetes model serving with two replicas
7. Health checks and Service exposure
8. End-to-end validation

## Architecture

```text
                    Fashion-MNIST Dataset
                            |
                            v
                   +-------------------+
                   | PyTorch Training  |
                   |     Job           |
                   +-------------------+
                            |
                            | classifier_v1.pt
                            v
                   +-------------------+
                   | PersistentVolume  |
                   |       PVC         |
                   +-------------------+
                            |
                +-----------+-----------+
                |                       |
                v                       v
        +---------------+       +---------------+
        | Serving Pod 1 |       | Serving Pod 2 |
        |   FastAPI     |       |   FastAPI     |
        +---------------+       +---------------+
                |                       |
                +-----------+-----------+
                            |
                            v
                  +---------------------+
                  | Kubernetes Service  |
                  | ClusterIP :80       |
                  +---------------------+
                            |
                            v
                     /health
                     /predict
```

## Repository Structure

```text
mlops-pytorch-pipeline/
├── .github/
│   └── workflows/
├── configs/
│   └── training_config.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── pvc.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   └── serving-service.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── serve.py
├── tests/
├── docs/
│   ├── validation_screenshots.docx
│   └── challenges_faced.docx
├── README.md
└── .gitignore
```

# Part A – Repository Setup

The project follows a Git-based development workflow using separate branches for different features.

The repository contains separate branches for:

* Repository setup
* PyTorch model implementation
* Docker containerization
* Kubernetes deployment
* Documentation and validation

Meaningful commit messages following Conventional Commit style were used where applicable.

Sensitive and generated files such as virtual environments, Python cache files and locally generated artifacts are excluded from version control.

# Part B – PyTorch Model

The project uses a PyTorch image classifier for the Fashion-MNIST dataset.

### Model

The model is implemented in:

```text
src/model.py
```

The implemented architecture is a CNN with 10 output classes corresponding to the Fashion-MNIST categories.

### Dataset

Dataset loading and preprocessing are implemented in:

```text
src/dataset.py
```

The Fashion-MNIST dataset is downloaded using `torchvision.datasets` and processed using torchvision transforms.

### Training

The training pipeline is implemented in:

```text
src/train.py
```

Training configuration is read from:

```text
configs/training_config.yaml
```

The training process:

* Loads the configured model
* Loads Fashion-MNIST
* Performs training and validation
* Calculates loss and accuracy
* Prints metrics as JSON lines
* Saves the best checkpoint
* Supports early stopping

The generated checkpoint is:

```text
classifier_v1.pt
```

# Part C – Docker Containerization

Two Docker images are used.

## Training Image

The training image is defined in:

```text
docker/Dockerfile.train
```

Example build command:

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v2 .
```

The image executes:

```bash
python src/train.py
```

## Serving Image

The serving image is defined in:

```text
docker/Dockerfile.serve
```

It contains only the dependencies required for inference and runs the FastAPI application as a non-root user.

Example:

```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v2 .
```

The serving application listens on:

```text
0.0.0.0:8080
```

Endpoints:

```text
GET  /health
POST /predict
```

# Part D – Kubernetes Training Job

The Kubernetes training pipeline uses the `ml-training` namespace.

## Namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

Namespace:

```text
ml-training
```

## ConfigMap

The training configuration is stored in:

```text
k8s/configmap.yaml
```

and mounted into the training container at:

```text
/app/configs
```

The configuration includes:

* Model architecture
* Number of classes
* Epochs
* Batch size
* Learning rate
* Early stopping patience
* Dataset location
* Checkpoint location

## Persistent Storage

The checkpoint PVC is:

```text
model-checkpoint-pvc
```

It is used for persistent storage of:

```text
/app/data
/app/checkpoints
```

The training Job uses the PVC with separate subpaths for data and checkpoints.

## Training Job

The Job is defined in:

```text
k8s/training-job.yaml
```

Apply it using:

```bash
kubectl apply -f k8s/training-job.yaml
```

Check the Job:

```bash
kubectl get jobs -n ml-training
```

Check the training pod:

```bash
kubectl get pods -n ml-training
```

View training logs:

```bash
kubectl logs job/mlops-training-job -n ml-training
```

The successful training output includes:

```text
checkpoint_saved
training_complete
```

The trained model checkpoint is stored on the PVC as:

```text
classifier_v1.pt
```

# Part E – Kubernetes Model Serving

The trained model is served using a FastAPI application running in Kubernetes.

## Deployment

The serving Deployment is defined in:

```text
k8s/serving-deployment.yaml
```

Deployment name:

```text
mlops-serving-deployment
```

The Deployment runs two replicas:

```yaml
replicas: 2
```

Both replicas use:

```text
mlops-serve:v2
```

The checkpoint PVC is mounted read-only at:

```text
/app/checkpoints
```

The Deployment uses a RollingUpdate strategy:

```yaml
maxSurge: 1
maxUnavailable: 0
```

### Resource Configuration

Requests:

```text
CPU:    500m
Memory: 1Gi
```

Limits:

```text
CPU:    1
Memory: 2Gi
```

### Health Checks

Liveness probe:

```text
GET /health
period: 10 seconds
failure threshold: 3
```

Readiness probe:

```text
GET /health
initial delay: 15 seconds
period: 5 seconds
```

Verify the Deployment:

```bash
kubectl get deployment mlops-serving-deployment -n ml-training
```

Verify the two replicas:

```bash
kubectl get pods -n ml-training -l app=mlops-serving
```

## Kubernetes Service

The Service is defined in:

```text
k8s/serving-service.yaml
```

Service name:

```text
mlops-serving-service
```

Service type:

```text
ClusterIP
```

Port mapping:

```text
Service port: 80
Container port: 8080
```

Verify:

```bash
kubectl get svc mlops-serving-service -n ml-training
```

# Part F – End-to-End Validation

The complete Kubernetes workflow was validated on the local Kubernetes cluster.

## Apply Kubernetes Resources

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
```

## Verify Training

```bash
kubectl get pods -n ml-training
```

The training Job completed successfully.

Training logs were verified using:

```bash
kubectl logs job/mlops-training-job -n ml-training
```

The logs showed the checkpoint being saved and training completing successfully.

## Verify Checkpoint

The trained checkpoint was verified inside a serving pod:

```bash
kubectl exec -it <serving-pod> -n ml-training -- \
ls -lh /app/checkpoints
```

The following file was present:

```text
classifier_v1.pt
```

## Verify Serving Replicas

```bash
kubectl get pods -n ml-training -l app=mlops-serving
```

Two serving replicas reached:

```text
1/1 Running
1/1 Running
```

## Port Forward

The Kubernetes Service was exposed locally using:

```bash
kubectl port-forward svc/mlops-serving-service 8081:80 -n ml-training
```

Port `8081` was used because local port `8080` was already occupied.

## Health Endpoint

The health endpoint was tested using:

```bash
curl http://localhost:8081/health
```

Successful response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

## Prediction Endpoint

The prediction API was tested using:

```bash
curl -X POST http://localhost:8081/predict \
  -F "image=@test_image.png"
```

The API returns:

* Predicted Fashion-MNIST class
* Prediction confidence
* Probability for each class

## Validation Evidence

Screenshots and terminal outputs for the Kubernetes training and serving workflow are available in:

```text
docs/validation_screenshots.docx
```

The validation evidence covers:

* Kubernetes namespace and configuration
* Training Job
* Training logs
* Checkpoint creation
* Serving Deployment
* Two serving replicas
* Kubernetes Service
* Health endpoint
* Prediction endpoint

# Conclusion

The project demonstrates an end-to-end PyTorch MLOps workflow using Docker and Kubernetes.

The training workload runs as a Kubernetes Job and stores the resulting model checkpoint on persistent storage. The trained model is then loaded by two FastAPI serving replicas. Kubernetes health probes monitor the serving containers, while a ClusterIP Service provides access to the inference API.

This setup demonstrates separation between training and serving workloads and provides a foundation for scalable model deployment.
