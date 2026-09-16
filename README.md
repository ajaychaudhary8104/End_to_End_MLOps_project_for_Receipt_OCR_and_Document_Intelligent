# End_to_End_MLOps_project_for_Receipt_OCR_and_Document_Intelligent

set MLFLOW_TRACKING_URI=https://dagshub.com/ajaychaudhary8104/End_to_End_MLOps_project_for_Receipt_OCR_and_Document_Intelligent.mlflow
set MLFLOW_TRACKING_USERNAME=ajaychaudhary8104
set MLFLOW_TRACKING_PASSWORD=gangapur8955

# Configuration Workflow

To modify the pipeline:

1. Update `config/config.yaml` - Set paths and parameters
2. Update `params.yaml` - Modify hyperparameters
3. Update `entity/config_entity.py` - Define configuration entities
4. Update `config/configuration.py` - Implement configuration manager
5. Update components in `components/` - Modify pipeline stages
6. Update pipeline stages in `pipeline/` - Update pipeline logic
7. Update `main.py` - Execute the pipeline
8. Update `dvc.yaml` - Define DVC pipeline stages


## Build Docker Image

```bash
docker build -t receipt-ocr-api:latest .
```

## Run Docker Container

```bash
docker run -p 8000:8000 receipt-ocr-api:latest
```

# AWS CI/CD Deployment with GitHub Actions

## Step 1: Login to AWS Console

Go to:

[AWS Console](https://aws.amazon.com/console/?utm_source=chatgpt.com)

---

# Step 2: Create IAM User for Deployment

Go to:

* IAM → Users → Create User

## Required Permissions

Attach these policies:

* `AmazonEC2FullAccess`
* `AmazonEC2ContainerRegistryFullAccess`
* `AmazonS3FullAccess`
AmazonEC2FullAccess
AmazonEC2ContainerRegistryFullAccess
AmazonEKSFullAccess
AmazonS3FullAccess

## Purpose of These Permissions

### EC2 Access

Used to manage virtual machines.

### ECR Access

Used to store Docker images in AWS Elastic Container Registry.

---

# CI/CD Deployment Flow

1. Build Docker image from source code
2. Push Docker image to Amazon ECR
3. Launch EC2 instance
4. Pull Docker image from ECR inside EC2
5. Run Docker container on EC2

---

# Step 3: Create ECR Repository

Go to:

* Elastic Container Registry (ECR)
* Create Repository

Example Repository URI:

```bash
577124149610.dkr.ecr.us-east-1.amazonaws.com/receipt-ocr
```

Save this URI for GitHub Secrets.

---

# Step 4: Launch EC2 Instance

Recommended Configuration:

* Ubuntu Server 22.04
* t2.medium or higher
* Minimum 20GB storage

Allow These Inbound Rules:

| Type       | Port |
| ---------- | ---- |
| SSH        | 22   |
| HTTP       | 80   |
| HTTPS      | 443  |
| Custom TCP | 8000 |

---

# Step 5: Install Docker on EC2

Connect to EC2:

```bash
ssh -i key.pem ubuntu@<EC2_PUBLIC_IP>
```

Run:

```bash
sudo apt-get update -y

sudo apt-get upgrade -y
```

Install Docker:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh
```

Add Ubuntu user to Docker group:

```bash
sudo usermod -aG docker ubuntu
```

Activate group changes:

```bash
newgrp docker
```

Verify Docker:

```bash
docker --version
```

---

# Step 6: Configure EC2 as GitHub Self-Hosted Runner

Go to your GitHub repository:

```text
Settings → Actions → Runners → New Self-hosted Runner
```

Choose:

* Linux
* x64

Run all commands provided by GitHub one-by-one on EC2.

Example:

```bash
mkdir actions-runner && cd actions-runner

curl -o actions-runner-linux-x64.tar.gz -L https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-linux-x64-2.317.0.tar.gz

tar xzf ./actions-runner-linux-x64.tar.gz
```

Configure runner:

```bash
./config.sh --url https://github.com/<username>/<repo> --token <TOKEN>
```

Start runner:

```bash
./run.sh
```

For background service:

```bash
sudo ./svc.sh install

sudo ./svc.sh start
```

---

# Step 7: Configure GitHub Secrets

Go to:

```text
Repository → Settings → Secrets and variables → Actions
```

Add:

```bash
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_DEFAULT_REGION=us-east-1

AWS_ECR_LOGIN_URI=

ECR_REPOSITORY_NAME=
```

---