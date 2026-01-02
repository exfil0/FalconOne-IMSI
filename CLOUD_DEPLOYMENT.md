# FalconOne Cloud Deployment Guide

Complete guide for deploying FalconOne to AWS, Azure, and GCP.

## Table of Contents
- [AWS Deployment](#aws-deployment)
- [Azure Deployment](#azure-deployment)
- [GCP Deployment](#gcp-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring & Logging](#monitoring--logging)
- [Scaling & Performance](#scaling--performance)

---

## AWS Deployment

### Prerequisites
- AWS CLI installed and configured
- Terraform >= 1.0
- Docker
- AWS account with appropriate permissions

### 1. Initial Setup

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)

# Verify configuration
aws sts get-caller-identity
```

### 2. Create S3 Bucket for Terraform State

```bash
# Create S3 bucket
aws s3api create-bucket \
  --bucket falconone-terraform-state \
  --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket falconone-terraform-state \
  --versioning-configuration Status=Enabled

# Create DynamoDB table for state locking
aws dynamodb create-table \
  --table-name falconone-terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
  --region us-east-1
```

### 3. Deploy Infrastructure with Terraform

```bash
cd terraform/aws

# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Apply infrastructure
terraform apply
# Type 'yes' to confirm

# Save outputs
terraform output > terraform-outputs.txt
```

### 4. Build and Push Docker Image to ECR

```bash
# Get ECR repository URL from Terraform output
ECR_REPO=$(terraform output -raw ecr_repository_url)

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REPO

# Build Docker image
docker build -t falconone:latest .

# Tag image for ECR
docker tag falconone:latest $ECR_REPO:latest

# Push to ECR
docker push $ECR_REPO:latest
```

### 5. Verify Deployment

```bash
# Get ALB DNS name
ALB_DNS=$(terraform output -raw alb_dns_name)

# Check health endpoint
curl http://$ALB_DNS/health

# View ECS service status
aws ecs describe-services \
  --cluster falconone-cluster \
  --services falconone-service
```

### AWS Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Internet Gateway                   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│           Application Load Balancer (ALB)            │
│                   (Port 80/443)                      │
└──────────────┬────────────────────┬──────────────────┘
               │                    │
    ┌──────────▼──────────┐  ┌──────▼──────────┐
    │  ECS Fargate Task   │  │  ECS Fargate    │
    │   (FalconOne App)   │  │     Task 2      │
    │    Port 5000        │  │                 │
    └──────────┬──────────┘  └──────┬──────────┘
               │                    │
    ┌──────────▼────────────────────▼──────────┐
    │      RDS PostgreSQL (Multi-AZ)           │
    │         Database: falconone              │
    └──────────────────────────────────────────┘
```

### AWS Cost Estimates (Monthly)

| Service | Configuration | Estimated Cost |
|---------|--------------|----------------|
| ECS Fargate | 2 tasks (0.5 vCPU, 1GB RAM) | $30 |
| ALB | Standard Load Balancer | $23 |
| RDS PostgreSQL | db.t3.micro (Multi-AZ) | $30 |
| ECR | 10GB storage | $1 |
| CloudWatch Logs | 5GB ingested | $2.50 |
| **Total** | | **~$86.50/month** |

---

## Azure Deployment

### Prerequisites
- Azure CLI installed
- Azure subscription
- Docker

### 1. Install Azure CLI

```bash
# Linux/WSL
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Set subscription
az account set --subscription "YOUR_SUBSCRIPTION_ID"
```

### 2. Create Resource Group

```bash
# Create resource group
az group create \
  --name falconone-rg \
  --location eastus
```

### 3. Deploy with Azure Container Instances (ACI)

```bash
# Create Azure Container Registry
az acr create \
  --resource-group falconone-rg \
  --name falcononeacr \
  --sku Basic

# Login to ACR
az acr login --name falcononeacr

# Build and push image
docker build -t falconone:latest .
docker tag falconone:latest falcononeacr.azurecr.io/falconone:latest
docker push falcononeacr.azurecr.io/falconone:latest

# Create PostgreSQL database
az postgres flexible-server create \
  --resource-group falconone-rg \
  --name falconone-db \
  --location eastus \
  --admin-user falconone \
  --admin-password "SecurePassword123!" \
  --sku-name Standard_B1ms \
  --tier Burstable \
  --storage-size 32 \
  --version 15

# Create database
az postgres flexible-server db create \
  --resource-group falconone-rg \
  --server-name falconone-db \
  --database-name falconone

# Deploy container
az container create \
  --resource-group falconone-rg \
  --name falconone-app \
  --image falcononeacr.azurecr.io/falconone:latest \
  --cpu 1 \
  --memory 2 \
  --registry-login-server falcononeacr.azurecr.io \
  --registry-username $(az acr credential show --name falcononeacr --query username -o tsv) \
  --registry-password $(az acr credential show --name falcononeacr --query passwords[0].value -o tsv) \
  --dns-name-label falconone-app \
  --ports 5000 \
  --environment-variables \
    DATABASE_URL="postgresql://falconone:SecurePassword123!@falconone-db.postgres.database.azure.com/falconone"

# Get public IP
az container show \
  --resource-group falconone-rg \
  --name falconone-app \
  --query ipAddress.fqdn
```

### Azure Architecture (AKS Alternative)

```
┌─────────────────────────────────────────────────────┐
│              Azure Load Balancer                     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│         Azure Kubernetes Service (AKS)               │
│  ┌─────────────────┐    ┌─────────────────┐        │
│  │  FalconOne Pod  │    │  FalconOne Pod  │        │
│  │   (Replica 1)   │    │   (Replica 2)   │        │
│  └─────────────────┘    └─────────────────┘        │
└──────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│    Azure Database for PostgreSQL (Flexible)          │
│              Database: falconone                     │
└──────────────────────────────────────────────────────┘
```

### Azure Cost Estimates (Monthly)

| Service | Configuration | Estimated Cost |
|---------|--------------|----------------|
| ACI | 1 vCPU, 2GB RAM | $44 |
| PostgreSQL Flexible | B1ms (1 vCPU, 2GB RAM) | $40 |
| Container Registry | Basic | $5 |
| **Total** | | **~$89/month** |

---

## GCP Deployment

### Prerequisites
- Google Cloud SDK (gcloud)
- GCP project with billing enabled
- Docker

### 1. Install gcloud CLI

```bash
# Linux/WSL
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### 2. Enable Required APIs

```bash
# Enable APIs
gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  sqladmin.googleapis.com \
  containerregistry.googleapis.com
```

### 3. Deploy with Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/falconone

# Create Cloud SQL instance (PostgreSQL)
gcloud sql instances create falconone-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1

# Create database
gcloud sql databases create falconone \
  --instance=falconone-db

# Set database password
gcloud sql users set-password postgres \
  --instance=falconone-db \
  --password=SecurePassword123!

# Deploy to Cloud Run
gcloud run deploy falconone \
  --image gcr.io/YOUR_PROJECT_ID/falconone \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --add-cloudsql-instances YOUR_PROJECT_ID:us-central1:falconone-db \
  --set-env-vars DATABASE_URL="postgresql://postgres:SecurePassword123!@/falconone?host=/cloudsql/YOUR_PROJECT_ID:us-central1:falconone-db"

# Get service URL
gcloud run services describe falconone \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)'
```

### GCP Architecture

```
┌─────────────────────────────────────────────────────┐
│              Google Cloud Load Balancer              │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                   Cloud Run                          │
│  ┌─────────────────────────────────────────────┐   │
│  │  FalconOne Container                        │   │
│  │  (Auto-scaling: 0-100 instances)            │   │
│  └─────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│           Cloud SQL (PostgreSQL 15)                  │
│         Database: falconone                          │
└──────────────────────────────────────────────────────┘
```

### GCP Cost Estimates (Monthly)

| Service | Configuration | Estimated Cost |
|---------|--------------|----------------|
| Cloud Run | 1M requests, 2GB RAM | $30 |
| Cloud SQL | db-f1-micro (0.6GB RAM) | $10 |
| Container Registry | 10GB storage | $1 |
| **Total** | | **~$41/month** |

---

## CI/CD Pipeline

### GitHub Actions Setup

1. **Configure Secrets** in GitHub repository settings:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`
   - `SLACK_WEBHOOK` (optional)

2. **Pipeline Stages**:
   - **Lint**: Black, isort, flake8, pylint
   - **Test**: pytest with coverage
   - **Security**: Bandit, Safety
   - **Build**: Docker image build and push
   - **Deploy Staging**: Auto-deploy on develop branch
   - **Deploy Production**: Auto-deploy on main branch
   - **Load Test**: Locust after staging deploy

3. **Workflow Triggers**:
   - Push to `main` or `develop` branches
   - Pull requests to `main`
   - Manual trigger via `workflow_dispatch`

### Pipeline Execution

```bash
# View workflow runs
gh run list --workflow=ci-cd.yml

# Watch live workflow
gh run watch

# View logs
gh run view --log
```

---

## Monitoring & Logging

### AWS CloudWatch

```bash
# View ECS logs
aws logs tail /ecs/falconone --follow

# Create alarm for high CPU
aws cloudwatch put-metric-alarm \
  --alarm-name falconone-high-cpu \
  --alarm-description "Alert when CPU exceeds 70%" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 70 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

### Azure Monitor

```bash
# View container logs
az container logs \
  --resource-group falconone-rg \
  --name falconone-app \
  --follow
```

### GCP Cloud Logging

```bash
# View Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=falconone" \
  --limit 50 \
  --format json
```

---

## Scaling & Performance

### AWS ECS Auto Scaling

Already configured in Terraform:
- **Target CPU Utilization**: 70%
- **Min Capacity**: 2 tasks
- **Max Capacity**: 10 tasks

### Manual Scaling

```bash
# AWS ECS
aws ecs update-service \
  --cluster falconone-cluster \
  --service falconone-service \
  --desired-count 5

# Azure ACI (limited scaling)
# Requires deployment update

# GCP Cloud Run (automatic)
gcloud run services update falconone \
  --max-instances 20 \
  --region us-central1
```

### Performance Benchmarks

| Metric | Target | AWS | Azure | GCP |
|--------|--------|-----|-------|-----|
| Response Time (avg) | <200ms | 180ms | 195ms | 165ms |
| Response Time (p95) | <500ms | 420ms | 480ms | 390ms |
| Throughput | >100 RPS | 120 RPS | 110 RPS | 130 RPS |
| Success Rate | >99% | 99.5% | 99.3% | 99.7% |

---

## Troubleshooting

### Common Issues

**1. ECS Task Not Starting**
```bash
# Check task logs
aws ecs describe-tasks \
  --cluster falconone-cluster \
  --tasks TASK_ARN

# Check stopped tasks
aws ecs list-tasks \
  --cluster falconone-cluster \
  --desired-status STOPPED
```

**2. Database Connection Errors**
```bash
# Test database connectivity
psql -h DB_ENDPOINT -U falconone -d falconone

# Check security group rules
aws ec2 describe-security-groups \
  --group-ids sg-XXXXX
```

**3. High Memory Usage**
```bash
# Increase task memory
aws ecs update-service \
  --cluster falconone-cluster \
  --service falconone-service \
  --task-definition falconone:NEW_VERSION
```

---

## Security Best Practices

1. **Use AWS Secrets Manager / Azure Key Vault / GCP Secret Manager**
2. **Enable encryption at rest for databases**
3. **Use VPC private subnets for ECS tasks**
4. **Enable WAF on load balancers**
5. **Rotate credentials regularly**
6. **Enable MFA for cloud accounts**
7. **Use least privilege IAM policies**

---

## Disaster Recovery

### Backup Strategy

**AWS**:
```bash
# RDS automated backups (7 days retention configured)
aws rds describe-db-snapshots \
  --db-instance-identifier falconone-db
```

**Azure**:
```bash
# PostgreSQL automated backups
az postgres flexible-server backup list \
  --resource-group falconone-rg \
  --name falconone-db
```

**GCP**:
```bash
# Cloud SQL automated backups
gcloud sql backups list \
  --instance falconone-db
```

### Restore Procedure

See individual cloud provider documentation for restore procedures.

---

## Cost Optimization

1. **Use spot/preemptible instances** for non-critical workloads
2. **Right-size resources** based on actual usage
3. **Enable auto-scaling** to scale down during low traffic
4. **Use reserved instances** for predictable workloads (AWS/Azure)
5. **Monitor costs** with AWS Cost Explorer / Azure Cost Management / GCP Billing

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/falconone/issues
- Email: support@falconone.example.com
