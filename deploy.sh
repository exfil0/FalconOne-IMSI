#!/bin/bash
# FalconOne Deployment Script
# Automated setup for Ubuntu 24.04.1 LTS
# Version 1.3 - Cloud Deployment Support

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║         FalconOne System Deployment Script               ║"
echo "║                    Version 1.3                            ║"
echo "║        Local & Cloud-Native Kubernetes Support            ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Parse command-line arguments
DEPLOYMENT_MODE="local"  # Default: local deployment
REGISTRY=""
NAMESPACE="falconone"

while [[ $# -gt 0 ]]; do
    case $1 in
        --cloud)
            DEPLOYMENT_MODE="cloud"
            shift
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cloud] [--registry REGISTRY] [--namespace NAMESPACE]"
            exit 1
            ;;
    esac
done

echo "[*] Deployment Mode: $DEPLOYMENT_MODE"

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "[!] Please do not run as root. Run as regular user with sudo access."
   exit 1
fi

# Function to print section headers
print_section() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ==================== CLOUD DEPLOYMENT ====================
if [ "$DEPLOYMENT_MODE" == "cloud" ]; then
    print_section "Cloud Deployment Mode - Building Docker Image"
    
    # Check for Docker
    if ! command -v docker &> /dev/null; then
        echo "[!] Docker not found. Installing Docker..."
        sudo apt update
        sudo apt install -y docker.io
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker $USER
        echo "[*] Docker installed. You may need to log out and back in."
    fi
    
    # Build Docker image
    echo "[*] Building FalconOne Docker image..."
    docker build -t falconone:1.3 --target production .
    
    if [ $? -ne 0 ]; then
        echo "[!] Docker build failed"
        exit 1
    fi
    
    # Tag and push to registry if specified
    if [ -n "$REGISTRY" ]; then
        print_section "Pushing to Container Registry"
        echo "[*] Tagging image for registry: $REGISTRY"
        docker tag falconone:1.3 $REGISTRY/falconone:1.3
        docker tag falconone:1.3 $REGISTRY/falconone:latest
        
        echo "[*] Pushing image to registry..."
        docker push $REGISTRY/falconone:1.3
        docker push $REGISTRY/falconone:latest
        
        if [ $? -ne 0 ]; then
            echo "[!] Docker push failed. Ensure you are logged in: docker login $REGISTRY"
            exit 1
        fi
    fi
    
    # Deploy to Kubernetes
    print_section "Deploying to Kubernetes"
    
    # Check for kubectl
    if ! command -v kubectl &> /dev/null; then
        echo "[!] kubectl not found. Please install kubectl first."
        echo "    https://kubernetes.io/docs/tasks/tools/"
        exit 1
    fi
    
    # Create namespace
    echo "[*] Creating namespace: $NAMESPACE"
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    echo "[*] Applying Kubernetes manifests..."
    kubectl apply -f k8s-deployment.yaml
    
    if [ $? -ne 0 ]; then
        echo "[!] Kubernetes deployment failed"
        exit 1
    fi
    
    # Wait for deployments to be ready
    echo "[*] Waiting for deployments to be ready..."
    kubectl rollout status deployment/falconone-agent -n $NAMESPACE --timeout=300s
    kubectl rollout status deployment/falconone-coordinator -n $NAMESPACE --timeout=300s
    
    # Get service endpoints
    print_section "Deployment Summary"
    echo "[*] FalconOne deployed successfully to Kubernetes!"
    echo ""
    echo "Namespace: $NAMESPACE"
    echo "Agents: $(kubectl get pods -n $NAMESPACE -l component=agent --no-headers | wc -l)"
    echo "Coordinator: $(kubectl get pods -n $NAMESPACE -l component=coordinator --no-headers | wc -l)"
    echo ""
    echo "Access services:"
    echo "  kubectl port-forward -n $NAMESPACE svc/falconone-agent-service 5000:5000"
    echo "  kubectl port-forward -n $NAMESPACE svc/falconone-agent-service 8080:8080"
    echo ""
    echo "View logs:"
    echo "  kubectl logs -n $NAMESPACE -l component=agent -f"
    echo "  kubectl logs -n $NAMESPACE -l component=coordinator -f"
    echo ""
    echo "Scale agents:"
    echo "  kubectl scale deployment/falconone-agent -n $NAMESPACE --replicas=5"
    echo ""
    
    exit 0
fi

# ==================== LOCAL DEPLOYMENT ====================

# ==================== LOCAL DEPLOYMENT ====================

print_section "Local Deployment Mode"

# Update system
print_section "Updating System Packages"
sudo apt update -y
sudo apt upgrade -y

# Install essential tools
print_section "Installing Essential Tools"
sudo apt install -y \
    git \
    cmake \
    build-essential \
    libusb-1.0-0-dev \
    python3-pip \
    python3-venv \
    unattended-upgrades \
    linux-tools-common \
    ubuntu-drivers-common

# Install SDR tools
print_section "Installing SDR Tools"
sudo apt install -y \
    rtl-sdr \
    hackrf \
    airspy \
    gnuradio \
    gnuradio-dev

# Create FalconOne directories
print_section "Creating FalconOne Directories"
sudo mkdir -p /var/log/falconone
sudo mkdir -p /var/lib/falconone
sudo chown -R $USER:$USER /var/log/falconone
sudo chown -R $USER:$USER /var/lib/falconone

# Create Python virtual environment
print_section "Setting Up Python Environment"
python3 -m venv ~/falconone_env
source ~/falconone_env/bin/activate

# Install Python dependencies
print_section "Installing Python Dependencies"
pip install --upgrade pip
pip install -r requirements.txt

# Configure system optimizations
print_section "Configuring System Optimizations"
# Set CPU governor to performance
echo "[*] Setting CPU governor to performance mode..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee $cpu > /dev/null
done

# Disable unnecessary services
echo "[*] Disabling unnecessary services..."
sudo systemctl disable snapd || true
sudo systemctl disable bluetooth || true

# Set up log rotation
print_section "Configuring Log Rotation"
sudo tee /etc/logrotate.d/falconone > /dev/null <<EOF
/var/log/falconone/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 $USER $USER
}
EOF

# Add user to required groups
print_section "Configuring User Permissions"
sudo usermod -a -G dialout,plugdev $USER

# Create systemd service (optional)
print_section "Creating Systemd Service (optional)"
cat << EOF | sudo tee /etc/systemd/system/falconone.service > /dev/null
[Unit]
Description=FalconOne IMSI/TMSI Catcher
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$HOME/falconone_env/bin/python run.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                 Deployment Complete!                      ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Next Steps:"
echo "1. Log out and log back in (for group changes)"
echo "2. Activate virtual environment: source ~/falconone_env/bin/activate"
echo "3. Edit configuration: nano config/config.yaml"
echo "4. Run system: python run.py"
echo ""
echo "Optional:"
echo "- Enable service: sudo systemctl enable falconone"
echo "- Start service: sudo systemctl start falconone"
echo ""
echo "Cloud Deployment:"
echo "- Build & deploy to Kubernetes: ./deploy.sh --cloud --registry YOUR_REGISTRY"
echo "- Example: ./deploy.sh --cloud --registry docker.io/yourname"
echo ""
echo "⚠️  REMEMBER: Research use only within Faraday cage!"
echo ""
