#!/bin/bash

# Trading Agents Docker Installation Script
# Complete installation for Docker deployment with custom domain
# Integrates with OpenAlgo and includes LangSmith tracing

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║   ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗        ║"
echo "║   ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝        ║"
echo "║      ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗       ║"
echo "║      ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║       ║"
echo "║      ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝       ║"
echo "║      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝        ║"
echo "║                                                                  ║"
echo "║              █████╗  ██████╗ ███████╗███╗   ██╗████████╗███████╗ ║"
echo "║             ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██╔════╝ ║"
echo "║             ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ███████╗ ║"
echo "║             ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ╚════██║ ║"
echo "║             ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ███████║ ║"
echo "║             ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ║"
echo "║                                                                  ║"
echo "║                    DOCKER INSTALLATION                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to log messages
log() {
    echo -e "${2}${1}${NC}"
}

# Function to check command status
check_status() {
    if [ $? -ne 0 ]; then
        log "Error: $1" "$RED"
        exit 1
    fi
}

# Function to generate random hex string
generate_hex() {
    python3 -c "import secrets; print(secrets.token_hex(32))"
}

# Start installation
log "Starting Trading Agents Docker Installation..." "$GREEN"
log "================================================" "$GREEN"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log "WARNING: Running as root user is not recommended for production." "$YELLOW"
   log "For better security, consider creating a non-root user with sudo privileges." "$YELLOW"
   read -p "Do you want to continue as root? (y/n): " continue_as_root
   if [[ ! $continue_as_root =~ ^[Yy]$ ]]; then
       log "Installation cancelled. Create a non-root user with:" "$BLUE"
       log "  adduser tradingagents" "$BLUE"
       log "  usermod -aG sudo tradingagents" "$BLUE"
       log "  su - tradingagents" "$BLUE"
       exit 0
   fi
   log "Continuing as root user..." "$YELLOW"
   SUDO=""
else
   SUDO="sudo"
fi

# Check OS
if [ ! -f /etc/os-release ]; then
    log "Unsupported operating system" "$RED"
    exit 1
fi

OS_TYPE=$(grep -w "ID" /etc/os-release | cut -d "=" -f 2 | tr -d '"')
log "Detected OS: $OS_TYPE" "$BLUE"

# Support Ubuntu/Debian
if [[ "$OS_TYPE" != "ubuntu" && "$OS_TYPE" != "debian" ]]; then
    log "This script currently supports Ubuntu/Debian. Detected: $OS_TYPE" "$YELLOW"
    read -p "Do you want to continue anyway? (y/n): " continue_anyway
    if [[ ! $continue_anyway =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Collect installation information
log "\n=== Installation Configuration ===" "$BLUE"

# Get API domain
while true; do
    read -p "Enter your API domain (e.g., api.yourdomain.com): " API_DOMAIN
    if [ -z "$API_DOMAIN" ]; then
        log "Error: API domain is required" "$RED"
        continue
    fi
    if [[ ! $API_DOMAIN =~ ^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$ ]]; then
        log "Error: Invalid domain format" "$RED"
        continue
    fi
    break
done

# Get frontend domain
while true; do
    read -p "Enter your frontend domain (e.g., trading.yourdomain.com): " FRONTEND_DOMAIN
    if [ -z "$FRONTEND_DOMAIN" ]; then
        log "Error: Frontend domain is required" "$RED"
        continue
    fi
    if [[ ! $FRONTEND_DOMAIN =~ ^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$ ]]; then
        log "Error: Invalid domain format" "$RED"
        continue
    fi
    break
done

# Get OpenAlgo URL
log "\n=== OpenAlgo Configuration ===" "$YELLOW"
log "If OpenAlgo is running on THIS same server, the agents will connect via localhost:5000" "$BLUE"
log "This is more secure and faster than going through the public domain." "$BLUE"
read -p "Is OpenAlgo running on this same server? (y/n): " openalgo_same_server
if [[ $openalgo_same_server =~ ^[Yy]$ ]]; then
    OPENALGO_URL="http://127.0.0.1:5000"
    log "Using internal connection: $OPENALGO_URL" "$GREEN"
else
    read -p "Enter your OpenAlgo URL (e.g., https://openalgo.yourdomain.com): " OPENALGO_URL
    if [ -z "$OPENALGO_URL" ]; then
        log "Warning: No OpenAlgo URL provided. You'll need to set this later." "$YELLOW"
        OPENALGO_URL="http://localhost:5000"
    fi
fi

# Get OpenAlgo API Key
read -p "Enter your OpenAlgo API key (from OpenAlgo dashboard): " OPENALGO_API_KEY
if [ -z "$OPENALGO_API_KEY" ]; then
    log "Warning: No OpenAlgo API key provided. You'll need to set this later." "$YELLOW"
    OPENALGO_API_KEY="your_openalgo_api_key"
fi

# Get Gemini API Key
log "\n=== Gemini AI Configuration ===" "$YELLOW"
read -p "Enter your Gemini API key (from Google AI Studio): " GEMINI_API_KEY
if [ -z "$GEMINI_API_KEY" ]; then
    log "Error: Gemini API key is required for the agents to work" "$RED"
    exit 1
fi

# LangSmith Configuration (Optional)
log "\n=== LangSmith Tracing (Optional) ===" "$YELLOW"
log "LangSmith provides observability for your AI agents." "$BLUE"
read -p "Do you want to enable LangSmith tracing? (y/n): " enable_langsmith
if [[ $enable_langsmith =~ ^[Yy]$ ]]; then
    read -p "Enter your LangSmith API key: " LANGSMITH_API_KEY
    LANGSMITH_ENABLED="true"
    LANGSMITH_PROJECT="TradingAgents"
else
    LANGSMITH_API_KEY=""
    LANGSMITH_ENABLED="false"
    LANGSMITH_PROJECT=""
fi

# Get email for SSL certificate
read -p "Enter your email for SSL certificate notifications: " ADMIN_EMAIL
if [ -z "$ADMIN_EMAIL" ]; then
    ADMIN_EMAIL="admin@${API_DOMAIN#*.}"
fi

# Set installation path
INSTALL_PATH="/opt/trading-agents"

log "\n=== Installation Summary ===" "$YELLOW"
log "API Domain: https://$API_DOMAIN" "$BLUE"
log "Frontend Domain: https://$FRONTEND_DOMAIN" "$BLUE"
log "OpenAlgo URL: $OPENALGO_URL" "$BLUE"
log "Installation Path: $INSTALL_PATH" "$BLUE"
log "Email: $ADMIN_EMAIL" "$BLUE"
log "LangSmith Enabled: $LANGSMITH_ENABLED" "$BLUE"
echo ""

read -p "Proceed with installation? (y/n): " proceed
if [[ ! $proceed =~ ^[Yy]$ ]]; then
    log "Installation cancelled." "$YELLOW"
    exit 0
fi

# Update system
log "\n=== Updating System ===" "$BLUE"
$SUDO apt-get update -y
$SUDO apt-get upgrade -y
check_status "System update failed"

# Install required packages
log "\n=== Installing Required Packages ===" "$BLUE"
$SUDO apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    nginx \
    certbot \
    python3-certbot-nginx \
    ufw \
    python3 \
    python3-pip \
    python3-venv
check_status "Package installation failed"

# Install Docker
log "\n=== Installing Docker ===" "$BLUE"
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    $SUDO sh get-docker.sh
    $SUDO usermod -aG docker $USER
    rm get-docker.sh
    check_status "Docker installation failed"
else
    log "Docker already installed" "$GREEN"
fi

# Verify Docker Compose
if ! docker compose version &> /dev/null; then
    log "Error: Docker Compose not found" "$RED"
    exit 1
fi
log "Docker Compose version: $(docker compose version --short)" "$GREEN"

# Clone Trading Agents repository
log "\n=== Cloning Trading Agents Repository ===" "$BLUE"
if [ -d "$INSTALL_PATH" ]; then
    log "Warning: $INSTALL_PATH already exists" "$YELLOW"
    read -p "Remove existing installation? (y/n): " remove_existing
    if [[ $remove_existing =~ ^[Yy]$ ]]; then
        $SUDO rm -rf $INSTALL_PATH
    else
        log "Installation cancelled" "$RED"
        exit 1
    fi
fi

$SUDO git clone https://github.com/MasoomChoudhury/TradingSystem.git $INSTALL_PATH
check_status "Git clone failed"

cd $INSTALL_PATH

# Create required directories
log "\n=== Creating Required Directories ===" "$BLUE"
$SUDO mkdir -p backend/db logs
$SUDO chown -R 1000:1000 backend/db logs
$SUDO chmod -R 755 backend logs
check_status "Directory creation failed"

# Create backend .env file
log "\n=== Configuring Backend Environment ===" "$BLUE"
$SUDO tee backend/.env > /dev/null << EOF
# OpenAlgo Configuration
OPENALGO_API_KEY=$OPENALGO_API_KEY
OPENALGO_HOST=$OPENALGO_URL

# Gemini AI Configuration
GEMINI_API_KEY=$GEMINI_API_KEY

# LangSmith Tracing (Optional)
LANGCHAIN_TRACING_V2=$LANGSMITH_ENABLED
LANGCHAIN_API_KEY=$LANGSMITH_API_KEY
LANGCHAIN_PROJECT=$LANGSMITH_PROJECT
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
EOF

check_status "Backend environment configuration failed"

# Create Dockerfile for backend
log "\n=== Creating Backend Dockerfile ===" "$BLUE"
$SUDO tee backend/Dockerfile > /dev/null << 'EOF'
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create db directory
RUN mkdir -p db && chmod 777 db

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

check_status "Backend Dockerfile creation failed"

# Create Dockerfile for frontend
log "\n=== Creating Frontend Dockerfile ===" "$BLUE"
$SUDO tee frontend/Dockerfile > /dev/null << EOF
FROM node:20-slim

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy application code
COPY . .

# Set environment variables
ENV NEXT_PUBLIC_API_URL=https://$API_DOMAIN
ENV NODE_ENV=production

# Build the application
RUN npm run build

# Expose port
EXPOSE 3000

# Run the application
CMD ["npm", "start"]
EOF

check_status "Frontend Dockerfile creation failed"

# Update next.config.ts to allow the domains
log "\n=== Updating Next.js Configuration ===" "$BLUE"
$SUDO tee frontend/next.config.ts > /dev/null << EOF
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  images: {
    unoptimized: true,
  },
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'https://$API_DOMAIN',
  },
};

export default nextConfig;
EOF

check_status "Next.js configuration failed"

# Update CORS in main.py
log "\n=== Updating Backend CORS Configuration ===" "$BLUE"
$SUDO sed -i "s|allow_origins=\[\"http://localhost:3000\", \"tauri://localhost\"\]|allow_origins=[\"http://localhost:3000\", \"tauri://localhost\", \"https://$FRONTEND_DOMAIN\", \"https://$API_DOMAIN\"]|g" backend/main.py
check_status "CORS configuration failed"

# Create docker-compose.yaml
log "\n=== Creating Docker Compose Configuration ===" "$BLUE"
$SUDO tee docker-compose.yaml > /dev/null << EOF
version: '3.8'

services:
  trading-agents:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: trading-agents
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - ./backend/db:/app/db
      - ./backend/.env:/app/.env:ro
    environment:
      - TZ=Asia/Kolkata
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
    networks:
      - trading-network

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      args:
        - NEXT_PUBLIC_API_URL=https://$API_DOMAIN
    container_name: trading-frontend
    ports:
      - "127.0.0.1:3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=https://$API_DOMAIN
      - NODE_ENV=production
    depends_on:
      - trading-agents
    restart: unless-stopped
    networks:
      - trading-network

networks:
  trading-network:
    driver: bridge
EOF

check_status "Docker Compose configuration failed"

# Configure firewall (matching OpenAlgo pattern)
log "\n=== Configuring Firewall ===" "$BLUE"
log "Ports 8000/3000 will only be accessible via localhost (through nginx)" "$BLUE"
$SUDO ufw --force enable
$SUDO ufw default deny incoming
$SUDO ufw default allow outgoing
$SUDO ufw allow ssh
$SUDO ufw allow 80/tcp    # HTTP (for SSL renewal and redirect)
$SUDO ufw allow 443/tcp   # HTTPS (main access via nginx)
# Note: Ports 8000 (backend) and 3000 (frontend) are NOT opened
# They are bound to 127.0.0.1 and only accessible through nginx reverse proxy
check_status "Firewall configuration failed"

# Initial Nginx configuration (for Certbot)
log "\n=== Configuring Nginx (Initial) ===" "$BLUE"

# API Nginx config
$SUDO tee /etc/nginx/sites-available/$API_DOMAIN > /dev/null << EOF
server {
    listen 80;
    listen [::]:80;
    server_name $API_DOMAIN;
    
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }
    
    location / {
        return 301 https://\$server_name\$request_uri;
    }
}
EOF

# Frontend Nginx config
$SUDO tee /etc/nginx/sites-available/$FRONTEND_DOMAIN > /dev/null << EOF
server {
    listen 80;
    listen [::]:80;
    server_name $FRONTEND_DOMAIN;
    
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }
    
    location / {
        return 301 https://\$server_name\$request_uri;
    }
}
EOF

$SUDO ln -sf /etc/nginx/sites-available/$API_DOMAIN /etc/nginx/sites-enabled/
$SUDO ln -sf /etc/nginx/sites-available/$FRONTEND_DOMAIN /etc/nginx/sites-enabled/
$SUDO rm -f /etc/nginx/sites-enabled/default
$SUDO nginx -t
check_status "Nginx configuration test failed"

$SUDO systemctl enable nginx
$SUDO systemctl reload nginx
check_status "Nginx reload failed"

# Obtain SSL certificates
log "\n=== Obtaining SSL Certificates ===" "$BLUE"
log "Please wait while we obtain SSL certificates from Let's Encrypt..." "$YELLOW"
$SUDO certbot --nginx -d $API_DOMAIN -d $FRONTEND_DOMAIN --non-interactive --agree-tos --email $ADMIN_EMAIL
check_status "SSL certificate obtention failed"

# Final Nginx configuration with SSL
log "\n=== Configuring Nginx (Production with SSL) ===" "$BLUE"

# API Nginx config (Production)
$SUDO tee /etc/nginx/sites-available/$API_DOMAIN > /dev/null << EOF
# Rate limiting
limit_req_zone \$binary_remote_addr zone=api_limit:10m rate=50r/s;

# Upstream
upstream trading_agents {
    server 127.0.0.1:8000;
    keepalive 64;
}

# HTTP - Redirect to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name $API_DOMAIN;

    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        return 301 https://\$host\$request_uri;
    }
}

# HTTPS
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    
    server_name $API_DOMAIN;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/$API_DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$API_DOMAIN/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # CORS Headers
    add_header Access-Control-Allow-Origin "https://$FRONTEND_DOMAIN" always;
    add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
    add_header Access-Control-Allow-Headers "Content-Type, Authorization, X-Requested-With" always;
    add_header Access-Control-Allow-Credentials "true" always;
    
    # Client settings
    client_max_body_size 50M;
    client_body_timeout 300s;
    
    # Logging
    access_log /var/log/nginx/${API_DOMAIN}_access.log;
    error_log /var/log/nginx/${API_DOMAIN}_error.log;

    # WebSocket for logs
    location /ws/ {
        proxy_pass http://trading_agents;
        proxy_http_version 1.1;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
        proxy_connect_timeout 60s;
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_redirect off;
    }

    # API Endpoints
    location / {
        limit_req zone=api_limit burst=100 nodelay;
        limit_req_status 429;
        
        # Handle preflight
        if (\$request_method = 'OPTIONS') {
            add_header Access-Control-Allow-Origin "https://$FRONTEND_DOMAIN" always;
            add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
            add_header Access-Control-Allow-Headers "Content-Type, Authorization, X-Requested-With" always;
            add_header Access-Control-Max-Age 3600;
            add_header Content-Type "text/plain; charset=utf-8";
            add_header Content-Length 0;
            return 204;
        }
        
        proxy_pass http://trading_agents;
        proxy_http_version 1.1;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Connection "";
        proxy_redirect off;
    }

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript 
               application/json application/javascript application/xml+rss;
}
EOF

# Frontend Nginx config (Production)
$SUDO tee /etc/nginx/sites-available/$FRONTEND_DOMAIN > /dev/null << EOF
# Upstream
upstream trading_frontend {
    server 127.0.0.1:3000;
    keepalive 64;
}

# HTTP - Redirect to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name $FRONTEND_DOMAIN;

    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        return 301 https://\$host\$request_uri;
    }
}

# HTTPS
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    
    server_name $FRONTEND_DOMAIN;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/$API_DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$API_DOMAIN/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Client settings
    client_max_body_size 50M;
    
    # Logging
    access_log /var/log/nginx/${FRONTEND_DOMAIN}_access.log;
    error_log /var/log/nginx/${FRONTEND_DOMAIN}_error.log;

    # Main Application
    location / {
        proxy_pass http://trading_frontend;
        proxy_http_version 1.1;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_buffering off;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_redirect off;
    }

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript 
               application/json application/javascript application/xml+rss;
}
EOF

$SUDO nginx -t
check_status "Nginx configuration test failed"

$SUDO systemctl reload nginx
check_status "Nginx reload failed"

# Build and start Docker containers
log "\n=== Building Docker Images ===" "$BLUE"
log "This may take several minutes..." "$YELLOW"
$SUDO docker compose build --no-cache
check_status "Docker build failed"

log "\n=== Starting Docker Containers ===" "$BLUE"
$SUDO docker compose up -d
check_status "Docker container start failed"

# Wait for containers to be healthy
log "\nWaiting for containers to be healthy..." "$YELLOW"
sleep 15

# Check container status
log "\n=== Container Status ===" "$BLUE"
$SUDO docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Create management scripts
log "\n=== Creating Management Scripts ===" "$BLUE"

# Status script
$SUDO tee /usr/local/bin/trading-agents-status > /dev/null << 'EOFSCRIPT'
#!/bin/bash
echo "=========================================="
echo "Trading Agents Status"
echo "=========================================="
echo ""
echo "Container Status:"
sudo docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "trading-agents|trading-frontend"
echo ""
echo "Recent Backend Logs:"
sudo docker logs trading-agents --tail=20 2>&1
echo ""
echo "Recent Frontend Logs:"
sudo docker logs trading-frontend --tail=10 2>&1
EOFSCRIPT

$SUDO chmod +x /usr/local/bin/trading-agents-status

# Restart script
$SUDO tee /usr/local/bin/trading-agents-restart > /dev/null << 'EOFSCRIPT'
#!/bin/bash
echo "Restarting Trading Agents..."
cd /opt/trading-agents
sudo docker compose restart
sleep 10
echo "Container Status:"
sudo docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "trading-agents|trading-frontend"
EOFSCRIPT

$SUDO chmod +x /usr/local/bin/trading-agents-restart

# Logs script
$SUDO tee /usr/local/bin/trading-agents-logs > /dev/null << 'EOFSCRIPT'
#!/bin/bash
cd /opt/trading-agents
sudo docker compose logs -f --tail=100
EOFSCRIPT

$SUDO chmod +x /usr/local/bin/trading-agents-logs

# Update script
$SUDO tee /usr/local/bin/trading-agents-update > /dev/null << 'EOFSCRIPT'
#!/bin/bash
echo "Updating Trading Agents..."
cd /opt/trading-agents
sudo git pull
sudo docker compose down
sudo docker compose build --no-cache
sudo docker compose up -d
echo ""
echo "Update complete!"
sudo docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "trading-agents|trading-frontend"
EOFSCRIPT

$SUDO chmod +x /usr/local/bin/trading-agents-update

log "Management scripts created successfully!" "$GREEN"

# Setup SSL auto-renewal
log "\n=== Setting Up SSL Auto-Renewal ===" "$BLUE"
$SUDO mkdir -p /etc/letsencrypt/renewal-hooks/deploy
$SUDO tee /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh > /dev/null << 'EOFSCRIPT'
#!/bin/bash
systemctl reload nginx
EOFSCRIPT
$SUDO chmod +x /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh

# Installation complete
log "\n" "$NC"
log "╔══════════════════════════════════════════════════════════════════╗" "$GREEN"
log "║                                                                  ║" "$GREEN"
log "║     Trading Agents Docker Installation Complete!                 ║" "$GREEN"
log "║                                                                  ║" "$GREEN"
log "╚══════════════════════════════════════════════════════════════════╝" "$GREEN"

log "\nInstallation Summary:" "$YELLOW"
log "API URL: https://$API_DOMAIN" "$BLUE"
log "Frontend URL: https://$FRONTEND_DOMAIN" "$BLUE"
log "OpenAlgo URL: $OPENALGO_URL" "$BLUE"
log "Installation Path: $INSTALL_PATH" "$BLUE"
if [[ "$LANGSMITH_ENABLED" == "true" ]]; then
    log "LangSmith: Enabled (view traces at smith.langchain.com)" "$BLUE"
fi

log "\nNext Steps:" "$YELLOW"
log "1. Visit https://$FRONTEND_DOMAIN to access the Trading UI" "$GREEN"
log "2. Open Developer Tools (F12) to see agent activity" "$GREEN"
log "3. Start a trading session from the UI" "$GREEN"

log "\nUseful Commands:" "$YELLOW"
log "View status:   trading-agents-status" "$BLUE"
log "View logs:     trading-agents-logs" "$BLUE"
log "Restart:       trading-agents-restart" "$BLUE"
log "Update:        trading-agents-update" "$BLUE"

log "\nDocker Commands:" "$YELLOW"
log "Restart:  cd $INSTALL_PATH && sudo docker compose restart" "$BLUE"
log "Stop:     cd $INSTALL_PATH && sudo docker compose stop" "$BLUE"
log "Start:    cd $INSTALL_PATH && sudo docker compose start" "$BLUE"
log "Rebuild:  cd $INSTALL_PATH && sudo docker compose down && sudo docker compose build --no-cache && sudo docker compose up -d" "$BLUE"

if [[ "$LANGSMITH_ENABLED" == "true" ]]; then
    log "\nLangSmith Dashboard:" "$YELLOW"
    log "View your agent traces at: https://smith.langchain.com" "$BLUE"
fi

log "\n" "$NC"
log "╔══════════════════════════════════════════════════════════════════╗" "$GREEN"
log "║              Installation completed successfully!                ║" "$GREEN"
log "╚══════════════════════════════════════════════════════════════════╝" "$GREEN"
