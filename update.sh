#!/bin/bash

# Deployment Update Script
# Usage: ./update.sh

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Starting System Update ===${NC}"

# Detect installation directory
if [ -d "/opt/trading-agents" ]; then
    DIR="/opt/trading-agents"
elif [ -d "/opt/trading" ]; then
    DIR="/opt/trading"
else
    DIR="$(pwd)"
fi

echo -e "${BLUE}Using directory: ${DIR}${NC}"
cd $DIR

# Pull latest changes
echo -e "${BLUE}Pulling latest changes from git...${NC}"
git pull origin main

# Rebuild containers
echo -e "${BLUE}Rebuilding Docker containers...${NC}"
# Use sudo if docker requires it
if groups | grep -q 'docker'; then
    DOCKER="docker"
else
    DOCKER="sudo docker"
fi

$DOCKER compose down
$DOCKER compose build --no-cache frontend
$DOCKER compose up -d

echo -e "${GREEN}=== Update Complete! ===${NC}"
$DOCKER ps
