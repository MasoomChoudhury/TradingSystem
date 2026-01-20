# TradingSystem Railway Deployment Guide

## Prerequisites
1. [Railway Account](https://railway.app/)
2. GitHub repository with your code
3. OpenAlgo broker credentials

## Quick Deploy

### Step 1: Push to GitHub
```bash
cd /Users/masoom/Developer/TradingSystem
git add .
git commit -m "Add Railway deployment"
git push origin main
```

### Step 2: Create Railway Project
1. Go to [railway.app](https://railway.app/)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your TradingSystem repository

### Step 3: Configure Services

**Backend Service:**
- Root Directory: `backend`
- Builder: Dockerfile
- Port: 8000

**Frontend Service:**
- Root Directory: `frontend`
- Builder: Dockerfile
- Port: 3000
- Environment: `DEPLOY_TARGET=railway`, `NEXT_PUBLIC_API_URL=https://your-backend.railway.app`

### Step 4: Set Environment Variables

In Railway dashboard, add these variables to the backend service:

| Variable | Value |
|----------|-------|
| `OPENALGO_API_KEY` | Your OpenAlgo API key |
| `OPENALGO_HOST` | `http://127.0.0.1:5000` (or external URL if separate) |
| `GEMINI_API_KEY` | Your Gemini API key |

### Step 5: Add Persistent Storage

For SQLite databases to persist:
1. Backend service → Settings → Volume
2. Mount path: `/data`
3. Update code to use `/data` for databases

### Step 6: OpenAlgo Setup

**Option A: Run OpenAlgo on Same Server**
- Add OpenAlgo as another service in Railway
- Use internal networking: `http://openalgo:5000`

**Option B: Use External OpenAlgo**
- Set `OPENALGO_HOST` to your external OpenAlgo URL
- Ensure it's accessible from Railway

## Testing
After deployment:
```bash
# Check health
curl https://your-app.railway.app/api/health

# Check session
curl https://your-app.railway.app/api/session/status
```

## Estimated Costs
- Backend: ~$5-7/month
- Frontend: ~$3-5/month
- Persistent Volume: ~$0.25/GB/month
- **Total: ~$10-15/month**

## Troubleshooting

### Cold Starts
Railway sleeps inactive services. To prevent:
- Use cron job to ping every 5 minutes
- Or upgrade to Pro plan

### SQLite Issues
If data disappears after restart:
- Ensure volume is mounted at `/data`
- Update database paths in code

### OpenAlgo Connection
If trades fail:
- Check OpenAlgo is running
- Verify API key is correct
- Check broker session is active
