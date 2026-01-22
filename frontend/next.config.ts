import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Use 'standalone' for Docker/Railway, 'export' for Tauri
  output: (process.env.DEPLOY_TARGET === 'railway' || process.env.DEPLOY_TARGET === 'docker') ? 'standalone' : 'export',
  images: {
    unoptimized: true,
  },
  // Allow Railway domain
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000',
  },
};

export default nextConfig;
