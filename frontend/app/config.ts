// API Configuration
// Change this to your server IP for remote deployment
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
export const WS_BASE_URL = API_BASE_URL.replace('http', 'ws');
