# Deployment Guide for Quantum-SwarmVLA

This guide explains how to deploy the **Quantum-SwarmVLA** application. The recommended approach is to deploy the Frontend and Backend separately for better scalability and maintainability.

## 1. Backend Deployment (Render / Heroku)

We recommend using **Render** (easiest free tier) or Heroku.

### Prerequisites
- Ensure `requirements.txt` is in the `backend/` folder.
- Ensure `Procfile` is in the `backend/` folder (Added automatically).

### Steps for Render.com

1. **Create Account**: Go to [Render](https://render.com) and sign up/login.
2. **New Web Service**: Click "New +" and select "Web Service".
3. **Connect Repository**: Connect your GitHub repository.
4. **Configuration**:
   - **Root Directory**: `backend` (Important: Set this to the backend folder).
   - **Name**: `quantum-backend` (or similar).
   - **Runtime**: `Python 3`.
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn quantum_swarmvla_backend:app`
5. **Environment Variables**:
   Add the following variables from your `.env` file:
   - `TWILIO_ACCOUNT_SID`: ...
   - `TWILIO_AUTH_TOKEN`: ...
   - `TWILIO_PHONE`: ...
   - `RESCUE_TEAM_PHONES`: ...
   - `NGROK_AUTH_TOKEN`: (Optional, not needed for production if using public URL)
   - `TESTING_MODE`: `False` (or `True` if you don't want to send real SMS during initial test)
   - `N_QUBITS`: `4`
6. **Deploy**: Click "Create Web Service".

Once deployed, Render will give you a URL (e.g., `https://quantum-backend.onrender.com`). **Copy this URL.**

---

## 2. Frontend Deployment (Vercel)

We recommend using **Vercel** for React applications.

### Steps for Vercel

1. **Create Account**: Go to [Vercel](https://vercel.com) and sign up/login.
2. **Add New Project**: Click "Add New..." -> "Project".
3. **Import Repository**: Import your GitHub repository.
4. **Configuration**:
   - **root-directory**: Edit this and select `frontend`.
   - **Framework Preset**: it should auto-detect `Create React App`.
   - **Build Command**: `npm run build` (default).
   - **Output Directory**: `build` (default).
5. **Environment Variables**:
   You need to tell the frontend where the backend is.
   - `REACT_APP_API_URL`: Paste the Render Backend URL (e.g., `https://quantum-backend.onrender.com`)
   
   *Note: usage of this variable requires updating the frontend code to use it instead of localhost. See below.*
6. **Deploy**: Click "Deploy".

---

## 3. Code Adjustments for Production

To ensure the Frontend connects to the deployed Backend instead of `localhost`, ensure your `App.jsx` or API service file uses the environment variable.

**Example Update in `frontend/src/App.jsx` (or wherever you make API calls):**

```javascript
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// Use API_URL in your axios calls
axios.post(`${API_URL}/api/analyze`, formData)
```

## 4. Combined Deployment (Alternative)

If you strictly want to deploy them together on a single server (like a VPS or PythonAnywhere):

1. **Build Frontend**:
   ```bash
   cd frontend
   npm run build
   ```
   This creates a `build` folder.
2. **Serve from Flask**:
   Move the `build` folder to `backend/static` (rename or configure Flask to serve static files from there).
   Update Flask (`quantum_swarmvla_backend.py`) to serve `index.html` on the root route `/`.

**Recommended**: Use Method 1 & 2 (Separate) for best results.
