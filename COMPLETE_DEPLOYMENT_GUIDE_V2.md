# 🚀 Quantum-SwarmVLA: Complete Deployment Guide

This guide provides a step-by-step walkthrough to deploy your **Quantum-SwarmVLA** application (React Frontend + Flask/Quantum Backend) to a public environment.

---

## 1. Prerequisites

Before starting, ensure you have the following tools and accounts:

### **Tools Installed**
*   **Node.js & npm**: For building the frontend (`node -v` should be v16+).
*   **Python 3.10+**: For the backend environment.
*   **Git**: For version control and pushing code to repositories.
*   **VS Code**: (Optional) For editing configuration files.

### **Accounts & Services**
*   **GitHub**: To host your source code (Required for Vercel/Render).
*   **Vercel Account**: Free tier is perfect for hosting the **Frontend**.
*   **Render.com Account**: Free tier supports Python web services for the **Backend**.
*   **Twilio Account**: (Existing) You already have SID/Auth Token for SMS alerts.

---

## 2. Front-End Deployment Steps (Vercel)

We will deploy the React Frontend to Vercel, which provides a Global CDN.

### **Step 2.1: Prepare the Code**
1.  Open `d:\QuantumSwarmVLA\frontend`.
2.  Ensure your `package.json` has the build script: `"build": "react-scripts build"`.
3.  **Optimization**: The build script automatically optimizes and minifies your CSS/JS.
4.  **Environment Variables**:
    *   Create a file named `.env.production` in the `frontend` folder (if strictly needed locally), but for Vercel, we set them in the dashboard.
    *   The key variable is `REACT_APP_API_URL`.

### **Step 2.2: Push to GitHub**
1.  Initialize a Git repo in your project root if you haven't.
2.  Commit your code and push it to a public/private GitHub repository.

### **Step 2.3: Deploy to Vercel**
1.  Log in to **[Vercel Dashboard](https://vercel.com)**.
2.  Click **"Add New Project"** > **"Import"**.
3.  Select your GitHub repository (`QuantumSwarmVLA`).
4.  **Configure Output Settings**:
    *   **Framework Preset**: Create React App (Auto-detected).
    *   **Root Directory**: Click "Edit" and select `frontend`.
5.  **Environment Variables**:
    *   Add `REACT_APP_API_URL`. *Value*: You will get this AFTER deploying the backend. For now, leave it or put a placeholder.
6.  Click **"Deploy"**.

---

## 3. Back-End Deployment Steps (Render.com)

We will deploy the Flask Backend to Render, which runs `gunicorn`.

### **Step 3.1: Prepare the Backend**
1.  **Requirements**: Ensure `d:\QuantumSwarmVLA\backend\requirements.txt` exists and includes `gunicorn`.
2.  **Procfile**: Ensure `d:\QuantumSwarmVLA\backend\Procfile` exists with:
    ```text
    web: gunicorn quantum_swarmvla_backend:app
    ```

### **Step 3.2: Deploy to Render**
1.  Log in to **[Render Dashboard](https://dashboard.render.com)**.
2.  Click **"New +"** > **"Web Service"**.
3.  Connect your GitHub repository.
4.  **Settings**:
    *   **Root Directory**: `backend`
    *   **Runtime**: Python 3
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `gunicorn quantum_swarmvla_backend:app`
5.  **Environment Variables** (Add these from your local `.env`):
    *   `TWILIO_ACCOUNT_SID`: ...
    *   `TWILIO_AUTH_TOKEN`: ...
    *   `TWILIO_PHONE`: ...
    *   `RESCUE_TEAM_PHONES`: ...
    *   `N_QUBITS`: 4
6.  Click **"Create Web Service"**.

*Render will provide a URL like `https://quantum-backend.onrender.com`. Copy this URL.*

### **Step 3.3: Link Frontend to Backend**
1.  Go back to **Vercel** > Project Settings > Environment Variables.
2.  Add/Edit `REACT_APP_API_URL` with value: `https://quantum-backend.onrender.com`.
3.  **Redeploy** the Frontend (Go to Deployments > Redeploy) for changes to take effect.

---

## 4. Using DOM Agent for Verification

Once deployed, you can use an automated "DOM Agent" (like the one I use, or tools like Playwright/Selenium) to validate the deployment.

### **Instruction Prompts for an AI Agent:**
If you validting using an AI, give it these instructions:
1.  **"Navigate to [Your Vercel URL]"**: Verify the page load status (HTTP 200).
2.  **"Check Page Title"**: Confirm it says "Quantum-SwarmVLA-Edge".
3.  **"Validate API Connection"**:
    *   Wait for the "System Metrics" section to load.
    *   If metrics appear (zeros or numbers), the **Backend Connection is SUCCESSFUL**.
    *   If it shows "Network Error", the connection failed (check CORS or URL).
4.  **"Test Interaction"**:
    *   "Click the 'Start Streaming' button".
    *   "Verify that the button text changes to 'Stop Streaming'".

---

## 5. Final Verification & Monitoring

### **Manual Verification Checklist**
-   [ ] **Frontend Load**: Does the dashboard appear without visual glitches?
-   [ ] **API Connectivity**: Do the "System Metrics" numbers load?
-   [ ] **Image Analysis**: Upload a test image (e.g., flood). Does it return a result?
-   [ ] **SMS Alert**: Does a High-Risk detection trigger a real SMS to your phone?

### **Monitoring Tools**
*   **Render Logs**: View real-time server logs in the Render Dashboard to see Python errors or print statements.
*   **Vercel Analytics**: Check standard web vitals and error rates.
*   **Sentry**: (Optional) integrate Sentry for real-time error tracking in React/Flask.

---

## 6. Troubleshooting

### **Common Issues**
1.  **CORS Error (Network Error)**:
    *   *Symptom*: Frontend console shows `Access-Control-Allow-Origin` error.
    *   *Fix*: Update `CORS(app)` in `backend/quantum_swarmvla_backend.py` to allow specific domains or `resources={r"/api/*": {"origins": "*"}}`.
2.  **Module Not Found (Backend)**:
    *   *Symptom*: Render build fails.
    *   *Fix*: Ensure all imports (like `qiskit`, `torch`) are listed in `requirements.txt`.
3.  **Environment Variable Missing**:
    *   *Symptom*: Twilio fails to send SMS.
    *   *Fix*: Double-check the spelling of `TWILIO_AUTH_TOKEN` in Render settings.
4.  **Frontend connecting to Localhost**:
    *   *Symptom*: App works but tries to hit `localhost:5000`.
    *   *Fix*: Ensure `REACT_APP_API_URL` is set in Vercel and **you have redeployed** the frontend after setting it.

---
*For further assistance, refer to the [Vercel Documentation](https://vercel.com/docs) or [Render Documentation](https://render.com/docs).*
