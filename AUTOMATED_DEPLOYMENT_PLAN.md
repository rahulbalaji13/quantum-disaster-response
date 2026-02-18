# Automated Deployment System Design for QuantumSwarmVLA

This document outlines the detailed design for an automated deployment system involving CI/CD, a custom DOM Agent for error handling, rollback mechanisms, and reporting.

## 1. System Architecture

The system utilizes **GitHub Actions** as the central orchestrator for Continuous Integration and Deployment. 

- **Frontend Host**: Vercel (Optimized for React)
- **Backend Host**: Render (Optimized for Python/Flask)
- **CI/CD Platform**: GitHub Actions
- **Monitoring/Agent**: Custom Python+Playwright "DOM Agent"

## 2. CI/CD Pipeline Strategy

We will use two separate workflows to ensure isolated deployments for frontend and backend changes.

### A. Frontend Pipeline (`frontend-ci.yml`)
**Triggers**: Push to `main` (Production) or Pull Requests (Preview).

1.  **Setup**: Ubuntu runner, Node.js environment.
2.  **Test**: 
    -   Install dependencies (`npm ci`).
    -   Run Unit Tests (`npm test`).
    -   **Gate**: Pipeline stops if tests fail.
3.  **Build**: Run `npm run build` to verify compilation.
4.  **Deploy**:
    -   Use `vercel/actions` to deploy to Vercel.
    -   Production deployment on `main` branch.
    -   Preview deployment on PRs.

### B. Backend Pipeline (`backend-ci.yml`)
**Triggers**: Push to `main` or Pull Requests affecting `backend/**`.

1.  **Setup**: Ubuntu runner, Python 3.10.
2.  **Test**:
    -   Install dependencies (`pip install -r requirements.txt`).
    -   Run Pytest (`pytest`).
    -   **Gate**: Pipeline stops if tests fail.
3.  **Deploy**:
    -   Trigger Render Deploy Hook (Webhook) or use Render API.
    -   Wait for deployment to complete (polling health endpoint).

## 3. Error Handling: The DOM Agent

If the deployment succeeds but the application is broken (e.g., white screen, API failures), standard CI might pass. The **DOM Agent** solves this.

### DOM Agent Specification
-   **Technology**: Python + Playwright (Headless Browser Automation).
-   **Location**: `qa_agent/dom_agent.py`.
-   **Execution**: Runs as the final step in the GitHub Action.

### Agent Logic
1.  **Navigation**: Accesses the deployed URL.
2.  **Visual Inspection**:
    -   Checks if `<div id="root">` is empty.
    -   Verifies Page Title is "Quantum-SwarmVLA-Edge".
    -   Wait for specific dashboard elements (e.g., "System Metrics").
3.  **Console Monitoring**: Listen for Browser Console Errors (red text).
    -   *Detection*: "Network Error" -> Suggests API CORS or URL mismatch.
    -   *Detection*: "404 Not Found" -> Suggests missing route or file.
4.  **API Validation**: The agent attempts to hit the backend `/api/health` via the frontend UI or direct fetch.

### Auto-Rectification & Notification
1.  **Log Analysis**: Captures a screenshot and console logs to `artifacts/error_report`.
2.  **Rectification**:
    -   *Scenario*: If API fails, the Agent checks if the `REACT_APP_API_URL` environment variable is reachable.
3.  **Notification**: Sends a structured report to a Slack/Discord webhook or Email.

## 4. Rollback Mechanism

If the DOM Agent reports a `CRITICAL` failure:
1.  **Frontend**: The pipeline executes `vercel rollback` to revert the alias to the previous deployment.
2.  **Backend**: The pipeline triggers the Render API to rollback to the previous commit.

## 5. Documentation and Reporting Strategy

-   **Deployment Logs**: Stored in GitHub Actions "Artifacts".
-   **Error Reports**: JSON + Screenshot saved to `deployment_logs/`.
-   **Status Badges**: Added to `README.md` (e.g., "Deployment: Passing").

## 6. Testing Integration Plan

1.  **Unit Tests**: Run immediately on commit.
2.  **Integration Tests**: Run in the CI environment (Connecting frontend build to a mock backend or staging backend).
3.  **E2E (DOM Agent)**: Runs post-deployment against the live URL.

## 7. Scalability and Security

-   **Secrets**: All API Keys (Twilio, Render, Vercel) stored in **GitHub Secrets**, injected only at runtime.
-   **Concurrency**: Separate workflows allow Frontend and Backend to scale independently.
-   **Access Control**: Branch protection rules on `main` require CI to pass before merging.

---

## Action Plan

1.  **Create CI Workflows**: Write `.github/workflows/deploy.yml`.
2.  **Develop DOM Agent**: Create `qa_agent/` with the Playwright script.
3.  **Configure Hooks**: Set up Vercel/Render webhooks.
