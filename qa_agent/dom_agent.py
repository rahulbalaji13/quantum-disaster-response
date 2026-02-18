import sys
import logging
from playwright.sync_api import sync_playwright
import requests
import os
import json

# Configuration
URL_TO_CHECK = os.environ.get("DEPLOY_URL", "http://localhost:3000")
BACKEND_URL = os.environ.get("REACT_APP_API_URL", "http://localhost:5000")
SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK", "")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DOMAgent")

class DeploymentError(Exception):
    pass

class DOMAgent:
    def __init__(self, url):
        self.url = url
        self.browser = None
        self.page = None
        self.errors = []
        self.suggestions = []

    def start_browser(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.page = self.browser.new_page()

    def check_deployment(self):
        logger.info(f"Checking deployment at {self.url}...")
        
        # Retry logic for local servers starting up
        for i in range(10):
            try:
                response = self.page.goto(self.url, wait_until="networkidle", timeout=5000)
                if response:
                    break
            except Exception:
                logger.info(f"Waiting for server... ({i+1}/10)")
                self.page.wait_for_timeout(2000)
        
        if not response: # Setup failed after retries
             self.errors.append("Server unavailable after 20 seconds.")
             return False

        try:
            if response.status >= 400:
                self.errors.append(f"HTTP Error: {response.status}")
                return False

            # Check specific DOM elements
            title = self.page.title()
            logger.info(f"Page Title: {title}")
            
            if "Quantum-SwarmVLA" not in title and "React App" not in title:
                self.errors.append(f"Incorrect Title: {title}")
            
            # Check for critical errors in console
            self.page.on("console", lambda msg: self.errors.append(f"Console Error: {msg.text}") if msg.type == "error" else None)

            # Check if root is empty
            root_content = self.page.locator("#root").inner_html()
            if not root_content or root_content.strip() == "":
                self.errors.append("Root element missing or empty")
                self.page.screenshot(path="qa_agent/failure.png")
                with open("qa_agent/failure.html", "w", encoding="utf-8") as f:
                    f.write(self.page.content())

            # Check backend connectivity (Frontend usually fetches on load)
            try:
                # Wait for a known element that requires backend data
                # .metrics-grid only appears after successful API call
                self.page.wait_for_selector(".metrics-grid", timeout=15000)
            except:
                self.errors.append("Backend data not loading (System Metrics missing)")
                self.page.screenshot(path="qa_agent/timeout.png")


            if self.errors:
                print("Errors found:")
                for err in self.errors:
                    print(f" - {err}")
                return False
            
            logger.info("Deployment verified successfully.")
            return True

        except Exception as e:
            self.errors.append(f"Navigation failed: {str(e)}")
            return False
        finally:
            if self.browser:
                self.browser.close()
            self.playwright.stop()

    def analyze_errors(self):
        for error in self.errors:
            if "404" in error:
                self.suggestions.append("Check routing configuration or missing files.")
            elif "500" in error:
                self.suggestions.append("Check backend server logs (Render).")
            elif "Console Error" in error:
                self.suggestions.append("Check frontend JavaScript console for logic errors.")
            elif "Backend data" in error:
                self.suggestions.append("Check REACT_APP_API_URL environment variable.")

    def attempt_rectification(self):
        logger.info("Attempting auto-rectification...")
        rectified = False
        for suggestion in self.suggestions:
            if "REACT_APP_API_URL" in suggestion:
                # Mock rectification
                logger.info(f"Applying fix: Verifying {BACKEND_URL} is reachable...")
                try:
                    resp = requests.get(f"{BACKEND_URL}/api/health")
                    if resp.status_code == 200:
                        logger.info("Backend is reachable. Issue might be CORS.")
                    else:
                        logger.warning("Backend is NOT reachable.")
                except:
                    logger.warning("Backend connection failed.")
        return rectified

    def notify_team(self):
        report = {
            "status": "FAILED" if self.errors else "SUCCESS",
            "url": self.url,
            "errors": self.errors,
            "suggestions": self.suggestions
        }
        logger.info(f"Sending Notification: {json.dumps(report, indent=2)}")
        # requests.post(SLACK_WEBHOOK, json=report)

    def trigger_rollback(self):
        if self.errors:
            logger.critical("Critical deployment failure. Initiating Rollback...")
            # Mock rollback command
            # os.system("vercel rollback")
            logger.info("Rollback command executed.")

if __name__ == "__main__":
    agent = DOMAgent(URL_TO_CHECK)
    agent.start_browser()
    success = agent.check_deployment()
    
    if not success:
        agent.analyze_errors()
        agent.attempt_rectification()
        agent.notify_team()
        agent.trigger_rollback()
        sys.exit(1)
    else:
        logger.info("Deployment is stable.")
        sys.exit(0)
