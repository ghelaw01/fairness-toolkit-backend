# AI Fairness Toolkit - Administrator Guide

**Version:** 1.0

---

## 1. Overview

This guide provides instructions for administrators to set up, maintain, and troubleshoot the AI Fairness Toolkit.

## 2. System Requirements

-   **Backend:** Python 3.11+, Flask, pandas, scikit-learn
-   **Frontend:** Node.js 22+, React 18+, Vite
-   **Deployment:** Render.com (or any platform that supports Python and Node.js)

## 3. Installation & Setup

### 3.1 Backend Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ghelaw01/fairness-toolkit-backend.git
    cd fairness-toolkit-backend
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the backend server:**
    ```bash
    flask run
    ```
    The backend will be available at `http://localhost:5000`.

### 3.2 Frontend Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ghelaw01/fairness-toolkit-frontend.git
    cd fairness-toolkit-frontend
    ```

2.  **Install dependencies:**
    ```bash
    pnpm install
    ```

3.  **Run the frontend development server:**
    ```bash
    pnpm run dev
    ```
    The frontend will be available at `http://localhost:5173`.

## 4. Deployment to Render

### 4.1 Backend Deployment

-   **Service Type:** Web Service
-   **Repository:** `ghelaw01/fairness-toolkit-backend`
-   **Root Directory:** (leave empty)
-   **Build Command:** `pip install -r requirements.txt`
-   **Start Command:** `python main.py`
-   **Environment Variables:**
    -   `PYTHON_VERSION`: `3.11.0`
    -   `PORT`: `5000`

### 4.2 Frontend Deployment

-   **Service Type:** Static Site
-   **Repository:** `ghelaw01/fairness-toolkit-frontend`
-   **Root Directory:** (leave empty)
-   **Build Command:** `pnpm install && pnpm run build`
-   **Publish Directory:** `dist`
-   **Environment Variables:**
    -   `VITE_API_URL`: (your backend URL from Render)

## 5. Maintenance

### 5.1 Audit Logs

-   Audit logs are stored in `backend/logs/audit.jsonl`.
-   Regularly back up these logs for compliance.

### 5.2 Updating Dependencies

-   **Backend:** `pip install --upgrade -r requirements.txt`
-   **Frontend:** `pnpm update`

### 5.3 Troubleshooting

-   **Cannot connect to backend:** Ensure the backend is running and the `VITE_API_URL` on the frontend is correct.
-   **Deployment fails:** Check the build logs on Render for errors. Common issues include dependency conflicts or incorrect build commands.
-   **App is slow on first load:** This is normal for Render’s free tier. Use a service like UptimeRobot to keep it awake or upgrade to a paid plan.

