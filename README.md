# Sales Prediction & Data Visualization - v2

Run:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
Open http://localhost:5000

Deploying to Render (recommended quick path)
-------------------------------------------

1. Commit and push this repository to GitHub (or your preferred Git host).
2. Sign in to https://render.com and create a new Web Service.
3. Connect your GitHub account and pick the repository containing this project.
4. Configure the service:
	- Build Command: pip install -r requirements-render.txt
	- Start Command: gunicorn -w 4 -b 0.0.0.0:$PORT app:app
	- Environment: set `FLASK_ENV=production` (Render sets sensible defaults but set this for clarity)
5. Choose an instance plan (free plans can work for light usage).
6. Create the service and watch the build logs.

Notes:
- The file `requirements-render.txt` is a trimmed dependency list that omits heavy optional packages (for example `xgboost` and `pmdarima`) to reduce build time and memory usage. If you need those features, use `requirements.txt` instead but expect longer builds and larger runtime memory.
- Render's filesystem is ephemeral. Session files written to `sessions/` will not persist across automatic redeploys or machine restarts. For durable storage, configure an external storage (S3, database, or persistent disk) and update `app.py` to use that storage for session artifacts.

A Dockerfile has been added to the repository for reproducible builds. Below are quick Docker build and run instructions you can use locally or in CI. These commands are tailored for PowerShell on Windows.

Build (uses the trimmed `requirements-render.txt` by default):

```powershell
docker build -t sales-pred-app .
```

If you want to build with the full `requirements.txt` instead, run:

```powershell
docker build --build-arg REQ=requirements.txt -t sales-pred-app:full .
```

Run:

```powershell
docker run -p 5000:5000 --rm -e FLASK_ENV=production sales-pred-app
```

Notes on Docker:
- The included `Dockerfile` is based on `python:3.11-slim` and installs dependencies from the chosen requirements file. It exposes port 5000 to match the app's Gunicorn start command.
- The `.dockerignore` excludes `sessions/` and other large or sensitive files to keep the image small. If you require session persistence, mount a host directory or use cloud storage (S3) and update `app.py` accordingly.

If you'd like, I can also add a multi-stage Dockerfile that builds wheels for optional heavy dependencies (xgboost/pmdarima) to speed up deploys, or create a GitHub Actions workflow to build and push container images automatically.
