# Deploy + GitHub Guide (Option 3: PythonAnywhere)

## A) Upload this project to GitHub

Run these commands in your project folder:

```bash
git init
git add .
git commit -m "Initial demand forecast frontend"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If repo already exists locally, only use:

```bash
git add .
git commit -m "Add Flask frontend and deployment files"
git push
```

## B) Deploy on PythonAnywhere

1. Create account at https://www.pythonanywhere.com/
2. Open **Bash Console** on PythonAnywhere.
3. Clone your repo:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

4. Create virtual env and install dependencies:

```bash
mkvirtualenv --python=/usr/bin/python3.12 demand-env
workon demand-env
pip install -r requirements.txt
```

5. In PythonAnywhere dashboard, go to **Web** and create a new web app (Manual config, Python 3.12).
6. Set **Virtualenv** path to your env, e.g. `/home/<username>/.virtualenvs/demand-env`.
7. Edit the WSGI file in Web tab and replace its contents with the content from `pythonanywhere_wsgi.py`.
8. Replace `<username>` in that file with your real PythonAnywhere username.
9. In **Static files** mappings (Web tab), add:
   - URL: `/static/`
   - Directory: `/home/<username>/<your-repo>/static`
10. Click **Reload**.

Your live URL will be:

`https://<username>.pythonanywhere.com`

## C) Notes

- Keep these files in repo root for app startup: `app.py`, `requirements.txt`, `Historical Product Demand.csv`, `demand_forecast_rf_model.pkl`.
- `scikit-learn` is pinned to avoid model version mismatch warnings.
