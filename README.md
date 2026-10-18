---
title: Optimised Cloud EHR
emoji: 
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# Optimised Cloud EHR: research prototype

Research prototype for the PhD study "Performance optimisation of cloud-based electronic health records using deep learning for disease detection and resource management".

- **Clinician view:** patient lookup by NIN token (HMAC-SHA-256) or probabilistic match, encounter history, new encounters, four-disease risk (hypertension, diabetes, malaria, respiratory infection) from the trained CNN-BiLSTM-Attention model, and attention weights per visit.
- **Administrator view:** LSTM workload forecast on held-out data, a live PSO resource decision against the static reference, and the controller comparison from the simulation study.

All patients in this app are synthetic. No UBTH record is included.

## Secrets (Space settings â†’ Variables and secrets)
- `EHR_USERS`: sign-in accounts, for example `clinician:<password>,admin:<password>`. A user whose name starts with `admin` sees the administrator view.
- `EHR_HMAC_KEY`: any long random string, used as the key for NIN tokens.

## Deploying for free on Hugging Face Spaces
1. Create a free account at huggingface.co and click **New Space**. Choose the **Gradio** SDK and the free **CPU basic** hardware.
2. Upload every file in this folder, keeping the `assets` folder (Files â†’ Add file â†’ Upload files).
3. In Settings â†’ Variables and secrets, add the two secrets above.
4. The Space builds on its own (a few minutes the first time). Sign in with one of the `EHR_USERS` accounts.
5. Put the Space URL into Section 4.6.4 of Chapter Four.

If huggingface.co/join shows a 418 error, the sign-up page has blocked the connection as automated traffic. Turn off any VPN, proxy or ad blocker, then try a private window, another browser, or another network (for example phone data), and try again later.

## Alternative free host: Render
The app needs no TensorFlow (the trained network runs in NumPy, `npmodel.py`) and uses under 300 MB of memory, so it fits Render's free web-service tier.
1. Put this folder in a GitHub repository (github.com â†’ New repository â†’ upload the files, keeping `assets`).
2. On render.com, sign up with GitHub and choose **New â†’ Web Service**, then pick the repository.
3. Set Language **Python 3**, Build command `pip install -r requirements.txt`, Start command `python app.py`, Instance type **Free**.
4. Under Environment, add `EHR_USERS`, `EHR_HMAC_KEY` and `PYTHON_VERSION` = `3.11.9`.
5. Deploy. The app is served at the `onrender.com` address Render shows. A free service sleeps when idle, so the first visit after a pause can take about a minute.

