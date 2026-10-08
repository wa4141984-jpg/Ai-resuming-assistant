# Ai-resuming-assistant
# 📄 AI Resume ATS Checker

A Streamlit app that scores a resume for Applicant Tracking System (ATS) friendliness and gives concrete, prioritised suggestions to improve it. Powered by Google Gemini Flash.

## Features

- Upload a resume as **PDF, DOCX or TXT**
- **ATS score (0-100)** with a breakdown: keywords, formatting, experience impact, education, readability
- Strengths, weaknesses and **missing keywords**
- Prioritised **improvement suggestions** with rewritten example lines
- Optional **job description** input for a targeted match score
- Instant rule-based checks (email, phone, sections, metrics, length)
- Download the full report as JSON

## Project structure

```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Getting started

### 1. Get a Gemini API key
Create a free key at <https://aistudio.google.com/apikey>.

### 2. Install
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your API key
Create `.streamlit/secrets.toml` (this file must **never** be committed):
```toml
GEMINI_API_KEY = "your-key-here"
# Optional: pin a specific model (default is the "gemini-flash-latest" alias)
# GEMINI_MODEL = "gemini-2.5-flash"
```
Alternatively, set the `GEMINI_API_KEY` environment variable, or just paste the key into the app's sidebar.

### 4. Run
```bash
streamlit run app.py
```
Open <http://localhost:8501>.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Notes and limitations

- Scanned/image-only resumes can't be read (ATS systems can't read them either). Use a text-based PDF or DOCX.
- The score is an AI estimate, not the output of any specific commercial ATS. Use it as guidance.
- Resumes are sent to the Gemini API for analysis; the app itself stores nothing.
- Free-tier Gemini keys have rate limits. If you see a quota error, wait a minute and retry.

## Troubleshooting

| Problem | Fix |
|---|---|
| `404 model not found` | Set `GEMINI_MODEL` to a currently available model name. |
| `API key not valid` | Re-check the key in secrets or the sidebar. |
| "Very little text extracted" | Your PDF is probably an image scan; export a text-based version. |
