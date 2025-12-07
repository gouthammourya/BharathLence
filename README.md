# BharatLens – Hugging Face Bus Assistant (Flask Web App)

BharatLens is a web application built with **Python + Flask** that uses **free Hugging Face Inference API** to:

- Read a **bus board image** using OCR (vision model)
- Detect the **route number** and basic details using an LLM
- Look up that route in a **local bus dataset** (JSON) and expose it via a small API
- Explain the route in **English, Hindi, or Kannada** using an instruction-tuned LLM
- Generate **text-to-speech (TTS)** audio for the explanation
- Log each scan to a small **SQLite database**

This version uses **only Hugging Face APIs**, no OpenAI.

---

## 1. Project Structure

```text
bharatlens_hf_web/
├── app.py
├── requirements.txt
├── README.md
├── bharatlens.db          # created automatically on first run
├── data/
│   └── bus_routes.json    # pre-built bus dataset (sample)
├── static/
│   ├── css/
│   │   └── style.css
│   └── audio/             # generated TTS files
└── templates/
    ├── index.html
    ├── result.html
    └── history.html
```

---

## 2. Prerequisites

Install:

1. **Python** 3.10+  
2. **pip**  
3. **Git**  
4. **VS Code** or any editor  
5. A **browser** (Chrome/Edge)  
6. A **Hugging Face account + access token** (free tier is enough)

Create your Hugging Face token:

1. Go to Hugging Face website
2. Sign in → click on your avatar → **Settings**
3. Open **Access Tokens**
4. Click **New token**, give a name like `bharatlens`, select **read** role
5. Copy the token (starts with `hf_...`)

---

## 3. Setup – Step by Step

### 3.1. Get the project

If this project is in GitHub:

```bash
git clone https://github.com/<your-username>/bharatlens_hf_web.git
cd bharatlens_hf_web
```

If you downloaded a ZIP:
- Unzip it
- Open a terminal in the unzipped folder

### 3.2. Create and activate a virtual environment

**Windows (PowerShell):**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### 3.3. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs:

- `Flask` – web framework  
- `python-dotenv` – load `.env`  
- `requests` – call Hugging Face HTTP APIs

---

## 4. Configure Hugging Face Token

Create a `.env` file in the project root:

```text
HF_API_TOKEN=your_huggingface_token_here
```

The app uses this token to call these models:

- **OCR**: `microsoft/trocr-small-printed` (image → text)  
- **LLM**: `mistralai/Mistral-7B-Instruct-v0.2` (text generation, reasoning, translation help)  
- **TTS**:  
  - English: `facebook/mms-tts-eng`  
  - Hindi: `facebook/mms-tts-hin`  
  - Kannada: `facebook/mms-tts-kan`  

You can change the model IDs in `app.py` if needed.

---

## 5. Run the Application

With venv active and `.env` created:

```bash
python app.py
```

You should see:

```text
 * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
```

Open in your browser:

- http://127.0.0.1:5000

---

## 6. Using the App

### 6.1. Home page (`/`)

- Upload a **bus board image**
- Select **output language**:
  - English (en)
  - Hindi (hi)
  - Kannada (kn)
- Click **“Analyze with Hugging Face”**

### 6.2. Backend steps

1. **OCR (vision)**  
   - `hf_ocr_image()` sends the image bytes to Hugging Face OCR model  
   - Returns `raw_text`

2. **Route understanding (LLM)**  
   - `analyze_image_and_extract()`:
     - Sends a prompt + `raw_text` to `mistralai/Mistral-7B-Instruct-v0.2`
     - Asks the model to return **strict JSON** with:
       - `raw_text`, `route_no`, `city`, `source`, `destination`
     - If `route_no` missing, we use a **regex fallback** to pick something like `356D`, `218`, etc.

3. **Dataset lookup**  
   - `lookup_bus_route(route_no)` looks into `data/bus_routes.json`  
   - Returns structured route info, or an error message if not found

4. **Explanation (LLM)**  
   - `format_bus_response()` builds a base English description and asks the LLM to:
     - Rewrite it in **English / Hindi / Kannada**
     - Use 2–3 simple sentences
     - Avoid technical language

5. **TTS (audio)**  
   - `hf_tts()` calls:
     - `facebook/mms-tts-eng` for English  
     - `facebook/mms-tts-hin` for Hindi  
     - `facebook/mms-tts-kan` for Kannada  
   - Saves MP3 file in `static/audio/`

6. **Logging**  
   - Each run inserts a row into the `scan_logs` table in `bharatlens.db`

### 6.3. Result page (`/analyze`)

Shows:

- Raw OCR text
- Dataset route details (if route found)
- AI explanation in the chosen language
- Audio playback (TTS MP3)

### 6.4. History page (`/history`)

Shows the last 20 scans from SQLite:

- Time (UTC)
- Route number
- City
- Language
- Short snippet of the final response

---

## 7. Dataset and API

### 7.3. Live BMTC ETA API (Real Public API)

For Bengaluru routes operated by **BMTC**, the app now also calls a **real public (but unofficial) API** to fetch live ETA:

- Endpoint: `http://bmtcmob.hostg.in/api/itsroutewise/details`
- Payload example:
  ```json
  {
    "direction": "1",
    "routeNO": "356D"
  }
  ```

The Flask helper `fetch_bmtc_eta()` (in `app.py`) uses this API and, if it finds a value, adds:

- `eta_minutes` → approximate minutes until the next bus

This value is then:

- Shown on the **Result** page (*Next Bus ETA (Live)*)
- Mentioned in the AI explanation sentence.

> Note: This is an **unofficial** API reverse-engineered from BMTC's app and may change or break at any time. Use responsibly and avoid sending too many requests.


### 7.1. Local bus dataset

`data/bus_routes.json` contains sample routes (BMTC, KSRTC, TSRTC). You can extend it by adding new JSON objects for more routes.

Example record:

```json
{
  "route_no": "356D",
  "operator": "BMTC",
  "city": "Bengaluru",
  "from": "Majestic",
  "to": "Electronic City",
  "first_bus": "05:15",
  "last_bus": "22:30",
  "frequency_minutes": 12,
  "stops": ["Majestic", "Corporation", "Shantinagar", "Silk Board", "Electronic City"]
}
```

### 7.2. Internal JSON API

The app exposes:

```http
GET /api/bus/<route_no>
```

Example:

```http
GET /api/bus/356D
```

Response:

```json
{
  "route_no": "356D",
  "operator": "BMTC",
  "city": "Bengaluru",
  "from": "Majestic",
  "to": "Electronic City",
  "first_bus": "05:15",
  "last_bus": "22:30",
  "frequency_minutes": 12,
  "stops": [...]
}
```

Use **browser**, **Postman**, or `curl` to test.

---

## 8. Database

`bharatlens.db` (auto-created) has table:

```sql
CREATE TABLE scan_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    route_no TEXT,
    language TEXT,
    city TEXT,
    raw_text TEXT,
    final_response TEXT
);
```

Every `/analyze` call inserts a row.

---

## 9. Push to GitHub (for Buildathon)

From inside `bharatlens_hf_web`:

```bash
git init
git add .
git commit -m "Initial BharatLens Hugging Face web app"
```

Create a repo on GitHub, then:

```bash
git remote add origin https://github.com/<your-username>/bharatlens_hf_web.git
git branch -M main
git push -u origin main
```

Now judges/teammates can clone and run it.

---

## 10. Demo Checklist

- [ ] Hugging Face token configured in `.env`  
- [ ] `python app.py` runs without errors  
- [ ] You have 3–4 test bus-board images ready  
- [ ] Upload image → see OCR → route details → explanation in selected language → audio  
- [ ] History page shows previous scans  
- [ ] README updated with your GitHub URL and screenshots  

---

## 11. Possible Extensions

- Replace/augment `data/bus_routes.json` with **real-time open data APIs** (BMTC, KSRTC, etc.)
- Add **live camera capture** in browser (for mobile users)
- Add user accounts & “favourite routes”
- Add support for more Indian languages with MMS TTS

---

Happy building with **BharatLens + Hugging Face**! 🚍🤖
