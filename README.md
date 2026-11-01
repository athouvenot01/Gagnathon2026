# Hagstofa AI – PxWeb API Backend (Track 2)

This project contains a dynamic **PxWeb API-driven backend** designed for Statistics Iceland (Hagstofa Íslands). Instead of relying on a pre-built static vector database, Track 2 focuses on live querying, automated API request generation, and real-time data retrieval directly from the official PxWeb API endpoints.


## 💡 Project Overview & Architecture

### What it Does
The application implements an intelligent API that enables users to ask questions in English about Icelandic statistics. The system matches queries to statistical datasets, constructs valid PX-Web queries, and formats retrieved data into easy-to-understand answers.

### Architecture & Pipelin
1. **Startup Indexing (`lifespan`)**: The server queries Statistics Iceland's metadata endpoint for each supported table, caching variable codes, labels, and allowed values.
2. **Table Resolution**: User questions are normalized (Icelandic character transliteration, stopword filtering, and synonym resolution). A weighted keyword scoring algorithm identifies the most relevant table.
3. **Query Construction (`build_query`)**: The system parses years, identifies specific dimension values matching the query (e.g., Harpa, Cod), sets appropriate "Total" defaults for unmentioned dimensions, and applies a safety cap (`MAX_CELLS = 200`) to avoid oversized payloads.
4. **Data Retrieval & Answer Generation**: The query is dispatched to the Hagstofa Íslands PX-Web API (`POST`). The resulting JSON payload is decoded and returned as structured, labeled text.

### Supported Tables
Currently, the backend queries data across 6 official Statistics Iceland tables:
- **Fertility 1853–2025** (`MAN05201`)
- **Households: Tenancy status by type of household 2004–2022** (`LIF03215`)
- **Concerts in concert halls 1999–2019** (`MEN05205`)
- **Catch and value by species and fishing areas 1993–2024** (`SJA09001`)
- **Hospital beds 2007–2020** (`HEI04103`)
- **Museum visits by type of museum and region 1995–2023** (`MEN07002`)



## 🛠️ Requirements & Setup (`requirements.txt`)

Create a file named `requirements.txt` in the root of your repository with the following contents:

```
fastapi
uvicorn
requests
pydantic
pandas
```

### System Requirements & Preparation:
1. **Python 3.10+** installed on your machine.
2. An active internet connection to communicate live with the Statistics Iceland PxWeb API (`https://px.hagstofa.is/pxen/api/v1/en/`).

## 📂 GitHub Repository Structure

Your GitHub repository for Track 2 should look like this:

```
hagstofa-ai-track2/
│
├── app.py                      # Main FastAPI application & PxWeb API integration
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation file
├── index.html                  # Frontend
├── examples.jsonl              # File to see how good the classification with the keywords work
└── .gitignore                  # Files/folders to ignore in Git

```

## ⚙️ Step 1: Environment Setup

Open a terminal in your project root directory and run the following commands:

1. **Create a virtual environment:**
   ```
   python -m venv venv
   ```
2. **Activate the virtual environment:**
   * On Mac / Linux:
     ```
     source venv/bin/activate
     ```
   * On Windows (CMD / PowerShell):
     ```
     venv\Scripts\activate
     ```
3. **Install the required packages:**
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 🚀 Step 2: Running the Application

Once your environment is set up and `app.py` is in place, start the FastAPI server:

```
uvicorn app:app --reload
```

* The server will run at: `http://127.0.0.1:8000`
* Interactive API Documentation (Swagger UI): `http://127.0.0.1:8000/docs`
* ReDoc Documentation: `http://127.0.0.1:8000/redoc`



## 🧪 Example Prompts & Testing

1. **Museum Visits**: How many museum visits were recorded in 2021?
2. **Hospital Beds**: How many hospital beds were available in 2020?
3. **Fish Catch Statistics**: What was the total catch of cod in 2022?
4. **Concerts in Harpa**: How many concerts were held in Harpa in 2017?



## 🛑 `.gitignore`

To ensure local binary files, caches, or virtual environments are not tracked by Git, include a `.gitignore` file with:

```
venv/
__pycache__/
*.pyc
.env
```

