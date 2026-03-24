# 💬 Release Notes Chat — Live API + RAG Assistant

An intelligent software update assistant that combines **live vendor data (GitHub, CISA, APIs)** with **Retrieval-Augmented Generation (RAG)** to deliver accurate, real-time, and contextual answers.

---

## 🚀 Features

- 🔍 **Live Data Integration**
  - GitHub Releases API
  - CISA Known Exploited Vulnerabilities (KEV)
  - OS & Reddit APIs
  - Atom/RSS feeds

- 🧠 **RAG (Retrieval-Augmented Generation)**
  - FAISS vector search
  - Sentence Transformers embeddings
  - Context retrieval from local dataset (`SoftwareUpdateSurvey.csv`)

- ⚡ **Smart Query Routing**
  - Automatically detects vendor (Python, Kubernetes, Redis, etc.)
  - Routes to correct data source
  - Supports natural language queries

- ⏱️ **Time-Aware Filtering**
  - Queries like:
    - "last month"
    - "October 2025"
    - "last quarter"
    - "week 42 of 2025"

- 🔄 **Smart Fallback System**
  - If exact results are not found → returns closest matches
  - Avoids empty or misleading responses

- 💬 **LLM-Powered Summarization**
  - Uses Gemini (via LangChain)
  - Combines live + RAG data into clean answers

- 🎨 **Modern Chat UI**
  - Inspired by ChatGPT / Perplexity
  - Inline source links
  - Response timing (Total / Live / RAG)

---

## 🏗️ Architecture

<img width="252" height="220" alt="image" src="https://github.com/user-attachments/assets/b726308e-9da4-4eeb-964b-03fc75cd58d7" />

---

## 📦 Tech Stack

- **Frontend/UI:** Streamlit  
- **LLM:** Google Gemini (via LangChain)  
- **Vector DB:** FAISS  
- **Embeddings:** Sentence Transformers  
- **Data Sources:**
  - GitHub Releases API
  - CISA KEV Feed
  - Custom APIs (OS + Reddit)
- **Backend Logic:** Python  

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

## Install dependencies

**pip install -r requirements.txt

## Add environment variables

**Create a .env file:
**GOOGLE_API_KEY=your_api_key_here

## Run the app

**streamlit run app.py

🧪 Example Queries

Try these:

🔍 Releases
	•	“Grafana releases in 2025”
	•	“Python versions released in 2024”
	•	“Redis updates this year”

⏱️ Time-based
	•	“Node.js patches September 2025”
	•	“Linux kernel updates last month”
	•	“TensorFlow releases last quarter”

🔐 Security
	•	“CVEs in October 2025”
	•	“latest vulnerabilities in Linux”
	•	“critical security issues this month”

🤯 Complex queries
	•	“Kubernetes updates and vulnerabilities last month”
	•	“Docker releases and user issues recently”
	•	“TensorFlow updates and security fixes”

⸻

🧠 How It Works
	1.	Understands your query
	•	Detects vendor + time range
	2.	Fetches live data
	•	GitHub / CISA / APIs
	3.	Retrieves context (RAG)
	•	From local dataset + embeddings
	4.	Combines everything
	•	Gemini generates final answer
	5.	Displays results
	•	Clean UI + clickable sources

⸻

📊 Performance

Each response includes:
	•	⏱️ Total Time
	•	⚡ Live API Time
	•	🧠 RAG Retrieval Time

⸻

🔐 Notes
	•	.env file is required (API key not included)
	•	Large vector store is generated locally
	•	Cached API responses improve performance

⸻

🎯 Use Cases
	•	Software release tracking
	•	Security vulnerability monitoring
	•	Developer research assistant
	•	DevOps / SRE tooling
	•	AI-powered tech insights

⸻

🚀 Future Improvements
	•	Multi-step reasoning agents
	•	Streaming responses
	•	More vendor integrations
	•	UI enhancements (charts, timelines)
	•	Deployment (Docker + cloud)

⸻

👨‍💻 Author

Mohammed Fahad
MS Computer Science — University of the Pacific



