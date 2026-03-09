# CADfactory — Backend API

FastAPI backend for the CADfactory physics-to-capital compiler.

## Stack
- **FastAPI** — async REST API, auto OpenAPI docs
- **SQLAlchemy + SQLite** — ORM + local database (swap to Postgres for prod)
- **JWT Auth** — bcrypt passwords, Bearer tokens
- **Gemini 2.0 Flash** — AI analysis, key stored server-side only
- **aiofiles** — async file I/O for CAD uploads

---

## Quick Start

### 1. Clone / open in VS Code
```
cd cadfactory
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment
```bash
cp .env.example .env
```

Edit `.env`:
```
SECRET_KEY=any-long-random-string-here
GEMINI_API_KEY=AIza...your-key-from-aistudio.google.com
```

### 5. Run the server
```bash
uvicorn main:app --reload
```

Server starts at: **http://localhost:8000**
Interactive API docs: **http://localhost:8000/docs**

---

## API Overview

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Create account |
| POST | `/auth/login` | Get JWT token |
| GET | `/auth/me` | Current user info |
| DELETE | `/auth/me` | Delete account |

### Files
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/files/upload` | Upload STL / STEP file |
| GET | `/files/` | List your uploaded files |
| GET | `/files/{id}` | Get file metadata |
| DELETE | `/files/{id}` | Delete file |

### Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analysis/run` | Full analysis (geometry + COGS + AI) |
| POST | `/analysis/quick?file_id=X` | Fast preview (no AI, no DB save) |
| GET | `/analysis/reports` | All your saved reports |
| GET | `/analysis/reports/{id}` | Single report |
| DELETE | `/analysis/reports/{id}` | Delete report |

### Reference
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/materials/` | Supported materials |
| GET | `/materials/processes` | Manufacturing processes |
| GET | `/materials/regions` | Manufacturing regions + rates |
| GET | `/materials/tiers` | Volume quantity tiers |

---

## Example Usage (curl)

```bash
# 1. Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"you@email.com","username":"batuhan","password":"mypassword"}'

# 2. Login → get token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"you@email.com","password":"mypassword"}'

# 3. Upload CAD file
curl -X POST http://localhost:8000/files/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@mypart.stl"

# 4. Run full analysis
curl -X POST http://localhost:8000/analysis/run \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"file_id": 1, "material": "Al6061-T6", "process": "CNC_3axis", "use_ai": true}'
```

---

## Supported Materials
- `Al6061-T6` — Aluminium 6061-T6 (default)
- `Al7075-T6` — Aluminium 7075-T6
- `SS304` — Stainless Steel 304
- `Titanium_Grade5` — Ti-6Al-4V
- `PEEK` — PEEK polymer

## Supported Processes
- `CNC_3axis` — 3-Axis CNC Milling (default)
- `CNC_5axis` — 5-Axis CNC Milling
- `Turning` — CNC Turning

---

## Project Structure
```
cadfactory/
├── main.py               # App entry point, CORS, lifespan
├── database.py           # SQLAlchemy engine + session
├── models.py             # ORM models (User, UploadedFile, Report)
├── schemas.py            # Pydantic request/response schemas
├── auth.py               # JWT + bcrypt utilities
├── requirements.txt
├── .env.example
├── routers/
│   ├── auth.py           # Register, login, me
│   ├── files.py          # Upload, list, delete
│   ├── analysis.py       # Run analysis, reports
│   └── materials.py      # Reference data
├── services/
│   ├── geometry.py       # STL + STEP parsers
│   ├── cogs.py           # Parametric COGS engine
│   ├── gemini.py         # Gemini AI (server-side key)
│   └── storage.py        # File I/O
└── uploads/              # CAD files stored here
```

---

## Connecting to the Frontend (cadfactory.html)

Update `cadfactory.html` to point to the backend:
```js
const API = "http://localhost:8000";

// Register / login → store token in memory
// Upload file → POST /files/upload
// Run analysis → POST /analysis/run  (no API key needed in UI anymore)
// Fetch reports → GET /analysis/reports
```

The Gemini key is now **only in `.env`** — the frontend never sees it.
