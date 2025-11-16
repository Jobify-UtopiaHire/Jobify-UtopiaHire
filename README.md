# 🎯 Jobify - AI-Powered Interview Practice Platform

> **Professional Interview Preparation with Advanced AI Technology and Enterprise Security**

[![Security](https://img.shields.io/badge/security-enterprise%20grade-brightgreen)](SECURITY.md)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/react-18.2+-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-latest-009688.svg)](https://fastapi.tiangolo.com/)

An intelligent, secure interview coaching platform that combines CV parsing, real-time AI voice interaction, comprehensive authentication, and personalized feedback to help job seekers excel in their interviews.

---

## 📋 Table of Contents

- [Features](#-features)
- [Security](#-security)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Development](#-development)
- [Deployment](#-deployment)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Features

### 🔐 Enterprise-Grade Security
- **Authentication & Authorization**
  - JWT-based secure sessions (2-hour access, 7-day refresh)
  - OAuth 2.0 (Google & GitHub)
  - Email verification system
  - Strong password requirements
  - Account lockout protection (5 attempts, 15-min lockout)
  
- **Protection Layers**
  - SQL injection prevention (parameterized queries)
  - XSS protection (input sanitization, CSP headers)
  - CSRF protection tokens
  - Rate limiting (per-endpoint throttling)
  - Secure file upload validation
  - Request size limits
  
- **Security Headers**
  - Content Security Policy (CSP)
  - HTTP Strict Transport Security (HSTS)
  - X-Frame-Options, X-Content-Type-Options
  - X-XSS-Protection

### 📄 CV Analysis & Management
- **Multi-Format Support**
  - PDF, DOCX, DOC, TXT, ODT, TEX, HTML, RTF
  - 10MB file size limit
  - Secure file validation and storage
  
- **Intelligent Processing**
  - AI-powered CV parsing
  - Job description matching
  - ATS score calculation
  - Skills gap analysis
  - Personalized recommendations

### 🎤 Real-Time Voice Interviews
- **LiveKit Integration**
  - Real-time voice communication
  - Video preview (optional)
  - Natural AI conversation
  - Low-latency audio processing
  
- **Interview Features**
  - Personalized questions based on CV
  - Adaptive difficulty
  - Real-time feedback
  - Performance scoring
  - Email report delivery

### 🤖 AI-Powered Coaching
- **Google Gemini AI**
  - Context-aware questioning
  - Natural language understanding
  - Behavioral assessment
  - Technical skill evaluation
  
- **Personalization**
  - Industry-specific questions
  - Role-based scenarios
  - Experience level adaptation
  - Cultural fit assessment

### 📊 Performance Analytics
- **Interview Tracking**
  - Score history
  - Performance trends
  - Improvement metrics
  - Session replays
  
- **Skills Assessment**
  - Technical skills evaluation
  - Soft skills analysis
  - Communication effectiveness
  - Areas for improvement

### 🎨 Modern User Experience
- **Responsive Design**
  - Dark theme interface
  - Mobile-friendly layout
  - Accessibility features
  - Smooth animations
  
- **Internationalization**
  - Multi-language support (EN, FR, AR)
  - RTL language support
  - Localized content
  
- **Interactive Elements**
  - 3D shader backgrounds
  - Drag-and-drop file upload
  - Real-time validation
  - Toast notifications

---

## 🔒 Security

This application implements **enterprise-grade security** measures to protect your data:

- ✅ **OWASP Top 10 Protection**
- ✅ **End-to-End Encryption** (in transit)
- ✅ **Secure Password Storage** (bcrypt hashing)
- ✅ **Rate Limiting** (prevents brute force)
- ✅ **Input Validation** (prevents injection attacks)
- ✅ **Secure File Uploads** (magic number validation)
- ✅ **Session Management** (token rotation)
- ✅ **Audit Logging** (security events)

📖 **[Read Full Security Documentation](SECURITY.md)**

---

## 🏗️ Architecture

```
UtopiaHire Platform
│
├── Frontend (React + TypeScript)
│   ├── Vite Build System
│   ├── React Router v6
│   ├── LiveKit Components
│   ├── Tailwind CSS
│   ├── Security Utilities
│   └── i18n Support
│
├── Backend (Python + FastAPI)
│   ├── Authentication API
│   ├── OAuth Integration
│   ├── File Upload Service
│   ├── LiveKit Token Service
│   ├── Interview Agent
│   ├── CV Analysis Engine
│   ├── Skills Gap Analysis
│   └── Security Middleware
│
└── Database (PostgreSQL)
    ├── User Management
    ├── Interview History
    └── Secure Configuration
```

### Technology Stack

#### Backend
- **Python 3.13+**
- **FastAPI** - High-performance async API
- **SQLAlchemy** - ORM with asyncpg
- **PostgreSQL** - Relational database
- **LiveKit** - Real-time communication
- **Google Gemini AI** - Natural language processing
- **Security**: SlowAPI, bleach, bcrypt, python-jose

#### Frontend
- **React 18** with **TypeScript**
- **Vite** - Fast build tool
- **React Router v6** - Client-side routing
- **LiveKit Components React** - Voice/video UI
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Animations
- **Axios** - HTTP client with interceptors
- **i18next** - Internationalization

#### Infrastructure
- **Docker Compose** - Container orchestration
- **PostgreSQL 15** - Database container
- **pgAdmin 4** - Database management UI

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.13+** ([Download](https://www.python.org/downloads/))
- **Node.js 18+** ([Download](https://nodejs.org/))
- **Docker & Docker Compose** ([Download](https://www.docker.com/))
- **Git** ([Download](https://git-scm.com/))
- **MiKTeX**([Download](https://miktex.org/download/))(it should be updated)

### API Keys & Accounts

You'll need accounts and API keys for:

1. **LiveKit** ([Sign up](https://cloud.livekit.io/))
   - WebSocket URL
   - API Key
   - API Secret

2. **Google Gemini AI** ([Get API Key](https://ai.google.dev/))
   - API Key

3. **OAuth (Optional)**
   - Google OAuth ([Console](https://console.cloud.google.com/))
   - GitHub OAuth ([Settings](https://github.com/settings/developers))

4. **Email (Optional)**
   - SMTP server credentials (e.g., Gmail App Password)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anas-dev0/cs-challange.git
cd cs-challange
```

### 2. Database Setup (Docker)

```bash
# Copy environment template
cp .env.docker.example .env

# Edit .env and change default passwords
nano .env

# Start PostgreSQL and pgAdmin
docker-compose up -d

# Verify containers are running
docker-compose ps
```

**Database will be available at:**
- PostgreSQL: `localhost:5432`
- pgAdmin: `http://localhost:5050`

### 3. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env and configure your settings
nano .env
```

**Required Environment Variables:**
```bash
# Generate strong secrets
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Add to .env:
JWT_SECRET=<generated-secret>
REFRESH_SECRET=<generated-secret>
DATABASE_URL=postgresql+asyncpg://postgres:your-password@localhost:5432/utopiahire
GOOGLE_API_KEY=<your-gemini-api-key>
LIVEKIT_API_KEY=<your-livekit-key>
LIVEKIT_API_SECRET=<your-livekit-secret>
LIVEKIT_URL=wss://your-project.livekit.cloud
```

### 4. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Create .env file (optional)
echo "VITE_API_URL=http://localhost:8000" > .env
```

---

## ⚙️ Configuration

### Security Configuration

1. **Change Default Passwords**
   ```bash
   # In .env (root directory for Docker)
   POSTGRES_PASSWORD=<strong-password>
   PGADMIN_PASSWORD=<strong-password>
   
   # In backend/.env
   JWT_SECRET=<32+ character random string>
   REFRESH_SECRET=<32+ character random string>
   SESSION_SECRET=<32+ character random string>
   ```

2. **Configure CORS Origins**
   ```bash
   # backend/.env
   CORS_ORIGINS=http://localhost:5173,http://localhost:3000
   ```

3. **Email Configuration (Optional)**
   ```bash
   # backend/.env
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SENDER_EMAIL=your-email@gmail.com
   SENDER_PASSWORD=your-app-password
   ```

4. **OAuth Setup (Optional)**
   - Configure Google OAuth in [Google Console](https://console.cloud.google.com/)
   - Configure GitHub OAuth in [GitHub Settings](https://github.com/settings/developers)
   - Add credentials to `backend/.env`

### Production Configuration

For production deployment:

1. **Enable HTTPS**
   ```python
   # backend/app/main.py
   # Set secure=True for session cookies
   ```

2. **Update CSP Headers**
   ```python
   # Add your production domain to CSP
   ```

3. **Database SSL**
   ```bash
   DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db?ssl=require
   ```

4. **Rate Limiting**
   - Review and adjust rate limits in `security_middleware.py`

---

## 🎮 Usage

### Starting the Application


```bash
# Terminal 1: Start database
docker-compose up

# Terminal 2: Start backend
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python -m app.main

# Terminal 3 : Start Agent
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
cd agent
python agent.py dev

# Terminal 4: Start frontend
cd frontend
npm run dev
```


### Accessing the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **pgAdmin**: http://localhost:5050

### Using the Platform

#### 1. Create Account
- Navigate to http://localhost:5173
- Click "Sign Up"
- Enter name, email, and strong password
- Verify email (check inbox)

#### 2. Upload CV
- Log in to your account
- Go to "Interview Setup"
- Upload your CV (PDF, DOCX, etc.)
- Enter job description
- Provide interview details

#### 3. Start Interview
- Click "Start Interview"
- Grant microphone access
- Answer AI interviewer's questions
- Toggle camera/mic as needed
- End call when complete

#### 4. Review Results
- Check email for detailed report
- View score and feedback
- Track progress in dashboard

---

## 📚 API Documentation

### Authentication Endpoints

```bash
POST /api/auth/register
POST /api/auth/login
POST /api/auth/verify-email
POST /api/auth/resend-verification
POST /api/auth/refresh
GET  /api/auth/me
```

### Service Endpoints

```bash
POST /api/upload-cv            # JWT required
POST /api/start-session        # JWT required
POST /api/save-interview       # JWT required
```

### OAuth Endpoints

```bash
GET  /api/auth/oauth/google
GET  /api/auth/oauth/google/callback
GET  /api/auth/oauth/github
GET  /api/auth/oauth/github/callback
```

### Interactive API Documentation

Visit http://localhost:8000/docs for:
- Complete API reference
- Try-it-out functionality
- Request/response schemas
- Authentication testing

---

## 🛠️ Development

### Project Structure

```
cs-challange/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── auth_routes.py       # Authentication endpoints
│   │   ├── security.py          # JWT utilities
│   │   ├── security_middleware.py  # Security layers
│   │   ├── models.py            # Database models
│   │   ├── schemas.py           # Pydantic schemas
│   │   └── config.py            # Configuration
│   ├── agent/
│   │   ├── agent.py             # LiveKit interview agent
│   │   ├── cv_parser.py         # CV text extraction
│   │   └── mailer.py            # Email service
│   └── core/
│       ├── ai_analyzer.py       # Gemini AI integration
│       └── skill_extractor.py   # Skills analysis
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Main application
│   │   ├── api.ts               # API client
│   │   ├── components/          # React components
│   │   ├── pages/               # Page components
│   │   ├── utils/               # Utilities
│   │   │   └── security.ts      # Security functions
│   │   └── locales/             # Translations
│   └── public/                  # Static assets
│
├── docker-compose.yml           # Database containers
├── SECURITY.md                  # Security documentation
└── README.md                    # This file
```

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make Changes**
   - Follow existing code style
   - Add comments for complex logic
   - Update tests if applicable

3. **Test Locally**
   ```bash
   # Backend
   cd backend
   python -m pytest
   
   # Frontend
   cd frontend
   npm run lint
   npm run build
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature
   ```

### Code Style

- **Python**: Follow PEP 8
- **TypeScript**: Follow Airbnb style guide
- **Commits**: Use conventional commits

---

## 🚢 Deployment

### Production Checklist

- [ ] Change all default passwords
- [ ] Generate strong secrets (32+ characters)
- [ ] Configure HTTPS/SSL
- [ ] Update CORS origins
- [ ] Enable secure cookies
- [ ] Configure database SSL
- [ ] Set up database backups
- [ ] Configure monitoring
- [ ] Set up logging aggregation
- [ ] Review rate limits
- [ ] Test OAuth callbacks
- [ ] Set up CDN (optional)
- [ ] Configure firewall rules

### Deployment Options

#### Docker Compose (Simple)

```bash
# Build and start all services
docker-compose -f docker-compose.prod.yml up -d
```

#### Cloud Platforms

- **Backend**: Heroku, AWS, DigitalOcean, Railway
- **Frontend**: Vercel, Netlify, AWS S3 + CloudFront
- **Database**: AWS RDS, DigitalOcean Managed DB

### Environment Variables

Set the following in production:

```bash
# Backend
JWT_SECRET=<production-secret>
REFRESH_SECRET=<production-secret>
DATABASE_URL=<production-db-url>
CORS_ORIGINS=https://your-domain.com
FRONTEND_URL=https://your-domain.com

# Frontend
VITE_API_URL=https://api.your-domain.com
```

---

## 🧪 Testing

### Backend Tests

```bash
cd backend

# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=app

# Run specific test
python -m pytest tests/test_auth.py
```

### Frontend Tests

```bash
cd frontend

# Lint
npm run lint

# Build test
npm run build

# Type check
npm run tsc --noEmit
```

### Security Testing

```bash
# Install OWASP ZAP
# Run security scan against localhost:8000

# Test rate limiting
for i in {1..20}; do curl http://localhost:8000/api/auth/login; done

# Test file upload
curl -X POST -F "cv=@malicious.pdf" http://localhost:8000/api/upload-cv
```

---

## 🔧 Troubleshooting

### Common Issues

#### Database Connection Failed

```bash
# Check if PostgreSQL is running
docker-compose ps

# Check logs
docker-compose logs db

# Restart database
docker-compose restart db
```

#### LiveKit Connection Error

```bash
# Verify credentials in backend/.env
LIVEKIT_API_KEY=your_key
LIVEKIT_API_SECRET=your_secret
LIVEKIT_URL=wss://your-project.livekit.cloud

# Check LiveKit dashboard for active rooms
```

#### Frontend Can't Reach Backend

```bash
# Verify backend is running
curl http://localhost:8000/health

# Check CORS configuration
# backend/.env: CORS_ORIGINS=http://localhost:5173
```

#### Email Verification Not Working

```bash
# Check SMTP settings in backend/.env
SMTP_SERVER=smtp.gmail.com
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=<app-password>  # Not your regular password!

# Check backend logs for email errors
```

#### Rate Limit Errors

```bash
# Wait for rate limit window to reset (usually 1 minute)
# Or adjust limits in backend/app/security_middleware.py
```

### Getting Help

1. **Check Logs**
   ```bash
   # Backend logs
   docker-compose logs backend
   
   # Database logs
   docker-compose logs db
   ```

2. **GitHub Issues**: [Create an issue](https://github.com/anas-dev0/cs-challange/issues)

3. **Documentation**: Read [SECURITY.md](SECURITY.md) for security-related issues

---





## 🙏 Acknowledgments

- **LiveKit Team** - Real-time communication platform
- **Google AI** - Gemini language models
- **FastAPI** - Modern Python web framework
- **React Team** - Frontend library
- **Open Source Community** - Various libraries and tools



<div align="center">

**Built with ❤️ for better interview preparation**

</div>
