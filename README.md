<a name="readme-top"></a>

<div align="center">

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]
[![Security][security-shield]](SECURITY.md)
[![Python][python-shield]](https://www.python.org/)
[![React][react-shield]](https://reactjs.org/)

</div>

---

# 🎯 Jobify - UtopiaHire

**Professional Interview Preparation with Advanced AI Technology and Enterprise Security**

<br />
<div align="center">
  <a href="https://github.com/Jobify-UtopiaHire/Jobify-UtopiaHire">
     <img src="https://github.com/Jobify-UtopiaHire/Jobify-UtopiaHire/blob/main/frontend/src/assets/images/logo.png?raw=true" alt="Jobify Logo" width="144" height="144">
  </a>
  <h3>Jobify - UtopiaHire</h3>
  <p align="center">
    <strong>Professional interview preparation powered by advanced AI technology</strong>
    <br />
    <br />
    <a href="https://github.com/Jobify-UtopiaHire/Jobify-UtopiaHire/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/Jobify-UtopiaHire/Jobify-UtopiaHire/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
      </p>
</div>

<br/>

---

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#-features">Features</a></li>
    <li><a href="#-security">Security</a></li>
    <li><a href="#-architecture">Architecture</a></li>
    <li><a href="#-getting-started">Getting Started</a></li>
    <li><a href="#-installation">Installation</a></li>
    <li><a href="#-configuration">Configuration</a></li>
    <li><a href="#-usage">Usage</a></li>
    <li><a href="#-api-documentation">API Documentation</a></li>
    <li><a href="#-development">Development</a></li>
    <li><a href="#-deployment">Deployment</a></li>
    <li><a href="#-testing">Testing</a></li>
    <li><a href="#-troubleshooting">Troubleshooting</a></li>
    <li><a href="#-contributing">Contributing</a></li>
    <li><a href="#-license">License</a></li>
     <li><a href="#-contact">Contact</a></li>
     <li><a href="#-acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## About The Project

**🚀 Jobify - UtopiaHire** is a comprehensive AI-powered interview coaching platform that combines intelligent CV parsing, real-time voice interaction, and enterprise-grade security to help job seekers excel in their interviews. The platform leverages cutting-edge technologies including Google Gemini AI for natural language processing, LiveKit for real-time communication, and FastAPI for high-performance backend services.

Whether you're preparing for your first job interview or looking to sharpen your skills for executive positions, Jobify provides personalized, adaptive interview practice sessions with detailed performance analytics and actionable feedback delivered straight to your inbox.

### 📸 Platform Overview

<div align="center">
  <img src="docs/screenshots/Landing_Page.png" alt="Jobify Landing Page" width="800">
  <p><em>Modern, intuitive landing page with seamless authentication</em></p>
</div>

### 🎯 Key Features

- 🔐 **Enterprise Security**: JWT authentication, OAuth 2.0, rate limiting, XSS/CSRF protection, and comprehensive security headers
- 📄 **Smart CV Analysis**: Multi-format support (PDF, DOCX, TXT, etc.) with AI-powered parsing, ATS scoring, and skills gap analysis
- 🎤 **Real-Time Voice Interviews**: LiveKit-powered voice communication with natural AI conversation and low-latency audio processing
- 🤖 **AI-Powered Coaching**: Context-aware questioning using Google Gemini AI with adaptive difficulty and behavioral assessment
- 📊 **Performance Analytics**: Comprehensive tracking with score history, improvement metrics, and detailed session reports
- 🌐 **Multi-Language Support**: Built-in internationalization with support for English, French, and Arabic (including RTL)
- 🎨 **Modern UI/UX**: Responsive dark theme design with 3D shader backgrounds, drag-and-drop uploads, and smooth animations

### 🎬 Feature Showcase

<div align="center">
  <img src="docs/screenshots/skills-gap-analysis.png" alt="Skills Gap Analysis" width="800">
  <p><em>AI-powered skills gap analysis showing matched strengths and areas for improvement</em></p>
  <br/>
  <img src="docs/screenshots/Cv_reader_and_rewriter.png" alt="CV Reader and Rewriter" width="800">
  <p><em>Intelligent CV parsing and optimization with ATS-friendly suggestions</em></p>
  <br/>
  <img src="docs/screenshots/AI_interviwer.png" alt="AI Interviewer" width="800">
  <p><em>Real-time AI interviewer with voice interaction and adaptive questioning</em></p>
  <br/>
  <img src="docs/screenshots/Job_matcher.png" alt="Job Matcher" width="800">
  <p><em>Smart job matching based on your skills and experience</em></p>
</div>

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

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

### Security Features

**Authentication & Authorization:**

- JWT-based secure sessions (2-hour access, 7-day refresh tokens)
- OAuth 2.0 integration (Google & GitHub)
- Email verification system with resend capability
- Strong password requirements enforcement
- Account lockout protection (5 attempts, 15-minute lockout)

**Protection Layers:**

- SQL injection prevention using parameterized queries
- XSS protection with input sanitization and CSP headers
- CSRF protection tokens for state-changing operations
- Per-endpoint rate limiting and throttling
- Secure file upload validation with magic number checks
- Request size limits to prevent DoS attacks

**Security Headers:**

- Content Security Policy (CSP)
- HTTP Strict Transport Security (HSTS)
- X-Frame-Options, X-Content-Type-Options
- X-XSS-Protection

📖 **[Read Full Security Documentation](https://github.com/Jobify-UtopiaHire/Jobify-UtopiaHire/blob/main/SECURITY.md)**

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 🏗️ Architecture

```
Jobify Platform
│
├── Frontend (React + TypeScript)
│   ├── Vite Build System
│   ├── React Router v6
│   ├── LiveKit Components
│   ├── Tailwind CSS
│   ├── Security Utilities
│   └── i18n Support (EN, FR, AR)
│
├── Backend (Python + FastAPI)
│   ├── Authentication API (JWT + OAuth)
│   ├── File Upload Service
│   ├── LiveKit Token Service
│   ├── Interview Agent
│   ├── CV Analysis Engine
│   ├── Skills Gap Analysis
│   ├── Email Service
│   └── Security Middleware
│
└── Database (PostgreSQL)
    ├── User Management
    ├── Interview History
    └── Secure Configuration
```

### Technology Stack

#### Backend

- **Python 3.13+** - Modern Python runtime
- **FastAPI** - High-performance async API framework
- **SQLAlchemy** - Async ORM with asyncpg driver
- **PostgreSQL** - Relational database
- **LiveKit** - Real-time communication infrastructure
- **Google Gemini AI** - Advanced language models
- **Security**: SlowAPI, bleach, bcrypt, python-jose

#### Frontend

- **React 18** with **TypeScript** - Type-safe UI development
- **Vite** - Lightning-fast build tool
- **React Router v6** - Client-side routing
- **LiveKit Components React** - Pre-built voice/video UI
- **Tailwind CSS** - Utility-first styling framework
- **Framer Motion** - Production-ready animations
- **Axios** - HTTP client with request/response interceptors
- **i18next** - Internationalization framework

#### Infrastructure

- **Docker Compose** - Container orchestration
- **PostgreSQL 15** - Database container
- **pgAdmin 4** - Web-based database management

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## ⚡ Getting Started

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.11+** ([Download](https://www.python.org/downloads/))
- **Node.js 18+** with npm ([Download](https://nodejs.org/))
- **Docker & Docker Compose** ([Download](https://www.docker.com/))
- **Git** ([Download](https://git-scm.com/))
- **MiKTeX** ([Download](https://miktex.org/download/)) - Required for LaTeX CV processing (keep it updated)

### API Keys & Accounts

You'll need to register for the following services and obtain API credentials:

1. **LiveKit** ([Sign up](https://cloud.livekit.io/))

   - WebSocket URL
   - API Key
   - API Secret

2. **Google Gemini AI** ([Get API Key](https://ai.google.dev/))

   - API Key for natural language processing

3. **OAuth Providers (Optional)**

   - Google OAuth ([Console](https://console.cloud.google.com/))
   - GitHub OAuth ([Developer Settings](https://github.com/settings/developers))

4. **Email Service (Optional)**
   - SMTP server credentials (e.g., Gmail App Password)

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 📚 Installation

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Jobify-UtopiaHire/Jobify-UtopiaHire.git

# Navigate to the project directory
cd Jobify-UtopiaHire
```

### Step 2: Database Setup (Docker)

```bash
# Copy the Docker environment template
cp .env.docker.example .env

# Edit the .env file and change default passwords
nano .env  # or use your preferred text editor

# Start PostgreSQL and pgAdmin containers
docker-compose up -d

# Verify containers are running
docker-compose ps
```

**Database will be available at:**

- PostgreSQL: `localhost:5432`
- pgAdmin: `http://localhost:5050`

### Step 3: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create a Python virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Copy backend environment template
cp .env.example .env

# Generate secure secrets
python -c "import secrets; print('JWT_SECRET=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('REFRESH_SECRET=' + secrets.token_urlsafe(32))"

# Edit .env and configure your settings (see Configuration section)
nano .env
```

### Step 4: Frontend Setup

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install Node.js dependencies
npm install

# Create frontend environment file (optional)
echo "VITE_API_URL=http://localhost:8000" > .env
```

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 🪛 Configuration

### Backend Environment Variables

Edit `backend/.env` with the following required configurations:

```env
# Security Secrets (Generate using: python -c "import secrets; print(secrets.token_urlsafe(32))")
JWT_SECRET=your_jwt_secret_here_32plus_chars
REFRESH_SECRET=your_refresh_secret_here_32plus_chars
SESSION_SECRET=your_session_secret_here_32plus_chars

# Database Configuration
DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/utopiahire

# AI Services
GOOGLE_API_KEY=your_gemini_api_key_here

# LiveKit Configuration
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
LIVEKIT_URL=wss://your-project.livekit.cloud

# CORS & Frontend
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
FRONTEND_URL=http://localhost:5173

# Email Configuration (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your_gmail_app_password

# OAuth Configuration (Optional)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
```

### Docker Environment Variables

Edit `.env` (root directory) for Docker Compose:

```env
# PostgreSQL Configuration
POSTGRES_DB=utopiahire
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_strong_password_here

# pgAdmin Configuration
PGADMIN_DEFAULT_EMAIL=admin@utopiahire.com
PGADMIN_DEFAULT_PASSWORD=your_admin_password_here
```

### Security Best Practices

1. **Generate Strong Secrets**: Use cryptographically secure random strings (32+ characters)
2. **Never Commit Secrets**: Keep `.env` files out of version control
3. **Change Default Passwords**: Replace all example passwords with strong, unique values
4. **Use App Passwords**: For Gmail, generate an app-specific password instead of using your account password
5. **Restrict CORS**: Only allow trusted origins in production

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 🎮 Usage

### Starting the Application

You'll need **four separate terminal windows** to run all services:

```bash
# Terminal 1: Start PostgreSQL and pgAdmin
docker-compose up

# Terminal 2: Start Backend API
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python -m app.main

# Terminal 3: Start Interview Agent
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
cd agent
python agent.py dev

# Terminal 4: Start Frontend Development Server
cd frontend
npm run dev
```

### Access Points

- **Frontend Application**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Interactive Swagger UI)
- **Health Check**: http://localhost:8000/health
- **pgAdmin**: http://localhost:5050 (Database management)

### Using the Platform

#### 1. Create Your Account

```bash
# Navigate to the application
Open http://localhost:5173 in your browser

# Register a new account
- Click "Sign Up"
- Enter your name, email, and a strong password
- Verify your email address (check your inbox)
```

#### 2. Upload Your CV

```bash
# Log in to your account
# Navigate to "Interview Setup" page

# Upload your CV
- Drag and drop your CV file, or click to browse
- Supported formats: PDF, DOCX, DOC, TXT, ODT, TEX, HTML, RTF
- Maximum file size: 10MB

# Provide interview context
- Enter the job description
- Add any specific requirements or focus areas
```

#### 3. Start Your Interview

```bash
# Begin the interview session
- Click "Start Interview"
- Grant microphone access when prompted
- (Optional) Enable camera for video preview

# During the interview
- Answer the AI interviewer's questions naturally
- Toggle microphone/camera as needed
- The AI adapts questions based on your CV and responses

# End the session
- Click "End Call" when you're ready to finish
```

#### 4. Review Your Performance

```bash
# Check your results
- Detailed report sent to your email within minutes
- View your interview score and feedback
- Access performance analytics in your dashboard
- Track progress over multiple sessions
```

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 📖 API Documentation

### Authentication Endpoints

| Method | Endpoint                        | Description               | Authentication |
| ------ | ------------------------------- | ------------------------- | -------------- |
| POST   | `/api/auth/register`            | Create new user account   | None           |
| POST   | `/api/auth/login`               | Authenticate user         | None           |
| POST   | `/api/auth/verify-email`        | Verify email address      | None           |
| POST   | `/api/auth/resend-verification` | Resend verification email | None           |
| POST   | `/api/auth/refresh`             | Refresh access token      | Refresh Token  |
| GET    | `/api/auth/me`                  | Get current user info     | JWT Required   |

### Service Endpoints

| Method | Endpoint              | Description             | Authentication |
| ------ | --------------------- | ----------------------- | -------------- |
| POST   | `/api/upload-cv`      | Upload and parse CV     | JWT Required   |
| POST   | `/api/start-session`  | Start interview session | JWT Required   |
| POST   | `/api/save-interview` | Save interview results  | JWT Required   |

### OAuth Endpoints

| Method | Endpoint                          | Description           |
| ------ | --------------------------------- | --------------------- |
| GET    | `/api/auth/oauth/google`          | Initiate Google OAuth |
| GET    | `/api/auth/oauth/google/callback` | Google OAuth callback |
| GET    | `/api/auth/oauth/github`          | Initiate GitHub OAuth |
| GET    | `/api/auth/oauth/github/callback` | GitHub OAuth callback |

### Interactive Documentation

Visit **http://localhost:8000/docs** for:

- Complete API reference with request/response schemas
- Interactive "Try it out" functionality
- Authentication testing interface
- Example requests and responses
- Model schemas and validation rules

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 🛠️ Development

### Project Structure

```
Jobify-UtopiaHire/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI application entry point
│   │   ├── auth_routes.py             # Authentication endpoints
│   │   ├── security.py                # JWT utilities and password hashing
│   │   ├── security_middleware.py     # Security layers and rate limiting
│   │   ├── models.py                  # SQLAlchemy database models
│   │   ├── schemas.py                 # Pydantic validation schemas
│   │   └── config.py                  # Application configuration
│   ├── agent/
│   │   ├── agent.py                   # LiveKit interview agent
│   │   ├── cv_parser.py               # CV text extraction utilities
│   │   └── mailer.py                  # Email service integration
│   ├── core/
│   │   ├── ai_analyzer.py             # Gemini AI integration
│   │   └── skill_extractor.py         # Skills analysis engine
│   ├── requirements.txt               # Python dependencies
│   └── .env.example                   # Environment variables template
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                    # Main React application
│   │   ├── api.ts                     # Axios API client
│   │   ├── components/                # Reusable React components
│   │   ├── pages/                     # Page-level components
│   │   ├── utils/
│   │   │   └── security.ts            # Frontend security utilities
│   │   └── locales/                   # i18n translation files
│   │       ├── en.json
│   │       ├── fr.json
│   │       └── ar.json
│   ├── public/                        # Static assets
│   ├── package.json                   # Node.js dependencies
│   └── vite.config.ts                 # Vite configuration
│
├── docker-compose.yml                 # Database container orchestration
├── .env.docker.example                # Docker environment template
├── SECURITY.md                        # Security documentation
├── LICENSE                            # MIT License
└── README.md                          # This file
```

### Development Workflow

1. **Create Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**

   - Follow existing code style and conventions
   - Add clear comments for complex logic
   - Update documentation if needed
   - Write tests for new features

3. **Test Locally**

   ```bash
   # Backend tests
   cd backend
   python -m pytest

   # Frontend linting and build
   cd frontend
   npm run lint
   npm run build
   ```

4. **Commit Your Changes**

   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   # Then create a PR on GitHub
   ```

### Code Style Guidelines

- **Python**: Follow PEP 8 style guide
- **TypeScript/React**: Follow Airbnb JavaScript/React style guide
- **Commits**: Use conventional commits format (feat:, fix:, docs:, etc.)
- **Comments**: Write clear, concise comments explaining the "why", not the "what"

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 🚢 Deployment

### Production Checklist

Security Configuration:

- [ ] Change all default passwords to strong, unique values
- [ ] Generate cryptographically secure secrets (32+ characters)
- [ ] Configure HTTPS/SSL certificates
- [ ] Update CORS origins to production domains only
- [ ] Enable secure cookies (secure=True, httponly=True)
- [ ] Configure database SSL/TLS connections

Infrastructure:

- [ ] Set up database backups and recovery procedures
- [ ] Configure monitoring and alerting (e.g., Sentry, DataDog)
- [ ] Set up centralized logging (e.g., ELK stack, CloudWatch)
- [ ] Configure CDN for static assets (optional)
- [ ] Set up firewall rules and network security groups

Application:

- [ ] Review and adjust rate limits for production traffic
- [ ] Test OAuth callbacks with production URLs
- [ ] Verify email service configuration
- [ ] Set up health check endpoints
- [ ] Configure proper error handling and logging

### Deployment Options

#### Using Docker Compose (Simple)

```bash
# Build and start all services in production mode
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Cloud Platform Deployments

**Backend Options:**

- Heroku (easy deployment with CLI)
- AWS Elastic Beanstalk or ECS
- DigitalOcean App Platform
- Railway (modern PaaS)
- Google Cloud Run

**Frontend Options:**

- Vercel (recommended for Vite/React)
- Netlify
- AWS S3 + CloudFront
- DigitalOcean Static Sites

**Database Options:**

- AWS RDS PostgreSQL
- DigitalOcean Managed Database
- Heroku Postgres
- Google Cloud SQL

### Production Environment Variables

```env
# Backend (.env)
JWT_SECRET=<production-secure-secret-32plus-chars>
REFRESH_SECRET=<production-secure-secret-32plus-chars>
DATABASE_URL=postgresql+asyncpg://user:pass@prod-host:5432/db?ssl=require
CORS_ORIGINS=https://yourdomain.com
FRONTEND_URL=https://yourdomain.com
LIVEKIT_URL=wss://your-production.livekit.cloud

# Frontend (.env.production)
VITE_API_URL=https://api.yourdomain.com
```

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 🧪 Testing

### Backend Testing

```bash
cd backend

# Run all tests
python -m pytest

# Run tests with coverage report
python -m pytest --cov=app --cov-report=html

# Run specific test file
python -m pytest tests/test_auth.py

# Run tests with verbose output
python -m pytest -v
```

### Frontend Testing

```bash
cd frontend

# Lint TypeScript/React code
npm run lint

# Fix linting issues automatically
npm run lint:fix

# Type checking
npx tsc --noEmit

# Build test (ensures no build errors)
npm run build
```

### Security Testing

```bash
# Test rate limiting (should block after limit)
for i in {1..20}; do curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"test"}'; done

# Test file upload security
curl -X POST -F "cv=@test.pdf" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/upload-cv

# OWASP ZAP Security Scan
# 1. Install OWASP ZAP
# 2. Run automated scan against http://localhost:8000
# 3. Review security vulnerabilities report
```

### Manual Testing Checklist

- [ ] User registration and email verification
- [ ] Login with email/password
- [ ] OAuth login (Google and GitHub)
- [ ] CV upload and parsing
- [ ] Interview session creation
- [ ] Real-time voice communication
- [ ] Performance analytics display
- [ ] Email report delivery
- [ ] Rate limiting behavior
- [ ] Error handling and user feedback

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 🔧 Troubleshooting

### Database Connection Failed

```bash
# Check if PostgreSQL container is running
docker-compose ps

# View database logs
docker-compose logs db

# Restart database service
docker-compose restart db

# Verify connection settings in backend/.env
DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/utopiahire
```

### LiveKit Connection Error

```bash
# Verify LiveKit credentials in backend/.env
LIVEKIT_API_KEY=your_key
LIVEKIT_API_SECRET=your_secret
LIVEKIT_URL=wss://your-project.livekit.cloud

# Check LiveKit dashboard
# - Visit cloud.livekit.io
# - Check active rooms and connection status
# - Verify API key permissions

# Test token generation
cd backend
python -c "from app.config import settings; print(settings.LIVEKIT_API_KEY)"
```

### Frontend Cannot Reach Backend

```bash
# Verify backend is running
curl http://localhost:8000/health
# Should return: {"status": "healthy"}

# Check CORS configuration in backend/.env
CORS_ORIGINS=http://localhost:5173

# Check frontend API URL in frontend/.env
VITE_API_URL=http://localhost:8000

# View browser console for CORS errors
# Open Developer Tools > Console in your browser
```

### Email Verification Not Working

```bash
# Check SMTP configuration in backend/.env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=<app-password>  # NOT your regular Gmail password!

# Generate Gmail App Password:
# 1. Go to Google Account > Security
# 2. Enable 2-Step Verification
# 3. Go to App Passwords
# 4. Generate password for "Mail"
# 5. Use generated password in SENDER_PASSWORD

# Check backend logs for email errors
docker-compose logs backend | grep -i email
```

### Rate Limit Errors (HTTP 429)

```bash
# Wait for rate limit window to reset (typically 60 seconds)

# Or adjust limits in backend/app/security_middleware.py
RATE_LIMITS = {
    "/api/auth/login": "5 per minute",  # Modify as needed
    "/api/auth/register": "3 per minute"
}

# Restart backend after changes
```

### Agent Not Starting

```bash
# Ensure all dependencies are installed
cd backend
pip install -r requirements.txt

# Check if LiveKit agent dependencies are available
python -c "import livekit; print(livekit.__version__)"

# Run agent with verbose logging
cd agent
python agent.py dev --verbose
```

### Getting More Help

1. **Check Application Logs**

   ```bash
   # Backend logs
   docker-compose logs backend -f

   # Database logs
   docker-compose logs db -f

   # All services
   docker-compose logs -f
   ```

2. **Search Existing Issues**: [GitHub Issues](https://github.com/Jobify-UtopiaHire/Jobify-UtopiaHire/issues)

3. **Create New Issue**: [Report Bug](https://github.com/Jobify-UtopiaHire/Jobify-UtopiaHire/issues/new)

4. **Review Documentation**: [SECURITY.md](https://github.com/Jobify-UtopiaHire/Jobify-UtopiaHire/blob/main/SECURITY.md) for security-related issues

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 🤝 Contributing

Contributions are what make the open source community amazing! Any contributions are **greatly appreciated**.

### How to Contribute

1. **Fork the Project**

   ```bash
   # Click the "Fork" button on GitHub
   # Clone your fork
   git clone https://github.com/YOUR_USERNAME/Jobify-UtopiaHire.git
   ```

2. **Create your Feature Branch**

   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Make Your Changes**

   - Write clean, documented code
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation as needed

4. **Commit your Changes**

   ```bash
   git commit -m 'feat: Add some AmazingFeature'
   ```

   Use conventional commit format:

   - `feat:` - New features
   - `fix:` - Bug fixes
   - `docs:` - Documentation changes
   - `style:` - Code style changes
   - `refactor:` - Code refactoring
   - `test:` - Test additions or changes
   - `chore:` - Maintenance tasks

5. **Push to the Branch**

   ```bash
   git push origin feature/AmazingFeature
   ```

6. **Open a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your feature branch
   - Describe your changes clearly
   - Link any related issues

### Contribution Guidelines

- **Code Quality**: Ensure your code passes linting and tests
- **Documentation**: Update README and comments as needed
- **Security**: Never commit secrets, API keys, or sensitive data
- **Testing**: Add tests for new features
- **Respect**: Be kind and professional in all interactions

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 📃 License

Distributed under the MIT License. See `LICENSE` file for more information.

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

## 🙏 Acknowledgments

This project was made possible thanks to these amazing technologies and communities:

- **[LiveKit Team](https://livekit.io/)** - Real-time communication platform that powers our voice interviews
- **[Google AI](https://ai.google.dev/)** - Gemini language models for intelligent conversation
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern, fast web framework for building APIs
- **[React Team](https://react.dev/)** - The library that powers our frontend
- **[Tailwind CSS](https://tailwindcss.com/)** - Utility-first CSS framework
- **[PostgreSQL](https://www.postgresql.org/)** - Robust and reliable database system
- **[Docker](https://www.docker.com/)** - Containerization platform
- **[Vite](https://vitejs.dev/)** - Next generation frontend tooling
- **Open Source Community** - All the amazing libraries and tools that make development easier

Special thanks to all contributors and users who help improve this platform!

<div align="right">
  <a href="#readme-top">
    <img src="https://img.shields.io/badge/Back_to_Top-⬆️-blue?style=for-the-badge" alt="Back to Top">
  </a>
</div>

---

<div align="center">

**Built with ❤️ for better interview preparation**

_Empowering job seekers with AI-powered practice and feedback_

[![GitHub stars](https://img.shields.io/github/stars/Jobify-UtopiaHire/Jobify-UtopiaHire?style=social)](https://github.com/Jobify-UtopiaHire/Jobify-UtopiaHire/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Jobify-UtopiaHire/Jobify-UtopiaHire?style=social)](https://github.com/Jobify-UtopiaHire/Jobify-UtopiaHire/network/members)

</div>

---

[contributors-shield]: https://img.shields.io/github/contributors/Jobify-UtopiaHire/Jobify-UtopiaHire.svg?style=for-the-badge
[contributors-url]: https://github.com/Jobify-UtopiaHire/Jobify-UtopiaHire/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Jobify-UtopiaHire/Jobify-UtopiaHire.svg?style=for-the-badge
[forks-url]: https://github.com/Jobify-UtopiaHire/Jobify-UtopiaHire/network/members
[stars-shield]: https://img.shields.io/github/stars/Jobify-UtopiaHire/Jobify-UtopiaHire.svg?style=for-the-badge
[security-shield]: https://img.shields.io/badge/Security-Enterprise%20Grade-green?style=for-the-badge
[stars-url]: https://github.com/Jobify-UtopiaHire/Jobify-UtopiaHire/stargazers
[python-shield]: https://img.shields.io/badge/Python-3.13%2B-blue?style=for-the-badge&logo=python&logoColor=white
[react-shield]: https://img.shields.io/badge/React-18-blue?style=for-the-badge&logo=react&logoColor=white
[issues-shield]: https://img.shields.io/github/issues/Jobify-UtopiaHire/Jobify-UtopiaHire.svg?style=for-the-badge
[issues-url]: https://github.com/Jobify-UtopiaHire/Jobify-UtopiaHire/issues
[license-shield]: https://img.shields.io/github/license/Jobify-UtopiaHire/Jobify-UtopiaHire.svg?style=for-the-badge
[license-url]: https://github.com/Jobify-UtopiaHire/Jobify-UtopiaHire/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
