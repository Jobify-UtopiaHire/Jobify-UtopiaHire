# Backend Architecture Documentation

## Overview

This document describes the modular architecture of the UtopiaHire backend application. The structure follows best practices for maintainability, security, and scalability.

## Design Principles

### 1. **Modular Architecture**
Each module has a single, well-defined responsibility:
- **Separation of Concerns**: Business logic, data access, and API layers are clearly separated
- **High Cohesion**: Related functionality is grouped together
- **Loose Coupling**: Modules interact through well-defined interfaces

### 2. **Security by Design**
Security is built into every layer:
- **Centralized Security**: All security features in dedicated modules
- **Input Validation**: Multiple layers of validation (client, API, database)
- **Defense in Depth**: Multiple security controls at each layer
- **Principle of Least Privilege**: Minimal permissions for each component

### 3. **Privacy by Design**
User privacy is protected by default:
- **Data Minimization**: Only necessary data is collected and stored
- **Secure Storage**: Passwords hashed, sensitive data encrypted
- **Access Control**: JWT-based authentication with refresh tokens
- **Audit Logging**: Security events are logged for accountability

### 4. **User-Centric Design**
The architecture supports excellent user experience:
- **Fast Response Times**: Async/await for non-blocking I/O
- **Clear Error Messages**: User-friendly error responses
- **Comprehensive API**: RESTful design with OpenAPI documentation
- **Reliability**: Error handling and graceful degradation

## Directory Structure

```
backend/
├── agent/                          # Interview Agent (runs separately)
│   ├── __init__.py
│   ├── agent.py                   # LiveKit voice interview agent
│   ├── cv_parser.py               # CV parsing for agent
│   ├── mailer.py                  # Email service for interview reports
│   ├── prompts.py                 # AI prompt templates
│   └── test_prompts.py            # Prompt testing utilities
│
├── app/                           # Main Application
│   ├── main.py                    # FastAPI application entry point
│   │
│   ├── config/                    # Configuration Management
│   │   ├── __init__.py
│   │   └── settings.py            # Application settings (env vars)
│   │
│   ├── database/                  # Database Layer
│   │   ├── __init__.py
│   │   └── connection.py          # SQLAlchemy async engine & session
│   │
│   ├── models/                    # Data Models (ORM)
│   │   ├── __init__.py
│   │   └── user.py                # User and Interview models
│   │
│   ├── schemas/                   # API Schemas (Pydantic)
│   │   ├── __init__.py
│   │   └── auth_schemas.py        # Request/response schemas
│   │
│   ├── routes/                    # API Endpoints (Controllers)
│   │   ├── __init__.py
│   │   ├── auth.py                # Authentication endpoints
│   │   ├── oauth.py               # OAuth (Google, GitHub)
│   │   ├── service.py             # Interview services
│   │   ├── job.py                 # Job search and scraping
│   │   └── cv_tools.py            # CV analysis and generation
│   │
│   ├── services/                  # Business Logic Layer
│   │   ├── __init__.py
│   │   └── gemini_service.py      # AI analysis service
│   │
│   ├── middleware/                # Middleware Components
│   │   ├── __init__.py
│   │   ├── logging.py             # Request logging
│   │   └── security.py            # Security middleware (rate limit, headers, etc.)
│   │
│   ├── utils/                     # Utility Functions
│   │   ├── __init__.py
│   │   └── security_utils.py      # JWT, password hashing, validation
│   │
│   └── parsers/                   # Document Processing
│       ├── __init__.py
│       ├── cv_parser.py           # Multi-format CV extraction
│       ├── cv_structure_parser.py # Structured CV parsing with AI
│       └── latex_generator.py     # LaTeX CV generation
│
├── api/                           # Skills Gap Analysis API
│   ├── __init__.py
│   └── analysis.py                # Skills gap analysis endpoints
│
├── core/                          # Core AI/ML Components
│   ├── __init__.py
│   ├── ai_analyzer.py             # Gemini AI integration
│   ├── data_loader.py             # ESCO skills data loader
│   └── skill_extractor.py         # GLiNER skill extraction
│
├── data/                          # Static Data Files
│   ├── skills_en.csv              # ESCO skills database
│   └── usa_job_posting_dataset.csv
│
├── cv_uploads/                    # User-uploaded CVs (temp storage)
│
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
├── main.py                        # Legacy Skills Gap API entry (deprecated)
└── README.md                      # Backend documentation
```

## Module Descriptions

### 🎯 Core Application (`app/`)

#### `main.py`
- **Purpose**: Application entry point and configuration
- **Responsibilities**:
  - FastAPI app initialization
  - Middleware registration
  - Router inclusion
  - Lifespan management (startup/shutdown)
  - Health check endpoint

#### `config/`
- **Purpose**: Centralized configuration management
- **Key Files**:
  - `settings.py`: Environment variables, secrets, and application settings
- **Security**: Uses Pydantic for type-safe configuration validation

#### `database/`
- **Purpose**: Database connection and session management
- **Key Files**:
  - `connection.py`: SQLAlchemy async engine, Base class, session factory
- **Features**:
  - Async database operations
  - Connection pooling
  - Dependency injection for sessions

#### `models/`
- **Purpose**: SQLAlchemy ORM models
- **Key Files**:
  - `user.py`: User and Interview models
- **Design**:
  - Clear relationships between entities
  - Type hints for better IDE support
  - Cascade delete for data integrity

#### `schemas/`
- **Purpose**: Pydantic models for request/response validation
- **Key Files**:
  - `auth_schemas.py`: Authentication-related schemas
- **Benefits**:
  - Automatic validation
  - API documentation generation
  - Type safety

#### `routes/`
- **Purpose**: API endpoint definitions (controllers)
- **Key Files**:
  - `auth.py`: Registration, login, email verification
  - `oauth.py`: OAuth flows (Google, GitHub)
  - `service.py`: CV upload, interview session management
  - `job.py`: Job search and LinkedIn scraping
  - `cv_tools.py`: CV analysis, improvement, and generation
- **Design**:
  - RESTful API design
  - Dependency injection for services
  - Comprehensive error handling

#### `services/`
- **Purpose**: Business logic layer
- **Key Files**:
  - `gemini_service.py`: AI-powered CV analysis
- **Design**:
  - Reusable business logic
  - Independent of web framework
  - Easy to test

#### `middleware/`
- **Purpose**: Request/response processing pipeline
- **Key Files**:
  - `logging.py`: Request/response logging
  - `security.py`: Rate limiting, security headers, input sanitization
- **Security Features**:
  - Rate limiting per endpoint
  - XSS protection
  - SQL injection prevention
  - CSRF protection
  - Security headers (CSP, HSTS, etc.)

#### `utils/`
- **Purpose**: Utility functions and helpers
- **Key Files**:
  - `security_utils.py`: JWT tokens, password hashing, authentication
- **Features**:
  - Password strength validation
  - Token creation and verification
  - User authentication dependency

#### `parsers/`
- **Purpose**: Document processing and generation
- **Key Files**:
  - `cv_parser.py`: Extract text from PDF, DOCX, TXT
  - `cv_structure_parser.py`: Parse CV into structured data using AI
  - `latex_generator.py`: Generate professional LaTeX CVs
- **Capabilities**:
  - Multi-format support
  - AI-powered parsing
  - Professional document generation

### 🤖 Interview Agent (`agent/`)

**Purpose**: Runs as a separate process for real-time voice interviews

- `agent.py`: LiveKit voice interview agent
- `cv_parser.py`: CV parsing for interview preparation
- `mailer.py`: Email interview reports to users
- `prompts.py`: AI prompt templates for interviews

**Architecture**: Decoupled from main app for:
- Independent scaling
- Resource isolation
- Separate deployment
- Terminal management

### 📊 Skills Gap Analysis (`api/`, `core/`)

**Purpose**: AI-powered skills gap analysis and job matching

- `api/analysis.py`: Skills analysis endpoints
- `core/ai_analyzer.py`: Gemini AI integration
- `core/data_loader.py`: ESCO skills database
- `core/skill_extractor.py`: GLiNER model for skill extraction

**Features**:
- Extract skills from CVs
- Match against job market data
- Generate personalized recommendations
- ATS score calculation

## Security Architecture

### 1. **Authentication & Authorization**

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ 1. Login (email + password)
       ▼
┌─────────────────────────────────────────┐
│  routes/auth.py                         │
│  ┌───────────────────────────────────┐  │
│  │ 1. Validate input                 │  │
│  │ 2. Check account lockout          │  │
│  │ 3. Verify password                │  │
│  │ 4. Generate JWT tokens            │  │
│  └───────────────────────────────────┘  │
└──────┬──────────────────────────────────┘
       │ 2. Return tokens
       ▼
┌─────────────┐
│   Client    │
│  (stores    │
│   tokens)   │
└──────┬──────┘
       │ 3. API Request + Bearer Token
       ▼
┌─────────────────────────────────────────┐
│  middleware/security.py                 │
│  ┌───────────────────────────────────┐  │
│  │ 1. Rate limiting                  │  │
│  │ 2. Input sanitization             │  │
│  │ 3. Security headers               │  │
│  └───────────────────────────────────┘  │
└──────┬──────────────────────────────────┘
       │ 4. Verify JWT
       ▼
┌─────────────────────────────────────────┐
│  utils/security_utils.py                │
│  ┌───────────────────────────────────┐  │
│  │ 1. Decode token                   │  │
│  │ 2. Verify signature               │  │
│  │ 3. Check expiration               │  │
│  │ 4. Load user from DB              │  │
│  └───────────────────────────────────┘  │
└──────┬──────────────────────────────────┘
       │ 5. Inject current_user
       ▼
┌─────────────────────────────────────────┐
│  Route Handler                          │
│  (has access to authenticated user)     │
└─────────────────────────────────────────┘
```

### 2. **Input Validation Pipeline**

```
Client Input
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Client-side Validation          │
│    (frontend/src/utils/security.ts)│
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ 2. Pydantic Schema Validation       │
│    (schemas/)                       │
│    - Type checking                  │
│    - Format validation              │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ 3. Middleware Sanitization          │
│    (middleware/security.py)         │
│    - HTML sanitization              │
│    - SQL injection detection        │
│    - Email format validation        │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ 4. Business Logic Validation        │
│    (routes/)                        │
│    - Business rules                 │
│    - Database constraints           │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ 5. Database Layer                   │
│    (models/)                        │
│    - Parameterized queries          │
│    - Type-safe operations           │
└─────────────────────────────────────┘
```

### 3. **Security Layers**

| Layer | Component | Protection |
|-------|-----------|------------|
| **Network** | CORS, HTTPS | Origin validation, encrypted transport |
| **Application** | Middleware | Rate limiting, size limits, security headers |
| **Input** | Validators | Sanitization, format validation, type checking |
| **Authentication** | JWT | Signed tokens, expiration, refresh rotation |
| **Authorization** | Dependencies | Role-based access, user verification |
| **Data** | ORM | Parameterized queries, type safety |
| **Storage** | Hashing | bcrypt passwords, secure file storage |

## API Design

### RESTful Endpoints

```
Authentication
├── POST   /api/auth/register        - Create new account
├── POST   /api/auth/login           - Login with credentials
├── POST   /api/auth/refresh         - Refresh access token
├── POST   /api/auth/verify-email    - Verify email address
├── POST   /api/auth/resend-verification - Resend verification email
└── GET    /api/auth/me              - Get current user info

OAuth
├── GET    /api/auth/oauth/google    - Initiate Google OAuth
├── GET    /api/auth/oauth/google/callback - Google OAuth callback
├── GET    /api/auth/oauth/github    - Initiate GitHub OAuth
└── GET    /api/auth/oauth/github/callback - GitHub OAuth callback

Services
├── POST   /api/upload-cv            - Upload CV file
├── POST   /api/start-session        - Start interview session
└── POST   /api/save-interview       - Save interview results

Jobs
├── POST   /api/search-jobs          - Search jobs (Region 8)
└── GET    /api/jobs/{id}            - Get job details

CV Tools
├── POST   /api/analyze-structured   - Analyze CV with AI
├── POST   /api/apply-suggestion     - Apply AI suggestion to CV
└── POST   /api/generate-latex       - Generate LaTeX CV

Skills Gap Analysis
├── POST   /api/analyze              - Analyze skills gap
└── GET    /api/skills               - Get skills database
```

## Data Flow

### Example: Interview Session

```
1. User uploads CV
   └─> POST /api/upload-cv
       ├─> routes/service.py (validate file)
       ├─> middleware/security.py (check file type, size)
       ├─> parsers/cv_parser.py (extract text)
       └─> Save to cv_uploads/

2. User starts interview session
   └─> POST /api/start-session
       ├─> routes/service.py (create LiveKit token)
       ├─> utils/security_utils.py (authenticate user)
       ├─> models/user.py (load user data)
       └─> Return LiveKit token

3. Agent processes interview
   └─> agent/agent.py
       ├─> Load CV from session_data
       ├─> agent/prompts.py (generate questions)
       ├─> core/ai_analyzer.py (Gemini AI)
       └─> Conduct real-time interview

4. Save interview results
   └─> POST /api/save-interview
       ├─> routes/service.py (save to database)
       ├─> models/user.py (create Interview record)
       ├─> agent/mailer.py (send email report)
       └─> Return success
```

## Development Guidelines

### 1. **Adding a New Feature**

**Example**: Add a new authentication provider (LinkedIn)

1. **Update models** (if needed)
   ```python
   # app/models/user.py
   linkedin_id: Mapped[str] = mapped_column(String(255), nullable=True)
   ```

2. **Add configuration**
   ```python
   # app/config/settings.py
   linkedin_client_id: str = Field("", alias="LINKEDIN_CLIENT_ID")
   linkedin_client_secret: str = Field("", alias="LINKEDIN_CLIENT_SECRET")
   ```

3. **Create route**
   ```python
   # app/routes/oauth.py
   @router.get("/linkedin")
   async def linkedin_login():
       # Implementation
   ```

4. **Update middleware** (if security changes needed)

5. **Add tests**

6. **Update documentation**

### 2. **Code Style**

- **Python**: Follow PEP 8
- **Type Hints**: Use type hints for all function signatures
- **Docstrings**: Document all public functions and classes
- **Comments**: Explain "why", not "what"
- **Naming**: Use descriptive names (no abbreviations)

### 3. **Security Checklist**

Before deploying new features:

- [ ] Input validation implemented
- [ ] Output encoding applied
- [ ] Authentication required (if protected endpoint)
- [ ] Authorization checked (if user-specific data)
- [ ] Rate limiting configured
- [ ] Error messages don't leak sensitive info
- [ ] Logging excludes sensitive data
- [ ] Tests include security scenarios

## Testing Strategy

### Unit Tests
```
tests/
├── test_utils/
│   └── test_security_utils.py      # JWT, password hashing
├── test_middleware/
│   └── test_security.py            # Rate limiting, sanitization
├── test_routes/
│   ├── test_auth.py                # Authentication endpoints
│   └── test_service.py             # Service endpoints
└── test_parsers/
    └── test_cv_parser.py           # CV parsing
```

### Integration Tests
- Database operations
- API endpoint workflows
- OAuth flows

### Security Tests
- SQL injection attempts
- XSS attempts
- CSRF attacks
- Brute force login
- Rate limit enforcement

## Deployment

### Environment Variables

Required in production:
```bash
# Security (generate with: python -c "import secrets; print(secrets.token_urlsafe(32))")
JWT_SECRET=<32+ character random string>
REFRESH_SECRET=<32+ character random string>
SESSION_SECRET=<32+ character random string>

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db?ssl=require

# CORS
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# OAuth
GOOGLE_CLIENT_ID=<google-client-id>
GOOGLE_CLIENT_SECRET=<google-client-secret>
GITHUB_CLIENT_ID=<github-client-id>
GITHUB_CLIENT_SECRET=<github-client-secret>

# LiveKit
LIVEKIT_API_KEY=<livekit-api-key>
LIVEKIT_API_SECRET=<livekit-api-secret>
LIVEKIT_URL=wss://your-project.livekit.cloud

# AI Services
GOOGLE_API_KEY=<gemini-api-key>
```

### Production Checklist

- [ ] Strong secrets generated
- [ ] Database SSL enabled
- [ ] CORS origins configured
- [ ] HTTPS enforced
- [ ] Rate limits reviewed
- [ ] Logging configured
- [ ] Error tracking enabled
- [ ] Backups configured
- [ ] Monitoring setup
- [ ] Security headers verified

## Monitoring & Logging

### What to Monitor

1. **Performance**
   - Response times
   - Database query times
   - CPU/Memory usage

2. **Security**
   - Failed login attempts
   - Rate limit violations
   - Suspicious patterns

3. **Errors**
   - 4xx/5xx responses
   - Unhandled exceptions
   - Database errors

### Log Format

```
INFO:     GET /api/auth/me -> 200 (42.3 ms)
WARNING:  Failed login attempt for email@example.com
ERROR:    Database connection failed: timeout
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Check `__init__.py` files
   - Verify relative imports use correct path
   - Ensure modules are in PYTHONPATH

2. **Database Connection**
   - Verify DATABASE_URL format
   - Check PostgreSQL is running
   - Confirm SSL settings

3. **Authentication Fails**
   - Verify JWT_SECRET matches
   - Check token expiration
   - Confirm user exists in database

## Contributing

1. Follow the existing structure
2. Update this documentation for significant changes
3. Add tests for new features
4. Update SECURITY.md for security changes
5. Keep modules focused and cohesive

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [OWASP Security Guidelines](https://owasp.org/)
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [12-Factor App](https://12factor.net/)

---

**Last Updated**: November 2024  
**Version**: 1.0.0  
**Maintainers**: UtopiaHire Team
