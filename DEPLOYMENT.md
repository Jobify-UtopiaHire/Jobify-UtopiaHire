# Deployment Guide - Jobify Platform

## Table of Contents
1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Environment Setup](#environment-setup)
3. [Database Configuration](#database-configuration)
4. [Backend Deployment](#backend-deployment)
5. [Frontend Deployment](#frontend-deployment)
6. [SSL/TLS Configuration](#ssltls-configuration)
7. [Monitoring & Logging](#monitoring--logging)
8. [Backup & Recovery](#backup--recovery)
9. [Security Hardening](#security-hardening)
10. [Post-Deployment Testing](#post-deployment-testing)

---

## Pre-Deployment Checklist

### Security Requirements
- [ ] All default passwords changed
- [ ] Strong secrets generated (32+ characters)
- [ ] SSL/TLS certificates obtained
- [ ] Firewall rules configured
- [ ] Security headers tested
- [ ] Rate limits reviewed
- [ ] OAuth callbacks configured
- [ ] CORS origins restricted to production domains

### Infrastructure Requirements
- [ ] Domain name registered
- [ ] DNS configured
- [ ] Server/hosting provisioned
- [ ] Database provisioned
- [ ] Email service configured
- [ ] Backup strategy defined
- [ ] CDN configured (optional)
- [ ] Monitoring tools set up

### Code Requirements
- [ ] All tests passing
- [ ] Code reviewed
- [ ] Dependencies updated
- [ ] Documentation complete
- [ ] Environment variables documented
- [ ] Secrets stored securely

---

## Environment Setup

### Generate Strong Secrets

```bash
# Generate JWT secrets
python -c "import secrets; print('JWT_SECRET=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('REFRESH_SECRET=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('SESSION_SECRET=' + secrets.token_urlsafe(32))"

# Generate database password
python -c "import secrets; print('DB_PASSWORD=' + secrets.token_urlsafe(24))"
```

### Environment Variables

Create a secure `.env` file:

```bash
# Production Backend Environment Variables
# NEVER commit this file to version control!

# Security Secrets (CHANGE THESE!)
JWT_SECRET=<generated-from-above>
REFRESH_SECRET=<generated-from-above>
SESSION_SECRET=<generated-from-above>

# Database (use managed database in production)
DATABASE_URL=postgresql+asyncpg://username:password@db-host:5432/database?ssl=require

# CORS (restrict to your domain only)
CORS_ORIGINS=https://utopiahire.com,https://www.utopiahire.com

# Frontend URL
FRONTEND_URL=https://utopiahire.com

# LiveKit
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=<your-production-key>
LIVEKIT_API_SECRET=<your-production-secret>

# Google Gemini AI
GOOGLE_API_KEY=<your-production-key>

# OAuth (Google)
GOOGLE_CLIENT_ID=<your-client-id>
GOOGLE_CLIENT_SECRET=<your-client-secret>
GOOGLE_REDIRECT_URI=https://api.utopiahire.com/api/auth/oauth/google/callback

# OAuth (GitHub)
GITHUB_CLIENT_ID=<your-client-id>
GITHUB_CLIENT_SECRET=<your-client-secret>
GITHUB_REDIRECT_URI=https://api.utopiahire.com/api/auth/oauth/github/callback

# Email (use production SMTP)
SMTP_SERVER=smtp.sendgrid.net
SMTP_PORT=587
SENDER_EMAIL=noreply@utopiahire.com
SENDER_PASSWORD=<sendgrid-api-key>
SENDER_NAME=UtopiaHire

# Security Settings
ACCESS_EXPIRES_HOURS=2
REFRESH_EXPIRES_DAYS=7
```

---

## Database Configuration

### Option 1: Managed Database (Recommended)

#### AWS RDS PostgreSQL
```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier utopiahire-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username admin \
  --master-user-password <strong-password> \
  --allocated-storage 20 \
  --vpc-security-group-ids sg-xxxxx \
  --backup-retention-period 7 \
  --storage-encrypted

# Get connection string
aws rds describe-db-instances \
  --db-instance-identifier utopiahire-db \
  --query 'DBInstances[0].Endpoint'
```

#### DigitalOcean Managed Database
```bash
# Create via UI or API
doctl databases create utopiahire-db \
  --engine pg \
  --size db-s-1vcpu-1gb \
  --region nyc1 \
  --num-nodes 1
```

### Option 2: Self-Hosted PostgreSQL

```bash
# Install PostgreSQL 15
sudo apt update
sudo apt install postgresql-15 postgresql-contrib

# Configure for production
sudo nano /etc/postgresql/15/main/postgresql.conf

# Enable SSL
ssl = on
ssl_cert_file = '/etc/ssl/certs/ssl-cert-snakeoil.pem'
ssl_key_file = '/etc/ssl/private/ssl-cert-snakeoil.key'

# Security settings
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB

# Restart PostgreSQL
sudo systemctl restart postgresql
```

### Database Security

```sql
-- Create database and user
CREATE DATABASE utopiahire;
CREATE USER utopiahire_user WITH ENCRYPTED PASSWORD 'strong_password_here';
GRANT ALL PRIVILEGES ON DATABASE utopiahire TO utopiahire_user;

-- Restrict permissions
REVOKE ALL ON DATABASE utopiahire FROM PUBLIC;
```

### Initialize Database

```bash
cd backend

# Run migrations (automatic on startup)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or manually
python -c "
from app.db import engine, Base
import asyncio
async def init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
asyncio.run(init())
"
```

---

## Backend Deployment

### Option 1: Docker Container

```dockerfile
# Dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create upload directory
RUN mkdir -p cv_uploads && chmod 755 cv_uploads

# Run as non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t utopiahire-backend .
docker run -d \
  --name utopiahire-backend \
  --env-file .env \
  -p 8000:8000 \
  --restart unless-stopped \
  utopiahire-backend
```

### Option 2: Systemd Service

```bash
# Create service file
sudo nano /etc/systemd/system/utopiahire-backend.service
```

```ini
[Unit]
Description=UtopiaHire Backend API
After=network.target postgresql.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/utopiahire/backend
Environment="PATH=/var/www/utopiahire/backend/venv/bin"
EnvironmentFile=/var/www/utopiahire/backend/.env
ExecStart=/var/www/utopiahire/backend/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable utopiahire-backend
sudo systemctl start utopiahire-backend
sudo systemctl status utopiahire-backend
```

### Option 3: Cloud Platform

#### Heroku
```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create utopiahire-api

# Add PostgreSQL
heroku addons:create heroku-postgresql:hobby-dev

# Set environment variables
heroku config:set JWT_SECRET=<your-secret>
heroku config:set REFRESH_SECRET=<your-secret>
# ... set all required variables

# Deploy
git push heroku main

# Check logs
heroku logs --tail
```

#### DigitalOcean App Platform
```yaml
# .do/app.yaml
name: utopiahire
services:
- name: backend
  github:
    repo: anas-dev0/cs-challange
    branch: main
    deploy_on_push: true
  source_dir: /backend
  http_port: 8000
  instance_count: 2
  instance_size_slug: basic-xxs
  routes:
  - path: /
  envs:
  - key: DATABASE_URL
    scope: RUN_TIME
    type: SECRET
  - key: JWT_SECRET
    scope: RUN_TIME
    type: SECRET
  health_check:
    http_path: /health
databases:
- name: db
  engine: PG
  version: "15"
```

---

## Frontend Deployment

### Build for Production

```bash
cd frontend

# Install dependencies
npm install

# Build
npm run build

# Output will be in dist/
ls -la dist/
```

### Option 1: Vercel (Recommended for React)

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
vercel --prod

# Set environment variables
vercel env add VITE_API_URL production
# Enter: https://api.utopiahire.com
```

### Option 2: Netlify

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login
netlify login

# Deploy
netlify deploy --prod --dir=dist

# Set environment variables
netlify env:set VITE_API_URL https://api.utopiahire.com
```

### Option 3: Nginx Static Hosting

```nginx
# /etc/nginx/sites-available/utopiahire
server {
    listen 80;
    server_name utopiahire.com www.utopiahire.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name utopiahire.com www.utopiahire.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/utopiahire.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/utopiahire.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Root directory
    root /var/www/utopiahire/frontend/dist;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # SPA routing
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/utopiahire /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## SSL/TLS Configuration

### Let's Encrypt (Free)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d utopiahire.com -d www.utopiahire.com

# Auto-renewal (already configured)
sudo certbot renew --dry-run
```

### CloudFlare (Alternative)

1. Add domain to CloudFlare
2. Update nameservers
3. Enable "Full (strict)" SSL mode
4. Enable "Always Use HTTPS"
5. Enable "Automatic HTTPS Rewrites"

---

## Monitoring & Logging

### Application Monitoring

```python
# backend/app/monitoring.py
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/utopiahire/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('utopiahire')

# Log security events
def log_security_event(event_type: str, details: dict):
    logger.warning(f"SECURITY: {event_type} - {details}")
```

### Health Checks

```bash
# Add to crontab
*/5 * * * * curl -f http://localhost:8000/health || echo "Backend down!" | mail -s "Alert" admin@utopiahire.com
```

### Error Tracking

#### Sentry Integration

```python
# requirements.txt
sentry-sdk[fastapi]

# backend/app/main.py
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="https://your-sentry-dsn",
    integrations=[FastApiIntegration()],
    traces_sample_rate=0.1,
    environment="production"
)
```

---

## Backup & Recovery

### Database Backups

```bash
# Automated backup script
#!/bin/bash
# /usr/local/bin/backup-db.sh

BACKUP_DIR="/var/backups/utopiahire"
DATE=$(date +%Y%m%d_%H%M%S)
FILENAME="utopiahire_$DATE.sql.gz"

# Create backup
pg_dump -h localhost -U utopiahire_user utopiahire | gzip > "$BACKUP_DIR/$FILENAME"

# Keep only last 30 days
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +30 -delete

# Upload to S3 (optional)
aws s3 cp "$BACKUP_DIR/$FILENAME" s3://utopiahire-backups/

echo "Backup completed: $FILENAME"
```

```bash
# Add to crontab (daily at 2 AM)
0 2 * * * /usr/local/bin/backup-db.sh
```

### File Backups

```bash
# Backup uploaded files
rsync -avz /var/www/utopiahire/backend/cv_uploads/ user@backup-server:/backups/cv_uploads/
```

---

## Security Hardening

### Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 80/tcp  # HTTP
sudo ufw allow 443/tcp # HTTPS
sudo ufw enable
```

### Fail2Ban (Prevent Brute Force)

```bash
# Install
sudo apt install fail2ban

# Configure
sudo nano /etc/fail2ban/jail.local
```

```ini
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true

[nginx-limit-req]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log
```

### Security Updates

```bash
# Enable automatic security updates
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

---

## Post-Deployment Testing

### Security Tests

```bash
# Test SSL
sslscan utopiahire.com

# Test HTTP headers
curl -I https://utopiahire.com

# Test rate limiting
for i in {1..20}; do curl https://api.utopiahire.com/api/auth/login; done

# OWASP ZAP scan
zap-cli quick-scan https://utopiahire.com
```

### Functional Tests

```bash
# Health check
curl https://api.utopiahire.com/health

# Registration
curl -X POST https://api.utopiahire.com/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"name":"Test","email":"test@test.com","password":"Test123!@#"}'

# Login
curl -X POST https://api.utopiahire.com/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"Test123!@#"}'
```

### Performance Tests

```bash
# Apache Bench
ab -n 1000 -c 10 https://api.utopiahire.com/health

# Load testing with Locust
pip install locust
locust -f loadtest.py --host=https://api.utopiahire.com
```

---

## Troubleshooting

### Check Logs

```bash
# Application logs
tail -f /var/log/utopiahire/app.log

# Nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log

# Systemd logs
journalctl -u utopiahire-backend -f
```

### Database Connection Issues

```bash
# Test connection
psql "postgresql://user:pass@host:5432/db?sslmode=require"

# Check PostgreSQL status
sudo systemctl status postgresql
```

### SSL Issues

```bash
# Renew certificate
sudo certbot renew --force-renewal

# Check certificate
openssl s_client -connect utopiahire.com:443 -servername utopiahire.com
```

---

## Rollback Procedure

```bash
# Backend
git checkout <previous-commit>
docker build -t utopiahire-backend:rollback .
docker stop utopiahire-backend
docker run -d --name utopiahire-backend utopiahire-backend:rollback

# Frontend
vercel rollback  # or
netlify rollback

# Database
pg_restore -h localhost -U user -d database backup.sql
```



