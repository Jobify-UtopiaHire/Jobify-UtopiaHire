# UtopiaHire Frontend

A modern React + TypeScript frontend for the UtopiaHire AI-powered career platform. Features OAuth authentication (Google & GitHub), animated 3D shader backgrounds, dark-themed UI, and smooth animations.

## 🚀 Technology Stack

- **React 18** with **TypeScript** - Modern UI with type safety
- **Vite** - Lightning-fast development and building
- **React Router v6** - Client-side routing with protected routes
- **Tailwind CSS** - Utility-first CSS with custom dark theme
- **Axios** - HTTP client with automatic token refresh interceptors
- **Framer Motion** - Smooth page transitions and animations
- **@shadergradient/react** - Interactive 3D shader backgrounds
- **React Icons** - Icon library (Font Awesome, etc.)
- **clsx + tailwind-merge** - Utility for conditional class names

## 📋 Prerequisites

- **Node.js** 18+ and **npm**
- Backend API running at `http://localhost:8000` (see [backend README](../auth_fastapi/README.md))

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Configure Environment Variables

Create a `.env` file in the `frontend/` directory:

```bash
# Backend API base URL
VITE_API_BASE_URL=http://localhost:8000/api

# OAuth URLs (optional - only if OAuth is configured)
VITE_OAUTH_GOOGLE_URL=http://localhost:8000/api/auth/oauth/google
VITE_OAUTH_GITHUB_URL=http://localhost:8000/api/auth/oauth/github
```

**Note**: OAuth URLs are optional. If not set, OAuth sign-in buttons will still work but point to the default backend URLs.

### 3. Start Development Server

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### 4. Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

### 5. Preview Production Build

```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/       # Reusable UI components
│   ├── pages/           # Page components
│   ├── api.ts           # API client configuration
│   ├── AuthContext.tsx  # Authentication context
│   ├── ServiceContext.tsx # Service state management
│   ├── types.ts         # TypeScript type definitions
│   ├── App.tsx          # Main app component
│   └── main.tsx         # Entry point
├── public/              # Static assets
├── index.html           # HTML template
├── package.json         # Dependencies
├── tsconfig.json        # TypeScript configuration
├── vite.config.ts       # Vite configuration
└── tailwind.config.js   # Tailwind CSS configuration
```

## ✨ Features

### Authentication
- 🔐 **Multiple Sign-In Methods**: Email/password, Google OAuth, GitHub OAuth
- 🔄 **Automatic Token Refresh**: Access tokens auto-refresh before expiration
- 🛡️ **Protected Routes**: Dashboard and tools require authentication
- 👤 **Persistent Sessions**: Stay logged in across browser sessions
- 🎯 **Account Picker**: Google OAuth always prompts for account selection

### UI/UX
- 🌙 **Dark Theme**: Professional dark-themed interface
- 🎨 **Animated Backgrounds**: Interactive 3D shader gradients on landing page
- ✨ **Smooth Animations**: Page transitions with Framer Motion
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- 🎭 **Modal System**: Elegant authentication modal with scroll lock
- 🔔 **Toast Notifications**: Success/error feedback with auto-dismiss

### Pages
- 🏠 **Home**: Landing page with animated shader background
- 📊 **Dashboard**: User dashboard (protected)
- 📄 **CV Tool**: AI-powered CV analysis and optimization (coming soon)
- 🎤 **Interview**: AI interview practice with real-time feedback (coming soon)
- � **Job Matcher**: Intelligent job matching (coming soon)
- ℹ️ **About**: Mission and features showcase with interactive cards
- 💰 **Pricing**: Pricing plans (coming soon)

### Technical Features
- ⚡ **Fast HMR**: Instant hot module replacement with Vite
- 🎯 **Type Safety**: Full TypeScript coverage
- � **Axios Interceptors**: Automatic token refresh on 401 errors
- 🎨 **Tailwind CSS**: Utility-first styling with custom theme
- � **Optimized Builds**: Tree-shaking and code splitting

## 🎨 Design System

### Color Palette

Primary color (HSL): `225deg 83% 68%` (Purple-blue gradient)

```css
--primary: 225 83% 68%;
--primary-foreground: 0 0% 100%;
--background: 224 71% 4%;
--foreground: 213 31% 91%;
```

### Theme

The application uses a dark theme with purple-blue accents:
- Background: Very dark blue (`#060919`)
- Foreground: Light gray (`#e1e7ef`)
- Primary: Purple-blue gradient
- Cards: Dark blue with opacity
- Borders: Primary color with low opacity

## 🔗 API Integration

The frontend connects to the FastAPI backend via Axios. Configuration in `src/api.ts`:

- **Base URL**: `http://localhost:8000/api` (configurable via `VITE_API_BASE_URL`)
- **Auth Endpoints**: `/auth/register`, `/auth/login`, `/auth/me`, `/auth/refresh`
- **OAuth Endpoints**: `/auth/oauth/google`, `/auth/oauth/github` (with callbacks)
- **Token Storage**: `localStorage` (`token`, `refreshToken`)
- **Auto Refresh**: Axios interceptor catches 401 errors and refreshes tokens

### API Client Features

- **Automatic token attachment**: Bearer token added to all requests
- **Token refresh logic**: On 401, tries to refresh using refresh token
- **Error handling**: Clears tokens and redirects to home on refresh failure
- **Request/response interceptors**: Centralized auth handling

## 🔐 Authentication Flow

### Email/Password Flow

1. User enters email/password in auth modal
2. Frontend sends POST to `/api/auth/login`
3. Backend validates credentials and returns JWT tokens
4. Frontend stores tokens in localStorage
5. Frontend fetches user data from `/api/auth/me`
6. User is redirected to dashboard

### OAuth Flow (Google/GitHub)

1. User clicks "Sign in with Google/GitHub"
2. Frontend redirects to backend OAuth endpoint (`/api/auth/oauth/google`)
3. Backend redirects to provider (Google/GitHub) with OAuth params
4. User authorizes on provider's page
5. Provider redirects to backend callback with authorization code
6. Backend exchanges code for provider access token
7. Backend fetches user info from provider API
8. Backend creates/updates user in database
9. Backend issues JWT tokens and redirects to frontend with tokens in URL params
10. Frontend extracts tokens, stores them, fetches user, and cleans URL
11. User is logged in and redirected to dashboard

**Note**: Google OAuth includes `prompt=select_account` to always show account picker.

## 📁 Project Structure Explained

```
frontend/
├── src/
│   ├── components/
│   │   ├── AuthModal.tsx          # Login/Register modal with OAuth buttons
│   │   ├── Nav.tsx                # Navigation bar with user menu
│   │   ├── Footer.tsx             # Footer component
│   │   ├── BackgroundShader.tsx   # 3D shader gradient background
│   │   ├── ProtectedRoute.tsx     # Route guard for authenticated pages
│   │   └── ui/
│   │       └── Marquee.tsx        # Animated marquee component
│   ├── pages/
│   │   ├── Home.tsx               # Landing page with shader background
│   │   ├── Dashboard.tsx          # User dashboard (protected)
│   │   ├── About.tsx              # About page with feature cards
│   │   ├── Pricing.tsx            # Pricing page
│   │   ├── CVTool.tsx             # CV optimization tool (coming soon)
│   │   ├── Interview.tsx          # AI interview page (coming soon)
│   │   ├── InterviewerSetup.tsx   # Interview configuration (coming soon)
│   │   └── JobMatcher.tsx         # Job matching tool (coming soon)
│   ├── lib/
│   │   └── utils.ts               # Utility functions (clsx + tw-merge)
│   ├── AuthContext.tsx            # Auth state management (login, logout, OAuth)
│   ├── ServiceContext.tsx         # Service state (interview/CV modes)
│   ├── api.ts                     # Axios instance with interceptors
│   ├── types.ts                   # TypeScript type definitions
│   ├── App.tsx                    # Main app with routing and OAuth handler
│   ├── main.tsx                   # Entry point (React + Router)
│   └── style.css                  # Global styles and Tailwind imports
├── public/                        # Static assets
├── index.html                     # HTML template
├── package.json                   # Dependencies and scripts
├── tsconfig.json                  # TypeScript configuration
├── tsconfig.node.json             # TypeScript config for Vite
├── vite.config.ts                 # Vite configuration
├── tailwind.config.js             # Tailwind CSS theme customization
├── postcss.config.js              # PostCSS configuration
└── README.md                      # This file
```

## 🐛 Troubleshooting

### Port Already in Use

If port 5173 is already in use:

```bash
# Kill the process using the port (Windows PowerShell)
$process = Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
Stop-Process -Id $process -Force

# Or use a different port
npm run dev -- --port 5174
```

### OAuth Not Working

- Verify backend is running: http://localhost:8000/health
- Check that OAuth credentials are configured in backend `.env`
- Ensure `VITE_OAUTH_*_URL` variables point to correct backend URLs
- Clear browser localStorage: `localStorage.clear()`
- Check browser console for error messages

### Build Errors

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf node_modules/.vite
npm run dev
```

### Type Errors

```bash
# Regenerate TypeScript types
npx tsc --noEmit

# Check for missing type definitions
npm install --save-dev @types/react @types/react-dom
```

### Styling Issues

- Ensure Tailwind CSS is properly configured
- Check that `style.css` imports Tailwind directives
- Verify `tailwind.config.js` includes all content paths
- Clear browser cache and hard reload (Ctrl+Shift+R)

## 🚀 Deployment

### Vercel (Recommended)

1. Push code to GitHub
2. Import repository in Vercel
3. Configure build settings:
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`
4. Add environment variables:
   - `VITE_API_BASE_URL=https://your-backend-api.com/api`
   - `VITE_OAUTH_GOOGLE_URL=https://your-backend-api.com/api/auth/oauth/google`
   - `VITE_OAUTH_GITHUB_URL=https://your-backend-api.com/api/auth/oauth/github`
5. Deploy!

### Netlify

1. Run `npm run build` locally
2. Drag and drop the `dist/` folder to Netlify
3. Or connect GitHub repo and configure:
   - **Build Command**: `npm run build`
   - **Publish Directory**: `dist`

### Other Platforms

The frontend is a standard Vite React app and can be deployed to:
- GitHub Pages
- AWS S3 + CloudFront
- Firebase Hosting
- Cloudflare Pages

## 📚 Learn More

- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)
- [TypeScript Documentation](https://www.typescriptlang.org/docs/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [React Router Documentation](https://reactrouter.com/)
- [Framer Motion Documentation](https://www.framer.com/motion/)

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature/amazing-feature`
2. Make your changes
3. Test thoroughly
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open a Pull Request

## 📝 Notes

- OAuth sign-in requires backend OAuth configuration (see [backend README](../auth_fastapi/README.md))
- Access tokens expire after 2 hours, refresh tokens after 7 days
- The app uses localStorage for token persistence (consider HttpOnly cookies for production)
- Shader background may impact performance on low-end devices
- Some features (CV Tool, Interview, Job Matcher) are marked as "coming soon"
