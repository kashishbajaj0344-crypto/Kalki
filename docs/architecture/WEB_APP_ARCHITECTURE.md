# Kalki Construction Copilot - Web Application

## Tech Stack

### Frontend
- **Framework**: Next.js 14 (React 18 + TypeScript)
- **Styling**: TailwindCSS + shadcn/ui components
- **State**: Zustand (lightweight, simple)
- **Data Fetching**: React Query (TanStack Query)
- **Forms**: React Hook Form + Zod validation
- **Deployment**: Vercel (free tier)

### Backend
- **API**: FastAPI (Python 3.13)
- **Database**: PostgreSQL (Supabase free tier)
- **Vector DB**: Pinecone (free tier for MVP)
- **Authentication**: Supabase Auth (built-in)
- **Payments**: Stripe (test mode)
- **Deployment**: Railway or Render (free tier)

### AI/ML
- **LLM**: OpenAI GPT-4 Turbo (pay-as-you-go)
- **Embeddings**: OpenAI text-embedding-3-small
- **Fallback**: Local models (HuggingFace) for cost savings

---

## Project Structure

```
kalki-web/
├── frontend/                  # Next.js app
│   ├── app/                  # App Router (Next.js 14)
│   │   ├── page.tsx         # Landing page
│   │   ├── auth/            # Login/Signup
│   │   ├── dashboard/       # User dashboard
│   │   ├── project/         # Project workspace
│   │   │   └── [id]/
│   │   │       ├── page.tsx         # Project overview
│   │   │       ├── step/page.tsx    # Current step
│   │   │       ├── materials/       # Material selector
│   │   │       ├── budget/          # Budget tracker
│   │   │       └── timeline/        # Timeline view
│   │   ├── pricing/         # Pricing page
│   │   └── api/             # API routes (optional)
│   ├── components/          # Reusable components
│   │   ├── ui/             # shadcn/ui components
│   │   ├── NextStepCard.tsx
│   │   ├── MaterialSelector.tsx
│   │   ├── BudgetTracker.tsx
│   │   └── CodeComplianceChecker.tsx
│   ├── lib/                # Utilities
│   │   ├── api.ts          # API client
│   │   ├── auth.ts         # Auth helpers
│   │   └── utils.ts        # General utils
│   ├── hooks/              # Custom React hooks
│   └── types/              # TypeScript types
│
├── backend/                  # FastAPI app
│   ├── main.py             # Entry point
│   ├── api/
│   │   ├── auth.py         # Authentication
│   │   ├── projects.py     # Project management
│   │   ├── steps.py        # Step guidance
│   │   ├── materials.py    # Material selection
│   │   ├── compliance.py   # Code compliance
│   │   └── payments.py     # Stripe integration
│   ├── models/             # Pydantic models
│   ├── db/                 # Database models
│   ├── services/           # Business logic
│   │   ├── copilot.py      # Construction copilot logic
│   │   ├── llm.py          # LLM integration
│   │   └── vector_db.py    # Vector database
│   ├── utils/              # Utilities
│   └── tests/              # Unit tests
│
└── shared/                  # Shared code
    └── types.ts            # Shared TypeScript types
```

---

## MVP Features (Launch in 90 Days)

### Week 1-4: Core Product
- [x] Foundation phase (11 steps) ← Already built!
- [ ] Project creation flow
- [ ] Step-by-step guidance UI
- [ ] Material selection (basic)
- [ ] Budget tracking
- [ ] User authentication

### Week 5-8: Complete Build
- [ ] All 15 construction phases
- [ ] Material database (1000+ products)
- [ ] Code compliance checker
- [ ] Timeline Gantt chart
- [ ] Professional finder

### Week 9-12: Launch Features
- [ ] Stripe payment integration
- [ ] Subscription management
- [ ] Email notifications
- [ ] Mobile responsive design
- [ ] Analytics (PostHog or Mixpanel)

---

## API Endpoints

### Authentication
```
POST /api/auth/signup
POST /api/auth/login
POST /api/auth/logout
GET  /api/auth/me
```

### Projects
```
GET    /api/projects              # List user's projects
POST   /api/projects              # Create new project
GET    /api/projects/:id          # Get project details
PATCH  /api/projects/:id          # Update project
DELETE /api/projects/:id          # Delete project
```

### Steps
```
GET    /api/projects/:id/next-step          # Get next step guidance
POST   /api/projects/:id/steps/:step_id/complete  # Mark step complete
GET    /api/projects/:id/steps/:step_id    # Get step details
```

### Materials
```
GET    /api/materials/search?q=concrete&category=foundation
POST   /api/materials/recommend             # AI-powered recommendations
GET    /api/materials/:id                   # Get material details
```

### Code Compliance
```
POST   /api/compliance/check                # Check design against codes
GET    /api/compliance/codes?jurisdiction=austin-tx
```

### Payments
```
POST   /api/payments/create-checkout        # Create Stripe checkout session
POST   /api/payments/webhook                # Stripe webhook handler
GET    /api/payments/subscription           # Get subscription status
POST   /api/payments/cancel                 # Cancel subscription
```

---

## Database Schema

### Users
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name TEXT,
    phone TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Projects
```sql
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    name TEXT NOT NULL,
    description TEXT,
    phase TEXT NOT NULL,  -- DREAMING, FOUNDATION, etc.
    budget_total DECIMAL(12, 2),
    budget_spent DECIMAL(12, 2) DEFAULT 0,
    timeline_days_total INTEGER,
    timeline_days_elapsed INTEGER DEFAULT 0,
    location_address TEXT,
    location_jurisdiction TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Steps
```sql
CREATE TABLE project_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    step_number INTEGER NOT NULL,
    phase TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT DEFAULT 'not_started',  -- not_started, in_progress, completed
    completed_at TIMESTAMP,
    cost_actual DECIMAL(12, 2),
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Materials
```sql
CREATE TABLE materials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    category TEXT NOT NULL,  -- concrete, lumber, roofing, etc.
    description TEXT,
    price_per_unit DECIMAL(10, 2),
    unit TEXT,  -- SF, LF, each, etc.
    supplier TEXT,
    supplier_url TEXT,
    specifications JSONB,  -- Technical specs
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Subscriptions
```sql
CREATE TABLE subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    stripe_customer_id TEXT UNIQUE,
    stripe_subscription_id TEXT UNIQUE,
    tier TEXT NOT NULL,  -- starter, professional, enterprise
    status TEXT NOT NULL,  -- active, canceled, past_due
    current_period_start TIMESTAMP,
    current_period_end TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

---

## Environment Variables

### Frontend (.env.local)
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=xxx
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_test_xxx
```

### Backend (.env)
```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/kalki
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=xxx
OPENAI_API_KEY=sk-xxx
PINECONE_API_KEY=xxx
STRIPE_SECRET_KEY=sk_test_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx
```

---

## Deployment

### Frontend (Vercel)
```bash
cd frontend
npm run build
vercel deploy --prod
```

### Backend (Railway)
```bash
cd backend
railway login
railway init
railway up
```

### Database (Supabase)
- Sign up at supabase.com
- Create new project
- Run migrations
- Get connection string

---

## Cost Breakdown (Monthly)

| Service | Tier | Cost |
|---------|------|------|
| **Vercel** | Hobby | $0 |
| **Railway** | Starter | $5 |
| **Supabase** | Free | $0 |
| **Pinecone** | Free | $0 |
| **OpenAI API** | Pay-as-you-go | $50-200 |
| **Stripe** | 2.9% + $0.30 per transaction | ~$50 |
| **Domain** | .build TLD | $30/yr |
| **Total** | | **$105-305/mo** |

---

## Quick Start (For Developers)

### Frontend Setup
```bash
# Install Node.js 20+
npx create-next-app@latest kalki-web --typescript --tailwind --app
cd kalki-web
npm install zustand @tanstack/react-query react-hook-form zod
npm install @supabase/supabase-js stripe
npm install -D @types/node @types/react
npm run dev  # http://localhost:3000
```

### Backend Setup
```bash
# Install Python 3.13
cd backend
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn sqlalchemy psycopg2-binary
pip install openai pinecone-client stripe python-jose
pip install python-multipart python-dotenv
uvicorn main:app --reload  # http://localhost:8000
```

---

## Next Steps

1. **This Week**: Build landing page (kalki.build)
2. **Next Week**: Implement authentication + project creation
3. **Week 3**: Build step guidance UI
4. **Week 4**: Integrate Stripe payments
5. **Week 5-8**: Complete all 15 phases
6. **Week 9-12**: Launch + marketing

---

## Resources

- **Design**: Use Figma for mockups (figma.com)
- **Icons**: Heroicons or Lucide (free)
- **Components**: shadcn/ui (ui.shadcn.com)
- **Hosting**: Vercel + Railway (both have free tiers)
- **Domain**: Namecheap or GoDaddy ($30/yr for .build)

---

**Last Updated**: November 8, 2025
**Status**: Ready to build!
