// API Client for KALKI Construction Copilot
import axios from 'axios'

// TypeScript: Vite provides types for import.meta.env, no need to redeclare ImportMeta or ImportMetaEnv

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export interface ChatRequest {
  user_input: string
  project_id?: string
  context?: Record<string, any>
}

export interface ChatResponse {
  response: string
  confidence: number
  reasoning?: string
  next_steps?: string[]
  enhancements_used: string[]
  timestamp: string
}

export interface RoadmapRequest {
  project_type: 'adu' | 'remodel' | 'new_construction'
  property_data: Record<string, any>
  user_preferences?: Record<string, any>
}

export interface RoadmapResponse {
  steps: any[]
  total_weeks: number
  total_cost: number
  timeline: Record<string, any>
  complexity_score: number
}

export interface ConsensusRequest {
  decision: string
  context: Record<string, any>
  require_unanimous?: boolean
}

export interface ConsensusResponse {
  agreement: number
  recommendation: string
  individual_analyses: any[]
  conflicts: string[]
  voting_breakdown: Record<string, string>
}

export interface PropertyRequest {
  address: string
  property_type: 'residential' | 'commercial'
  project_intent: string
}

export interface PropertyResponse {
  complexity_score: number
  zoning_info: Record<string, any>
  setbacks?: Record<string, any>
  permit_requirements: string[]
  estimated_timeline: string
  risks: string[]
}

export interface LoginRequest {
  username: string
  password: string
}

export interface RegisterRequest {
  username: string
  email: string
  password: string
  full_name?: string
}

export interface AuthResponse {
  access_token: string
  token_type: string
  user: {
    username: string
    email: string
    full_name?: string
  }
}

export const kalkiAPI = {
  // Auth
  login: async (data: LoginRequest): Promise<AuthResponse> => {
    const response = await api.post('/api/auth/login', data)
    return response.data
  },

  register: async (data: RegisterRequest): Promise<AuthResponse> => {
    const response = await api.post('/api/auth/register', data)
    return response.data
  },

  // Chat
  chat: async (data: ChatRequest): Promise<ChatResponse> => {
    const response = await api.post('/api/chat', data)
    return response.data
  },

  // Roadmap
  generateRoadmap: async (data: RoadmapRequest): Promise<RoadmapResponse> => {
    const response = await api.post('/api/roadmap', data)
    return response.data
  },

  // Consensus
  getConsensus: async (data: ConsensusRequest): Promise<ConsensusResponse> => {
    const response = await api.post('/api/consensus', data)
    return response.data
  },

  // Property
  analyzeProperty: async (data: PropertyRequest): Promise<PropertyResponse> => {
    const response = await api.post('/api/property', data)
    return response.data
  },

  // Health
  healthCheck: async () => {
    const response = await api.get('/health')
    return response.data
  },
}

export default api
