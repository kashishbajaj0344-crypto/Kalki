import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from './stores/authStore'
import Layout from './components/Layout'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import ChatPage from './pages/ChatPage'
import RoadmapPage from './pages/RoadmapPage'
import PropertyPage from './pages/PropertyPage'
import ConsensusPage from './pages/ConsensusPage'

function App() {
  const { isAuthenticated } = useAuthStore()

  return (
    <Routes>
      {/* Public routes */}
      <Route path="/login" element={
        isAuthenticated ? <Navigate to="/chat" /> : <LoginPage />
      } />
      <Route path="/register" element={
        isAuthenticated ? <Navigate to="/chat" /> : <RegisterPage />
      } />

      {/* Protected routes */}
      <Route element={isAuthenticated ? <Layout /> : <Navigate to="/login" />}>
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/roadmap" element={<RoadmapPage />} />
        <Route path="/property" element={<PropertyPage />} />
        <Route path="/consensus" element={<ConsensusPage />} />
        <Route path="/" element={<Navigate to="/chat" />} />
      </Route>

      {/* Catch all */}
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  )
}

export default App
