import { Outlet } from 'react-router-dom'
import { Link } from 'react-router-dom'
import { useAuthStore } from '../stores/authStore'
import { MessageSquare, MapIcon, Home, Users, LogOut } from 'lucide-react'

export default function Layout() {
  const { user, logout } = useAuthStore()

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-2">
              <div className="text-2xl font-bold text-primary-600">🏗️ KALKI</div>
              <span className="text-sm text-gray-500">Construction Copilot</span>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-sm text-gray-600">Hi, {user?.username}</span>
              <button
                onClick={logout}
                className="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 rounded-lg transition"
              >
                <LogOut size={16} />
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-4rem)]">
        {/* Sidebar */}
        <aside className="w-64 bg-white border-r">
          <nav className="p-4 space-y-2">
            <Link
              to="/chat"
              className="flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-primary-50 text-gray-700 hover:text-primary-600 transition"
            >
              <MessageSquare size={20} />
              <span>Chat</span>
            </Link>
            <Link
              to="/roadmap"
              className="flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-primary-50 text-gray-700 hover:text-primary-600 transition"
            >
              <MapIcon size={20} />
              <span>Roadmap</span>
            </Link>
            <Link
              to="/property"
              className="flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-primary-50 text-gray-700 hover:text-primary-600 transition"
            >
              <Home size={20} />
              <span>Property Analysis</span>
            </Link>
            <Link
              to="/consensus"
              className="flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-primary-50 text-gray-700 hover:text-primary-600 transition"
            >
              <Users size={20} />
              <span>Multi-Agent Consensus</span>
            </Link>
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
