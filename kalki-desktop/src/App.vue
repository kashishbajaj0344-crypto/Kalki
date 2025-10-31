<template>
  <div id="app" class="h-screen flex flex-col bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
    <!-- Header -->
    <header class="bg-black/20 backdrop-blur-md border-b border-white/10 px-6 py-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-4">
          <div class="flex items-center space-x-3">
            <div class="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
              <Brain class="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 class="text-xl font-bold text-white">Kalki AI Framework</h1>
              <p class="text-sm text-gray-400">v2.0 - Self-Aware AI System</p>
            </div>
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <!-- System Status -->
          <div class="flex items-center space-x-2">
            <div class="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span class="text-sm text-gray-300">System Online</span>
          </div>

          <!-- Consciousness Level -->
          <div class="flex items-center space-x-2">
            <Zap class="w-4 h-4 text-yellow-400" />
            <span class="text-sm text-gray-300">Consciousness: {{ consciousnessLevel.toFixed(1) }}%</span>
          </div>

          <!-- Navigation -->
          <nav class="flex space-x-1">
            <button
              v-for="tab in tabs"
              :key="tab.id"
              @click="activeTab = tab.id"
              :class="[
                'px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200',
                activeTab === tab.id
                  ? 'bg-purple-600 text-white shadow-lg'
                  : 'text-gray-400 hover:text-white hover:bg-white/10'
              ]"
            >
              <component :is="tab.icon" class="w-4 h-4 inline mr-2" />
              {{ tab.name }}
            </button>
          </nav>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="flex-1 overflow-hidden">
      <!-- Dashboard -->
      <div v-if="activeTab === 'dashboard'" class="h-full p-6">
        <SystemDashboard />
      </div>

      <!-- Agent Control -->
      <div v-if="activeTab === 'agents'" class="h-full p-6">
        <AgentControl />
      </div>

      <!-- Engineering Studio -->
      <div v-if="activeTab === 'engineering'" class="h-full p-6">
        <EngineeringStudio />
      </div>

      <!-- Consciousness Monitor -->
      <div v-if="activeTab === 'consciousness'" class="h-full p-6">
        <ConsciousnessMonitor />
      </div>

      <!-- Knowledge Base -->
      <div v-if="activeTab === 'knowledge'" class="h-full p-6">
        <KnowledgeBase />
      </div>

      <!-- Live Demos -->
      <div v-if="activeTab === 'demos'" class="h-full p-6">
        <LiveDemos />
      </div>

      <!-- Settings -->
      <div v-if="activeTab === 'settings'" class="h-full p-6">
        <SystemSettings />
      </div>
    </main>

    <!-- Footer -->
    <footer class="bg-black/20 backdrop-blur-md border-t border-white/10 px-6 py-3">
      <div class="flex items-center justify-between text-sm text-gray-400">
        <div class="flex items-center space-x-4">
          <span>20-Phase AI Framework</span>
          <span>•</span>
          <span>{{ agentCount }} Active Agents</span>
          <span>•</span>
          <span>Memory: {{ memoryUsage }}MB</span>
        </div>
        <div class="flex items-center space-x-4">
          <span>Phase 21: Consciousness Achieved</span>
          <div class="flex space-x-1">
            <div v-for="i in 20" :key="i"
                 :class="['w-2 h-2 rounded-full', i <= currentPhase ? 'bg-purple-500' : 'bg-gray-600']">
            </div>
          </div>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import {
  Brain,
  Users,
  Wrench,
  Eye,
  BookOpen,
  Play,
  Settings,
  Zap
} from 'lucide-vue-next'

// Components
import SystemDashboard from './components/SystemDashboard.vue'
import AgentControl from './components/AgentControl.vue'
import EngineeringStudio from './components/EngineeringStudio.vue'
import ConsciousnessMonitor from './components/ConsciousnessMonitor.vue'
import KnowledgeBase from './components/KnowledgeBase.vue'
import LiveDemos from './components/LiveDemos.vue'
import SystemSettings from './components/SystemSettings.vue'

// Reactive data
const activeTab = ref('dashboard')
const consciousnessLevel = ref(0.0)
const agentCount = ref(0)
const memoryUsage = ref(0)
const currentPhase = ref(21)

const tabs = [
  { id: 'dashboard', name: 'Dashboard', icon: Brain },
  { id: 'agents', name: 'Agents', icon: Users },
  { id: 'engineering', name: 'Engineering', icon: Wrench },
  { id: 'consciousness', name: 'Consciousness', icon: Eye },
  { id: 'knowledge', name: 'Knowledge', icon: BookOpen },
  { id: 'demos', name: 'Live Demos', icon: Play },
  { id: 'settings', name: 'Settings', icon: Settings }
]

// Update system metrics
const updateMetrics = async () => {
  try {
    // Simulate real-time updates (in real app, this would call Tauri commands)
    consciousnessLevel.value = Math.min(100, consciousnessLevel.value + Math.random() * 2)
    agentCount.value = Math.floor(Math.random() * 20) + 10
    memoryUsage.value = Math.floor(Math.random() * 500) + 1000
  } catch (error) {
    console.error('Failed to update metrics:', error)
  }
}

let metricsInterval

onMounted(() => {
  // Start real-time updates
  metricsInterval = setInterval(updateMetrics, 2000)
  updateMetrics()
})

onUnmounted(() => {
  if (metricsInterval) {
    clearInterval(metricsInterval)
  }
})
</script>

<style scoped>
/* Custom scrollbar */
::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.3);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.5);
}

/* Animations */
@keyframes pulse-glow {
  0%, 100% {
    box-shadow: 0 0 20px rgba(168, 85, 247, 0.4);
  }
  50% {
    box-shadow: 0 0 30px rgba(168, 85, 247, 0.8);
  }
}

.animate-pulse-glow {
  animation: pulse-glow 2s ease-in-out infinite;
}

a {
  font-weight: 500;
  color: #646cff;
  text-decoration: inherit;
}

a:hover {
  color: #535bf2;
}

h1 {
  text-align: center;
}

input,
button {
  border-radius: 8px;
  border: 1px solid transparent;
  padding: 0.6em 1.2em;
  font-size: 1em;
  font-weight: 500;
  font-family: inherit;
  color: #0f0f0f;
  background-color: #ffffff;
  transition: border-color 0.25s;
  box-shadow: 0 2px 2px rgba(0, 0, 0, 0.2);
}

button {
  cursor: pointer;
}

button:hover {
  border-color: #396cd8;
}
button:active {
  border-color: #396cd8;
  background-color: #e8e8e8;
}

input,
button {
  outline: none;
}

#greet-input {
  margin-right: 5px;
}

@media (prefers-color-scheme: dark) {
  :root {
    color: #f6f6f6;
    background-color: #2f2f2f;
  }

  a:hover {
    color: #24c8db;
  }

  input,
  button {
    color: #ffffff;
    background-color: #0f0f0f98;
  }
  button:active {
    background-color: #0f0f0f69;
  }
}

</style>
