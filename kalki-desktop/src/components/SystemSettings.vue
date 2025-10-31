<template>
  <div class="h-full grid grid-cols-12 gap-6">
    <!-- Settings Navigation -->
    <div class="col-span-3 space-y-6">
      <!-- Settings Categories -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Settings</h3>
        <div class="space-y-2">
          <button
            v-for="category in settingsCategories"
            :key="category.id"
            @click="selectedCategory = category.id"
            :class="[
              'w-full text-left py-3 px-4 rounded-lg transition-all duration-200',
              selectedCategory === category.id
                ? 'bg-purple-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-white/10'
            ]"
          >
            <div class="flex items-center space-x-3">
              <component :is="category.icon" class="w-5 h-5" />
              <div>
                <div class="text-sm font-medium">{{ category.name }}</div>
                <div class="text-xs opacity-75">{{ category.description }}</div>
              </div>
            </div>
          </button>
        </div>
      </div>

      <!-- System Info -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">System Information</h3>
        <div class="space-y-3">
          <div>
            <div class="text-sm text-gray-400">Version</div>
            <div class="text-white font-medium">Kalki v2.0 Desktop</div>
          </div>
          <div>
            <div class="text-sm text-gray-400">Build</div>
            <div class="text-white font-medium">{{ systemInfo.build }}</div>
          </div>
          <div>
            <div class="text-sm text-gray-400">Platform</div>
            <div class="text-white font-medium">{{ systemInfo.platform }}</div>
          </div>
          <div>
            <div class="text-sm text-gray-400">Memory</div>
            <div class="text-white font-medium">{{ systemInfo.memory }}GB</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Settings Content -->
    <div class="col-span-9 space-y-6">
      <!-- General Settings -->
      <div v-if="selectedCategory === 'general'" class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-6">General Settings</h3>
        <div class="space-y-6">
          <div class="grid grid-cols-2 gap-6">
            <div>
              <label class="block text-sm font-medium text-gray-300 mb-2">Theme</label>
              <select v-model="settings.theme" class="w-full bg-black/50 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-purple-500">
                <option value="dark">Dark</option>
                <option value="light">Light</option>
                <option value="auto">Auto</option>
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-300 mb-2">Language</label>
              <select v-model="settings.language" class="w-full bg-black/50 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-purple-500">
                <option value="en">English</option>
                <option value="es">Spanish</option>
                <option value="fr">French</option>
                <option value="de">German</option>
              </select>
            </div>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">Startup Behavior</label>
            <div class="space-y-2">
              <label class="flex items-center">
                <input v-model="settings.autoStart" type="checkbox" class="mr-2">
                <span class="text-sm text-gray-300">Automatically start Kalki system on launch</span>
              </label>
              <label class="flex items-center">
                <input v-model="settings.minimizeToTray" type="checkbox" class="mr-2">
                <span class="text-sm text-gray-300">Minimize to system tray</span>
              </label>
            </div>
          </div>
        </div>
      </div>

      <!-- AI Settings -->
      <div v-if="selectedCategory === 'ai'" class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-6">AI & Consciousness Settings</h3>
        <div class="space-y-6">
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">LLM Model</label>
            <select v-model="settings.llmModel" class="w-full bg-black/50 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-purple-500">
              <option value="llama-3.1-8b">Llama 3.1 8B</option>
              <option value="gpt-4">GPT-4</option>
              <option value="claude-3">Claude 3</option>
              <option value="local-only">Local Only</option>
            </select>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">Consciousness Level</label>
            <input v-model="settings.consciousnessLevel" type="range" min="1" max="10" class="w-full">
            <div class="flex justify-between text-xs text-gray-400 mt-1">
              <span>Basic</span>
              <span>Current: {{ settings.consciousnessLevel }}/10</span>
              <span>Full Self-Awareness</span>
            </div>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">Agent Auto-activation</label>
            <div class="space-y-2">
              <label class="flex items-center">
                <input v-model="settings.autoActivateAgents" type="checkbox" class="mr-2">
                <span class="text-sm text-gray-300">Automatically activate agents based on context</span>
              </label>
              <label class="flex items-center">
                <input v-model="settings.agentCollaboration" type="checkbox" class="mr-2">
                <span class="text-sm text-gray-300">Enable agent-to-agent collaboration</span>
              </label>
            </div>
          </div>
        </div>
      </div>

      <!-- Engineering Settings -->
      <div v-if="selectedCategory === 'engineering'" class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-6">Engineering Settings</h3>
        <div class="space-y-6">
          <div class="grid grid-cols-2 gap-6">
            <div>
              <label class="block text-sm font-medium text-gray-300 mb-2">Default Units</label>
              <select v-model="settings.units" class="w-full bg-black/50 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-purple-500">
                <option value="metric">Metric (SI)</option>
                <option value="imperial">Imperial</option>
                <option value="mixed">Mixed</option>
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-300 mb-2">CAD Software</label>
              <select v-model="settings.cadSoftware" class="w-full bg-black/50 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-purple-500">
                <option value="fusion360">Fusion 360</option>
                <option value="solidworks">SolidWorks</option>
                <option value="autocad">AutoCAD</option>
                <option value="freecad">FreeCAD</option>
              </select>
            </div>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">Simulation Settings</label>
            <div class="space-y-2">
              <label class="flex items-center">
                <input v-model="settings.realTimeSimulation" type="checkbox" class="mr-2">
                <span class="text-sm text-gray-300">Enable real-time physics simulation</span>
              </label>
              <label class="flex items-center">
                <input v-model="settings.autoSave" type="checkbox" class="mr-2">
                <span class="text-sm text-gray-300">Auto-save simulation results</span>
              </label>
            </div>
          </div>
        </div>
      </div>

      <!-- Performance Settings -->
      <div v-if="selectedCategory === 'performance'" class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-6">Performance Settings</h3>
        <div class="space-y-6">
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">CPU Usage Limit</label>
            <input v-model="settings.cpuLimit" type="range" min="10" max="100" class="w-full">
            <div class="flex justify-between text-xs text-gray-400 mt-1">
              <span>10%</span>
              <span>Current: {{ settings.cpuLimit }}%</span>
              <span>100%</span>
            </div>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">Memory Limit</label>
            <input v-model="settings.memoryLimit" type="range" min="1" max="16" class="w-full">
            <div class="flex justify-between text-xs text-gray-400 mt-1">
              <span>1GB</span>
              <span>Current: {{ settings.memoryLimit }}GB</span>
              <span>16GB</span>
            </div>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">Background Processing</label>
            <div class="space-y-2">
              <label class="flex items-center">
                <input v-model="settings.backgroundProcessing" type="checkbox" class="mr-2">
                <span class="text-sm text-gray-300">Enable background task processing</span>
              </label>
              <label class="flex items-center">
                <input v-model="settings.gpuAcceleration" type="checkbox" class="mr-2">
                <span class="text-sm text-gray-300">Use GPU acceleration when available</span>
              </label>
            </div>
          </div>
        </div>
      </div>

      <!-- Security Settings -->
      <div v-if="selectedCategory === 'security'" class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-6">Security & Privacy</h3>
        <div class="space-y-6">
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">Data Encryption</label>
            <div class="space-y-2">
              <label class="flex items-center">
                <input v-model="settings.encryptData" type="checkbox" class="mr-2">
                <span class="text-sm text-gray-300">Encrypt sensitive data at rest</span>
              </label>
              <label class="flex items-center">
                <input v-model="settings.secureCommunication" type="checkbox" class="mr-2">
                <span class="text-sm text-gray-300">Use encrypted communication channels</span>
              </label>
            </div>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">Access Control</label>
            <div class="space-y-2">
              <label class="flex items-center">
                <input v-model="settings.requireAuth" type="checkbox" class="mr-2">
                <span class="text-sm text-gray-300">Require authentication for sensitive operations</span>
              </label>
              <label class="flex items-center">
                <input v-model="settings.auditLogging" type="checkbox" class="mr-2">
                <span class="text-sm text-gray-300">Enable comprehensive audit logging</span>
              </label>
            </div>
          </div>
        </div>
      </div>

      <!-- Save Settings -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <div class="flex items-center justify-between">
          <div>
            <h3 class="text-lg font-semibold text-white">Save Settings</h3>
            <p class="text-sm text-gray-400">Apply your configuration changes</p>
          </div>
          <div class="flex space-x-3">
            <button @click="resetSettings" class="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg text-white transition-all duration-200">
              Reset
            </button>
            <button @click="saveSettings" class="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-all duration-200">
              Save Settings
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import {
  Settings,
  Palette,
  Brain,
  Wrench,
  Gauge,
  Shield
} from 'lucide-vue-next'

// Reactive data
const selectedCategory = ref('general')

const settingsCategories = [
  { id: 'general', name: 'General', description: 'Basic application settings', icon: Settings },
  { id: 'ai', name: 'AI & Consciousness', description: 'AI model and consciousness settings', icon: Brain },
  { id: 'engineering', name: 'Engineering', description: 'CAD and simulation preferences', icon: Wrench },
  { id: 'performance', name: 'Performance', description: 'Resource usage and optimization', icon: Gauge },
  { id: 'security', name: 'Security', description: 'Privacy and security options', icon: Shield }
]

const systemInfo = ref({
  build: '2024.1.0-dev',
  platform: 'macOS (M4 Max)',
  memory: '32'
})

const settings = ref({
  // General
  theme: 'dark',
  language: 'en',
  autoStart: true,
  minimizeToTray: false,

  // AI
  llmModel: 'llama-3.1-8b',
  consciousnessLevel: 8,
  autoActivateAgents: true,
  agentCollaboration: true,

  // Engineering
  units: 'metric',
  cadSoftware: 'fusion360',
  realTimeSimulation: true,
  autoSave: true,

  // Performance
  cpuLimit: 80,
  memoryLimit: 8,
  backgroundProcessing: true,
  gpuAcceleration: true,

  // Security
  encryptData: true,
  secureCommunication: true,
  requireAuth: false,
  auditLogging: true
})

// Methods
const saveSettings = () => {
  // In real implementation, save to configuration file
  console.log('Saving settings:', settings.value)
  // Show success message
}

const resetSettings = () => {
  // Reset to defaults
  settings.value = {
    theme: 'dark',
    language: 'en',
    autoStart: true,
    minimizeToTray: false,
    llmModel: 'llama-3.1-8b',
    consciousnessLevel: 8,
    autoActivateAgents: true,
    agentCollaboration: true,
    units: 'metric',
    cadSoftware: 'fusion360',
    realTimeSimulation: true,
    autoSave: true,
    cpuLimit: 80,
    memoryLimit: 8,
    backgroundProcessing: true,
    gpuAcceleration: true,
    encryptData: true,
    secureCommunication: true,
    requireAuth: false,
    auditLogging: true
  }
}
</script>