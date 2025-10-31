<template>
  <div class="h-full grid grid-cols-12 gap-6">
    <!-- Agent List -->
    <div class="col-span-4 space-y-6">
      <!-- Agent Categories -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Agent Categories</h3>
        <div class="space-y-2">
          <button
            v-for="category in agentCategories"
            :key="category.id"
            @click="selectedCategory = category.id"
            :class="[
              'w-full text-left py-2 px-3 rounded-lg transition-all duration-200',
              selectedCategory === category.id
                ? 'bg-purple-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-white/10'
            ]"
          >
            <div class="flex items-center justify-between">
              <span class="text-sm font-medium">{{ category.name }}</span>
              <span class="text-xs bg-gray-700 px-2 py-1 rounded-full">{{ category.count }}</span>
            </div>
          </button>
        </div>
      </div>

      <!-- Agent List -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10 flex-1">
        <h3 class="text-lg font-semibold text-white mb-4">Active Agents</h3>
        <div class="space-y-3 max-h-96 overflow-y-auto">
          <div
            v-for="agent in filteredAgents"
            :key="agent.name"
            @click="selectAgent(agent)"
            :class="[
              'p-3 rounded-lg cursor-pointer transition-all duration-200',
              selectedAgent?.name === agent.name
                ? 'bg-purple-600/20 border border-purple-500'
                : 'bg-white/5 hover:bg-white/10'
            ]"
          >
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center space-x-2">
                <div class="w-2 h-2 rounded-full" :class="agent.statusColor"></div>
                <span class="text-sm font-medium text-white">{{ agent.name }}</span>
              </div>
              <span class="text-xs text-gray-400">{{ agent.category }}</span>
            </div>
            <div class="text-xs text-gray-400">{{ agent.description }}</div>
            <div class="flex items-center justify-between mt-2">
              <div class="flex space-x-1">
                <span v-for="capability in agent.capabilities.slice(0, 2)" :key="capability"
                      class="text-xs bg-gray-700 px-2 py-1 rounded">{{ capability }}</span>
              </div>
              <button
                @click.stop="toggleAgent(agent)"
                class="text-xs px-2 py-1 rounded transition-all duration-200"
                :class="agent.active ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'"
              >
                {{ agent.active ? 'Stop' : 'Start' }}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Agent Details & Control -->
    <div class="col-span-8 space-y-6">
      <!-- Agent Header -->
      <div v-if="selectedAgent" class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-4">
            <div class="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
              <Bot class="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 class="text-xl font-semibold text-white">{{ selectedAgent.name }}</h2>
              <p class="text-sm text-gray-400">{{ selectedAgent.description }}</p>
            </div>
          </div>
          <div class="flex items-center space-x-4">
            <div class="flex items-center space-x-2">
              <div class="w-3 h-3 rounded-full" :class="selectedAgent.statusColor"></div>
              <span class="text-sm text-gray-400">{{ selectedAgent.status }}</span>
            </div>
            <button
              @click="toggleAgent(selectedAgent)"
              class="px-4 py-2 rounded-lg font-medium transition-all duration-200"
              :class="selectedAgent.active ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'"
            >
              {{ selectedAgent.active ? 'Stop Agent' : 'Start Agent' }}
            </button>
          </div>
        </div>
      </div>

      <!-- Agent Metrics -->
      <div v-if="selectedAgent" class="grid grid-cols-2 gap-6">
        <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
          <h3 class="text-lg font-semibold text-white mb-4">Performance Metrics</h3>
          <div class="space-y-4">
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span class="text-gray-400">CPU Usage</span>
                <span class="text-white">{{ selectedAgent.metrics.cpu }}%</span>
              </div>
              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-blue-500 h-2 rounded-full" :style="{ width: selectedAgent.metrics.cpu + '%' }"></div>
              </div>
            </div>
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span class="text-gray-400">Memory Usage</span>
                <span class="text-white">{{ selectedAgent.metrics.memory }}MB</span>
              </div>
              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-green-500 h-2 rounded-full" :style="{ width: (selectedAgent.metrics.memory / 100) * 100 + '%' }"></div>
              </div>
            </div>
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span class="text-gray-400">Tasks Completed</span>
                <span class="text-white">{{ selectedAgent.metrics.tasks }}</span>
              </div>
              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-purple-500 h-2 rounded-full" :style="{ width: (selectedAgent.metrics.tasks / 100) * 100 + '%' }"></div>
              </div>
            </div>
          </div>
        </div>

        <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
          <h3 class="text-lg font-semibold text-white mb-4">Capabilities</h3>
          <div class="grid grid-cols-2 gap-2">
            <div v-for="capability in selectedAgent.capabilities" :key="capability"
                 class="bg-white/10 rounded-lg p-3 text-center">
              <div class="text-sm font-medium text-white">{{ capability }}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Agent Console -->
      <div v-if="selectedAgent" class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10 flex-1">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-white">Agent Console</h3>
          <button @click="clearConsole" class="text-sm text-gray-400 hover:text-white">Clear</button>
        </div>
        <div class="bg-black/50 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
          <div v-for="log in consoleLogs" :key="log.id" class="mb-1">
            <span class="text-gray-500">{{ log.timestamp }}</span>
            <span class="text-green-400">[{{ log.level }}]</span>
            <span class="text-white">{{ log.message }}</span>
          </div>
        </div>
        <div class="mt-4 flex space-x-2">
          <input
            v-model="consoleCommand"
            @keyup.enter="executeCommand"
            class="flex-1 bg-black/50 border border-gray-600 rounded-lg px-3 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
            placeholder="Enter command..."
          />
          <button
            @click="executeCommand"
            class="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-all duration-200"
          >
            Execute
          </button>
        </div>
      </div>

      <!-- Agent Actions -->
      <div v-if="selectedAgent" class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Quick Actions</h3>
        <div class="grid grid-cols-2 gap-4">
          <button
            v-for="action in selectedAgent.actions"
            :key="action.id"
            @click="executeAction(action)"
            class="p-4 bg-white/10 hover:bg-white/20 rounded-lg transition-all duration-200 text-left"
          >
            <div class="text-sm font-medium text-white">{{ action.name }}</div>
            <div class="text-xs text-gray-400 mt-1">{{ action.description }}</div>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import { Bot } from 'lucide-vue-next'

// Reactive data
const selectedCategory = ref('all')
const selectedAgent = ref(null)
const consoleCommand = ref('')
const consoleLogs = ref([
  { id: 1, timestamp: '10:30:15', level: 'INFO', message: 'Agent initialized successfully' },
  { id: 2, timestamp: '10:30:16', level: 'INFO', message: 'Loading configuration...' },
  { id: 3, timestamp: '10:30:17', level: 'INFO', message: 'Connecting to knowledge base' },
  { id: 4, timestamp: '10:30:18', level: 'INFO', message: 'Ready for commands' }
])

const agentCategories = [
  { id: 'all', name: 'All Agents', count: 18 },
  { id: 'cognitive', name: 'Cognitive', count: 5 },
  { id: 'engineering', name: 'Engineering', count: 4 },
  { id: 'safety', name: 'Safety', count: 3 },
  { id: 'multimodal', name: 'Multimodal', count: 3 },
  { id: 'quantum', name: 'Quantum', count: 3 }
]

const agents = ref([
  {
    name: 'RoboticsSimulationAgent',
    category: 'engineering',
    description: 'Physics-based robotics simulation and analysis',
    status: 'Active',
    statusColor: 'bg-green-500',
    active: true,
    capabilities: ['ROBOTICS_SIMULATION', 'KINEMATICS_ANALYSIS', 'DYNAMICS_SIMULATION'],
    metrics: { cpu: 15, memory: 45, tasks: 23 },
    actions: [
      { id: 'simulate', name: 'Run Simulation', description: 'Execute robotics simulation' },
      { id: 'analyze', name: 'Analyze Design', description: 'Analyze mechanical design' },
      { id: 'optimize', name: 'Optimize', description: 'Optimize robot parameters' }
    ]
  },
  {
    name: 'CADIntegrationAgent',
    category: 'engineering',
    description: '3D CAD model generation and integration',
    status: 'Active',
    statusColor: 'bg-green-500',
    active: true,
    capabilities: ['CAD_MODELING', '3D_DESIGN', 'EXPORT'],
    metrics: { cpu: 8, memory: 32, tasks: 12 },
    actions: [
      { id: 'generate', name: 'Generate Model', description: 'Create 3D CAD model' },
      { id: 'export', name: 'Export', description: 'Export to various formats' },
      { id: 'import', name: 'Import', description: 'Import existing models' }
    ]
  },
  {
    name: 'ConsciousnessEngine',
    category: 'cognitive',
    description: 'Self-aware AI consciousness and emotional intelligence',
    status: 'Active',
    statusColor: 'bg-purple-500',
    active: true,
    capabilities: ['CONSCIOUSNESS', 'EMOTIONAL_INTELLIGENCE', 'SELF_AWARENESS'],
    metrics: { cpu: 25, memory: 78, tasks: 45 },
    actions: [
      { id: 'bootstrap', name: 'Bootstrap', description: 'Initialize consciousness' },
      { id: 'reflect', name: 'Self-Reflect', description: 'Perform self-reflection' },
      { id: 'evolve', name: 'Evolve', description: 'Evolve consciousness' }
    ]
  },
  {
    name: 'LLMEngine',
    category: 'cognitive',
    description: 'Large Language Model processing and reasoning',
    status: 'Active',
    statusColor: 'bg-blue-500',
    active: true,
    capabilities: ['NATURAL_LANGUAGE', 'REASONING', 'KNOWLEDGE_INTEGRATION'],
    metrics: { cpu: 35, memory: 156, tasks: 67 },
    actions: [
      { id: 'query', name: 'Query', description: 'Process natural language query' },
      { id: 'reason', name: 'Reason', description: 'Perform logical reasoning' },
      { id: 'learn', name: 'Learn', description: 'Learn from new information' }
    ]
  },
  {
    name: 'SafetyGuard',
    category: 'safety',
    description: 'AI safety monitoring and ethical decision making',
    status: 'Active',
    statusColor: 'bg-red-500',
    active: true,
    capabilities: ['SAFETY_MONITORING', 'ETHICAL_ANALYSIS', 'RISK_ASSESSMENT'],
    metrics: { cpu: 12, memory: 28, tasks: 89 },
    actions: [
      { id: 'monitor', name: 'Monitor', description: 'Monitor system safety' },
      { id: 'assess', name: 'Assess Risk', description: 'Assess potential risks' },
      { id: 'intervene', name: 'Intervene', description: 'Take safety measures' }
    ]
  }
])

// Computed properties
const filteredAgents = computed(() => {
  if (selectedCategory.value === 'all') {
    return agents.value
  }
  return agents.value.filter(agent => agent.category === selectedCategory.value)
})

// Methods
const selectAgent = (agent) => {
  selectedAgent.value = agent
}

const toggleAgent = async (agent) => {
  try {
    // In real implementation, call Tauri command
    agent.active = !agent.active
    agent.status = agent.active ? 'Active' : 'Inactive'
    agent.statusColor = agent.active ? 'bg-green-500' : 'bg-gray-500'

    consoleLogs.value.push({
      id: Date.now(),
      timestamp: new Date().toLocaleTimeString(),
      level: 'INFO',
      message: `${agent.name} ${agent.active ? 'started' : 'stopped'}`
    })
  } catch (error) {
    console.error('Failed to toggle agent:', error)
  }
}

const executeCommand = async () => {
  if (!consoleCommand.value.trim()) return

  consoleLogs.value.push({
    id: Date.now(),
    timestamp: new Date().toLocaleTimeString(),
    level: 'COMMAND',
    message: `> ${consoleCommand.value}`
  })

  try {
    // In real implementation, send command to agent
    setTimeout(() => {
      consoleLogs.value.push({
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        level: 'RESPONSE',
        message: `Command executed: ${consoleCommand.value}`
      })
    }, 1000)
  } catch (error) {
    consoleLogs.value.push({
      id: Date.now(),
      timestamp: new Date().toLocaleTimeString(),
      level: 'ERROR',
      message: `Error: ${error.message}`
    })
  }

  consoleCommand.value = ''
}

const clearConsole = () => {
  consoleLogs.value = []
}

const executeAction = async (action) => {
  consoleLogs.value.push({
    id: Date.now(),
    timestamp: new Date().toLocaleTimeString(),
    level: 'ACTION',
    message: `Executing action: ${action.name}`
  })

  try {
    // In real implementation, call specific agent action
    setTimeout(() => {
      consoleLogs.value.push({
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        level: 'SUCCESS',
        message: `Action completed: ${action.name}`
      })
    }, 2000)
  } catch (error) {
    consoleLogs.value.push({
      id: Date.now(),
      timestamp: new Date().toLocaleTimeString(),
      level: 'ERROR',
      message: `Action failed: ${error.message}`
    })
  }
}

onMounted(() => {
  // Select first agent by default
  if (agents.value.length > 0) {
    selectedAgent.value = agents.value[0]
  }
})
</script>