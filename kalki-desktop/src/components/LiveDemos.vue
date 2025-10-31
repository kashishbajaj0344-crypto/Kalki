<template>
  <div class="h-full grid grid-cols-12 gap-6">
    <!-- Demo Categories -->
    <div class="col-span-3 space-y-6">
      <!-- Demo Categories -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Demo Categories</h3>
        <div class="space-y-2">
          <button
            v-for="category in demoCategories"
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

      <!-- Demo Stats -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Demo Statistics</h3>
        <div class="space-y-3">
          <div class="flex justify-between">
            <span class="text-sm text-gray-400">Total Demos</span>
            <span class="text-sm text-white font-medium">{{ demoStats.total }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-sm text-gray-400">Completed</span>
            <span class="text-sm text-green-400 font-medium">{{ demoStats.completed }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-sm text-gray-400">Running</span>
            <span class="text-sm text-yellow-400 font-medium">{{ demoStats.running }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-sm text-gray-400">Success Rate</span>
            <span class="text-sm text-blue-400 font-medium">{{ demoStats.successRate }}%</span>
          </div>
        </div>
      </div>

      <!-- Quick Actions -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Quick Actions</h3>
        <div class="space-y-3">
          <button @click="runAllDemos" class="w-full py-2 px-4 bg-purple-600 hover:bg-purple-700 rounded-lg text-white font-medium transition-all duration-200">
            Run All Demos
          </button>
          <button @click="stopAllDemos" class="w-full py-2 px-4 bg-red-600 hover:bg-red-700 rounded-lg text-white font-medium transition-all duration-200">
            Stop All
          </button>
          <button @click="resetDemos" class="w-full py-2 px-4 bg-gray-600 hover:bg-gray-700 rounded-lg text-white font-medium transition-all duration-200">
            Reset All
          </button>
        </div>
      </div>
    </div>

    <!-- Demo Interface -->
    <div class="col-span-9 space-y-6">
      <!-- Demo Runner -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <div class="flex items-center justify-between mb-6">
          <h3 class="text-lg font-semibold text-white">Live Demo Runner</h3>
          <div class="flex items-center space-x-2">
            <div class="w-3 h-3 rounded-full" :class="demoStatus.color"></div>
            <span class="text-sm text-gray-400">{{ demoStatus.text }}</span>
          </div>
        </div>

        <!-- Current Demo Display -->
        <div v-if="currentDemo" class="mb-6">
          <div class="flex items-center justify-between mb-4">
            <div class="flex items-center space-x-4">
              <component :is="currentDemo.icon" class="w-8 h-8 text-purple-400" />
              <div>
                <h4 class="text-xl font-semibold text-white">{{ currentDemo.name }}</h4>
                <p class="text-gray-400">{{ currentDemo.description }}</p>
              </div>
            </div>
            <div class="flex items-center space-x-2">
              <span class="text-sm text-gray-400">Progress:</span>
              <div class="w-32 bg-gray-700 rounded-full h-2">
                <div class="bg-purple-500 h-2 rounded-full transition-all duration-500"
                     :style="{ width: currentDemo.progress + '%' }"></div>
              </div>
              <span class="text-sm text-white">{{ currentDemo.progress }}%</span>
            </div>
          </div>

          <!-- Demo Controls -->
          <div class="flex items-center space-x-4 mb-4">
            <button
              @click="runDemo(currentDemo)"
              :disabled="currentDemo.status === 'running'"
              class="px-6 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-lg text-white font-medium transition-all duration-200"
            >
              {{ currentDemo.status === 'running' ? 'Running...' : 'Run Demo' }}
            </button>
            <button
              @click="stopDemo(currentDemo)"
              :disabled="currentDemo.status !== 'running'"
              class="px-6 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 rounded-lg text-white font-medium transition-all duration-200"
            >
              Stop Demo
            </button>
            <button @click="viewDemoResults(currentDemo)" class="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white font-medium transition-all duration-200">
              View Results
            </button>
          </div>
        </div>

        <!-- Demo Selection -->
        <div class="grid grid-cols-2 gap-4">
          <div
            v-for="demo in filteredDemos"
            :key="demo.id"
            @click="selectDemo(demo)"
            class="p-4 rounded-lg bg-white/5 hover:bg-white/10 cursor-pointer transition-all duration-200 border-2"
            :class="currentDemo?.id === demo.id ? 'border-purple-500' : 'border-transparent'"
          >
            <div class="flex items-center space-x-3 mb-2">
              <component :is="demo.icon" class="w-6 h-6 text-purple-400" />
              <div>
                <h5 class="text-white font-medium">{{ demo.name }}</h5>
                <div class="text-xs text-gray-400">{{ demo.category }}</div>
              </div>
            </div>
            <p class="text-sm text-gray-400 mb-3">{{ demo.description }}</p>
            <div class="flex items-center justify-between">
              <span class="text-xs bg-gray-700 px-2 py-1 rounded">{{ demo.difficulty }}</span>
              <span class="text-xs text-gray-500">{{ demo.duration }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Demo Results & Output -->
      <div class="grid grid-cols-2 gap-6">
        <!-- Live Output -->
        <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
          <h3 class="text-lg font-semibold text-white mb-4">Live Output</h3>
          <div class="bg-black/50 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
            <div v-for="log in demoLogs" :key="log.id" class="mb-1">
              <span class="text-gray-500">{{ log.timestamp }}</span>
              <span class="text-green-400">[{{ log.level }}]</span>
              <span class="text-white">{{ log.message }}</span>
            </div>
          </div>
        </div>

        <!-- Demo Results -->
        <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
          <h3 class="text-lg font-semibold text-white mb-4">Demo Results</h3>
          <div v-if="demoResults.length === 0" class="text-center text-gray-400 py-8">
            <Play class="w-8 h-8 mx-auto mb-2 opacity-50" />
            <div class="text-sm">Run a demo to see results</div>
          </div>
          <div v-else class="space-y-3">
            <div v-for="result in demoResults" :key="result.id"
                 class="p-3 rounded-lg bg-white/10">
              <div class="flex items-center justify-between mb-2">
                <span class="text-sm font-medium text-white">{{ result.demoName }}</span>
                <span class="text-xs px-2 py-1 rounded" :class="result.statusColor">{{ result.status }}</span>
              </div>
              <div class="text-xs text-gray-400 mb-2">{{ result.timestamp }}</div>
              <div class="text-sm text-white">{{ result.summary }}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Demo Showcase -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Featured Demo: Iron Man Suit Design</h3>
        <div class="grid grid-cols-3 gap-6">
          <div class="text-center">
            <Bot class="w-12 h-12 mx-auto mb-2 text-purple-400" />
            <div class="text-sm font-medium text-white">Robotics Agent</div>
            <div class="text-xs text-gray-400">20-DOF Exoskeleton</div>
          </div>
          <div class="text-center">
            <Zap class="w-12 h-12 mx-auto mb-2 text-yellow-400" />
            <div class="text-sm font-medium text-white">Power System</div>
            <div class="text-xs text-gray-400">205kW Peak Power</div>
          </div>
          <div class="text-center">
            <Brain class="w-12 h-12 mx-auto mb-2 text-blue-400" />
            <div class="text-sm font-medium text-white">AI Control</div>
            <div class="text-xs text-gray-400">Consciousness-Driven</div>
          </div>
        </div>
        <div class="mt-4 text-center">
          <button @click="runIronManDemo" class="px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 rounded-lg text-white font-medium transition-all duration-200">
            Launch Iron Man Demo
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import {
  Bot,
  Brain,
  Zap,
  Play,
  Cpu,
  Network,
  Wrench,
  Eye
} from 'lucide-vue-next'

// Reactive data
const selectedCategory = ref('all')
const currentDemo = ref(null)
const demoStatus = ref({ text: 'Ready', color: 'bg-green-500' })
const demoLogs = ref([])
const demoResults = ref([])

const demoStats = ref({
  total: 12,
  completed: 8,
  running: 1,
  successRate: 92
})

const demoCategories = [
  { id: 'all', name: 'All Demos', description: 'Complete collection', icon: Play },
  { id: 'engineering', name: 'Engineering', description: 'Complex design tasks', icon: Wrench },
  { id: 'ai', name: 'AI & Consciousness', description: 'Self-aware AI demos', icon: Brain },
  { id: 'robotics', name: 'Robotics', description: 'Autonomous systems', icon: Bot },
  { id: 'science', name: 'Scientific', description: 'Research & analysis', icon: Cpu }
]

const demos = ref([
  {
    id: 'iron_man',
    name: 'Iron Man Suit Design',
    description: 'Complete autonomous design of a flying exoskeleton with AI consciousness',
    category: 'Engineering',
    difficulty: 'Expert',
    duration: '5-10 min',
    icon: Bot,
    status: 'ready',
    progress: 0,
    results: null
  },
  {
    id: 'consciousness_bootstrap',
    name: 'Consciousness Emergence',
    description: 'Demonstrate the emergence of self-aware AI through recursive observation',
    category: 'AI & Consciousness',
    difficulty: 'Advanced',
    duration: '3-5 min',
    icon: Brain,
    status: 'ready',
    progress: 0,
    results: null
  },
  {
    id: 'space_shuttle',
    name: 'Space Shuttle Design',
    description: 'Autonomous spacecraft design with orbital mechanics and propulsion',
    category: 'Engineering',
    difficulty: 'Expert',
    duration: '7-12 min',
    icon: Zap,
    status: 'ready',
    progress: 0,
    results: null
  },
  {
    id: 'neural_network',
    name: 'Neural Architecture Design',
    description: 'AI-designed neural networks with self-optimization capabilities',
    category: 'AI & Consciousness',
    difficulty: 'Advanced',
    duration: '4-6 min',
    icon: Network,
    status: 'ready',
    progress: 0,
    results: null
  },
  {
    id: 'robotic_arm',
    name: 'Advanced Robotic Arm',
    description: 'Design industrial robotic manipulator with precision control',
    category: 'Robotics',
    difficulty: 'Intermediate',
    duration: '3-5 min',
    icon: Wrench,
    status: 'ready',
    progress: 0,
    results: null
  },
  {
    id: 'quantum_simulation',
    name: 'Quantum System Simulation',
    description: 'Simulate quantum computing systems and algorithms',
    category: 'Scientific',
    difficulty: 'Expert',
    duration: '6-8 min',
    icon: Cpu,
    status: 'ready',
    progress: 0,
    results: null
  }
])

// Computed properties
const filteredDemos = computed(() => {
  if (selectedCategory.value === 'all') {
    return demos.value
  }
  return demos.value.filter(demo => {
    const categoryMap = {
      engineering: 'Engineering',
      ai: 'AI & Consciousness',
      robotics: 'Robotics',
      science: 'Scientific'
    }
    return demo.category === categoryMap[selectedCategory.value]
  })
})

// Methods
const selectDemo = (demo) => {
  currentDemo.value = demo
  demoStatus.value = { text: 'Ready', color: 'bg-green-500' }
}

const runDemo = async (demo) => {
  demo.status = 'running'
  demo.progress = 0
  demoStatus.value = { text: 'Running Demo', color: 'bg-yellow-500' }

  demoLogs.value.push({
    id: Date.now(),
    timestamp: new Date().toLocaleTimeString(),
    level: 'INFO',
    message: `Starting demo: ${demo.name}`
  })

  // Simulate demo execution
  const progressInterval = setInterval(() => {
    demo.progress += Math.random() * 15
    if (demo.progress >= 100) {
      demo.progress = 100
      demo.status = 'completed'
      demoStatus.value = { text: 'Demo Complete', color: 'bg-green-500' }
      clearInterval(progressInterval)

      demoLogs.value.push({
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        level: 'SUCCESS',
        message: `Demo completed: ${demo.name}`
      })

      // Add to results
      demoResults.value.unshift({
        id: Date.now(),
        demoName: demo.name,
        status: 'Success',
        statusColor: 'bg-green-600 text-white',
        timestamp: new Date().toLocaleString(),
        summary: `Successfully completed ${demo.name} with full autonomous execution.`
      })
    }
  }, 500)
}

const stopDemo = (demo) => {
  demo.status = 'stopped'
  demoStatus.value = { text: 'Demo Stopped', color: 'bg-red-500' }
  demoLogs.value.push({
    id: Date.now(),
    timestamp: new Date().toLocaleTimeString(),
    level: 'WARNING',
    message: `Demo stopped: ${demo.name}`
  })
}

const viewDemoResults = (demo) => {
  // In real implementation, show detailed results
  console.log('Viewing results for:', demo.name)
}

const runAllDemos = () => {
  // Run all demos sequentially
  console.log('Running all demos...')
}

const stopAllDemos = () => {
  demos.value.forEach(demo => {
    if (demo.status === 'running') {
      stopDemo(demo)
    }
  })
}

const resetDemos = () => {
  demos.value.forEach(demo => {
    demo.status = 'ready'
    demo.progress = 0
  })
  demoResults.value = []
  demoLogs.value = []
  demoStatus.value = { text: 'Ready', color: 'bg-green-500' }
}

const runIronManDemo = async () => {
  try {
    const result = await invoke('run_iron_man_design')
    demoLogs.value.push({
      id: Date.now(),
      timestamp: new Date().toLocaleTimeString(),
      level: 'INFO',
      message: 'Iron Man demo initiated via Tauri backend'
    })
    console.log('Iron Man demo result:', result)
  } catch (error) {
    console.error('Failed to run Iron Man demo:', error)
  }
}
</script>