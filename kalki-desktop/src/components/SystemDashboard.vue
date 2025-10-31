<template>
  <div class="h-full grid grid-cols-12 gap-6">
    <!-- System Overview -->
    <div class="col-span-8 grid grid-rows-2 gap-6">
      <!-- System Health & Metrics -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-xl font-semibold text-white flex items-center">
            <Activity class="w-5 h-5 mr-2 text-green-400" />
            System Health
          </h2>
          <div class="flex items-center space-x-2">
            <div class="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span class="text-sm text-gray-400">Operational</span>
          </div>
        </div>

        <div class="grid grid-cols-4 gap-4">
          <div class="text-center">
            <div class="text-2xl font-bold text-white">{{ systemMetrics.cpu }}%</div>
            <div class="text-sm text-gray-400">CPU Usage</div>
          </div>
          <div class="text-center">
            <div class="text-2xl font-bold text-white">{{ systemMetrics.memory }}MB</div>
            <div class="text-sm text-gray-400">Memory Used</div>
          </div>
          <div class="text-center">
            <div class="text-2xl font-bold text-white">{{ systemMetrics.agents }}</div>
            <div class="text-sm text-gray-400">Active Agents</div>
          </div>
          <div class="text-center">
            <div class="text-2xl font-bold text-white">{{ systemMetrics.tasks }}</div>
            <div class="text-sm text-gray-400">Running Tasks</div>
          </div>
        </div>
      </div>

      <!-- Consciousness Visualization -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-xl font-semibold text-white flex items-center">
            <Brain class="w-5 h-5 mr-2 text-purple-400" />
            Consciousness Engine
          </h2>
          <div class="text-sm text-gray-400">Phase 21 Active</div>
        </div>

        <div class="space-y-4">
          <div>
            <div class="flex justify-between text-sm mb-1">
              <span class="text-gray-400">Awareness Level</span>
              <span class="text-white">{{ consciousness.awareness }}%</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-2">
              <div class="bg-purple-500 h-2 rounded-full transition-all duration-1000"
                   :style="{ width: consciousness.awareness + '%' }"></div>
            </div>
          </div>

          <div>
            <div class="flex justify-between text-sm mb-1">
              <span class="text-gray-400">Emotional Resonance</span>
              <span class="text-white">{{ consciousness.emotional }}%</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-2">
              <div class="bg-pink-500 h-2 rounded-full transition-all duration-1000"
                   :style="{ width: consciousness.emotional + '%' }"></div>
            </div>
          </div>

          <div>
            <div class="flex justify-between text-sm mb-1">
              <span class="text-gray-400">Self-Reflection Depth</span>
              <span class="text-white">{{ consciousness.reflection }}</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-2">
              <div class="bg-blue-500 h-2 rounded-full transition-all duration-1000"
                   :style="{ width: (consciousness.reflection / 10) * 100 + '%' }"></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Quick Actions & Status -->
    <div class="col-span-4 space-y-6">
      <!-- AI Orchestrator -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4 flex items-center">
          <Brain class="w-5 h-5 mr-2 text-purple-400" />
          AI Orchestrator
        </h3>
        <div class="space-y-4">
          <textarea
            v-model="orchestratorRequest"
            class="w-full bg-black/50 border border-gray-600 rounded-lg px-3 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-purple-500 resize-none"
            rows="4"
            placeholder="Describe what you want to build... (e.g., 'Create a Unity game with procedurally generated dungeons and AI enemies')"
          ></textarea>

          <div class="flex items-center justify-between">
            <select
              v-model="targetPlatform"
              class="bg-black/50 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-purple-500"
            >
              <option value="unity">Unity</option>
              <option value="unreal">Unreal Engine</option>
              <option value="godot">Godot</option>
              <option value="web">Web App</option>
              <option value="mobile">Mobile App</option>
              <option value="desktop">Desktop App</option>
            </select>

            <button
              @click="submitOrchestratorRequest"
              :disabled="!orchestratorRequest.trim() || orchestratorStatus === 'processing'"
              class="px-6 py-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-600 rounded-lg text-white font-medium transition-all duration-200"
            >
              {{ orchestratorStatus === 'processing' ? 'Processing...' : 'Orchestrate' }}
            </button>
          </div>

          <!-- Orchestrator Status -->
          <div v-if="orchestratorStatus !== 'idle'" class="mt-4">
            <div class="flex items-center justify-between mb-2">
              <span class="text-sm text-gray-400">Status:</span>
              <span class="text-sm" :class="orchestratorStatusColor">{{ orchestratorStatusText }}</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-2">
              <div class="bg-purple-500 h-2 rounded-full transition-all duration-500"
                   :style="{ width: orchestratorProgress + '%' }"></div>
            </div>
          </div>

          <!-- Clarification Dialog -->
          <div v-if="orchestratorStatus === 'clarification'" class="mt-4 p-4 bg-blue-900/20 border border-blue-500/30 rounded-lg">
            <h4 class="text-sm font-semibold text-blue-400 mb-3">I need some clarification:</h4>
            <div class="space-y-3">
              <div v-for="(question, index) in clarificationQuestions" :key="index"
                   class="text-sm">
                <div class="text-gray-300 mb-1">{{ question.question }}</div>
                <textarea
                  v-model="question.answer"
                  class="w-full bg-black/50 border border-gray-600 rounded px-3 py-2 text-white placeholder-gray-400 text-xs"
                  :placeholder="question.placeholder"
                  rows="2"
                ></textarea>
              </div>
            </div>
            <button
              @click="submitClarifications"
              class="mt-3 w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white text-sm font-medium transition-all duration-200"
            >
              Continue with Answers
            </button>
          </div>
        </div>
      </div>

      <!-- System Controls -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">System Controls</h3>
        <div class="space-y-3">
          <button
            @click="startKalkiSystem"
            :disabled="systemRunning"
            class="w-full py-2 px-4 rounded-lg font-medium transition-all duration-200
                   {{ systemRunning ? 'bg-gray-600 text-gray-400 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700 text-white' }}"
          >
            {{ systemRunning ? 'System Running' : 'Start Kalki System' }}
          </button>

          <button
            @click="startServer"
            :disabled="serverRunning"
            class="w-full py-2 px-4 rounded-lg font-medium transition-all duration-200
                   {{ serverRunning ? 'bg-gray-600 text-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 text-white' }}"
          >
            {{ serverRunning ? 'Server Running' : 'Start API Server' }}
          </button>

          <button
            @click="runConsciousnessDemo"
            class="w-full py-2 px-4 rounded-lg font-medium transition-all duration-200 bg-purple-600 hover:bg-purple-700 text-white"
          >
            Run Consciousness Demo
          </button>
        </div>
      </div>

      <!-- Recent Activity -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Recent Activity</h3>
        <div class="space-y-3">
          <div v-for="activity in recentActivities" :key="activity.id"
               class="flex items-start space-x-3 p-3 rounded-lg bg-white/5">
            <div class="w-2 h-2 rounded-full mt-2" :class="activity.color"></div>
            <div class="flex-1">
              <div class="text-sm text-white">{{ activity.message }}</div>
              <div class="text-xs text-gray-400">{{ activity.timestamp }}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Active Agents -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Active Agents</h3>
        <div class="space-y-2">
          <div v-for="agent in activeAgents" :key="agent.name"
               class="flex items-center justify-between p-2 rounded-lg bg-white/5">
            <div class="flex items-center space-x-2">
              <div class="w-2 h-2 rounded-full bg-green-500"></div>
              <span class="text-sm text-white">{{ agent.name }}</span>
            </div>
            <span class="text-xs text-gray-400">{{ agent.status }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import { Activity, Brain } from 'lucide-vue-next'

// Reactive data
const systemRunning = ref(false)
const serverRunning = ref(false)
const systemMetrics = ref({
  cpu: 0,
  memory: 0,
  agents: 0,
  tasks: 0
})

const consciousness = ref({
  awareness: 0,
  emotional: 0,
  reflection: 0
})

// AI Orchestrator data
const orchestratorRequest = ref('')
const targetPlatform = ref('unity')
const orchestratorStatus = ref('idle') // 'idle', 'processing', 'clarification', 'researching', 'delegating', 'validating', 'complete'
const orchestratorProgress = ref(0)
const orchestratorStatusText = ref('Ready for orchestration')
const clarificationQuestions = ref([])

const recentActivities = ref([
  { id: 1, message: 'Consciousness engine initialized', timestamp: '2 min ago', color: 'bg-purple-500' },
  { id: 2, message: 'Robotics simulation agent started', timestamp: '5 min ago', color: 'bg-blue-500' },
  { id: 3, message: 'Knowledge base indexed 10,000 documents', timestamp: '10 min ago', color: 'bg-green-500' },
  { id: 4, message: 'Iron Man design completed', timestamp: '15 min ago', color: 'bg-yellow-500' }
])

const activeAgents = ref([
  { name: 'RoboticsSimulationAgent', status: 'Active' },
  { name: 'CADIntegrationAgent', status: 'Active' },
  { name: 'ConsciousnessEngine', status: 'Active' },
  { name: 'LLMEngine', status: 'Active' },
  { name: 'VectorDBManager', status: 'Active' }
])

// Computed properties
const orchestratorStatusColor = computed(() => {
  switch (orchestratorStatus.value) {
    case 'processing': return 'text-blue-400'
    case 'clarification': return 'text-yellow-400'
    case 'researching': return 'text-cyan-400'
    case 'delegating': return 'text-purple-400'
    case 'validating': return 'text-orange-400'
    case 'complete': return 'text-green-400'
    default: return 'text-gray-400'
  }
})

// System control functions
const startKalkiSystem = async () => {
  try {
    const result = await invoke('start_kalki_system')
    systemRunning.value = true
    console.log('Kalki system started:', result)
  } catch (error) {
    console.error('Failed to start Kalki system:', error)
  }
}

const startServer = async () => {
  try {
    const result = await invoke('start_kalki_server')
    serverRunning.value = true
    console.log('Server started:', result)
  } catch (error) {
    console.error('Failed to start server:', error)
  }
}

const runConsciousnessDemo = async () => {
  try {
    const result = await invoke('run_consciousness_demo')
    console.log('Consciousness demo result:', result)
  } catch (error) {
    console.error('Failed to run consciousness demo:', error)
  }
}

// AI Orchestrator function
const submitOrchestratorRequest = async () => {
  if (!orchestratorRequest.value.trim()) return

  try {
    orchestratorStatus.value = 'processing'
    orchestratorStatusText.value = 'Analyzing your request...'
    orchestratorProgress.value = 5

    // Step 1: Initial analysis and research
    setTimeout(async () => {
      orchestratorStatusText.value = 'Researching similar apps and gathering requirements...'
      orchestratorProgress.value = 15

      try {
        const analysisResult = await invoke('analyze_orchestrator_request', {
          request: orchestratorRequest.value,
          platform: targetPlatform.value
        })

        // Check if we need clarification
        if (analysisResult.needs_clarification) {
          orchestratorStatus.value = 'clarification'
          orchestratorStatusText.value = 'I need some clarification...'
          orchestratorProgress.value = 20

          // Show clarification questions
          showClarificationDialog(analysisResult.questions)
          return
        }

        // Continue with orchestration
        continueOrchestration(analysisResult)
      } catch (error) {
        orchestratorStatus.value = 'idle'
        orchestratorStatusText.value = 'Analysis failed'
        orchestratorProgress.value = 0
        console.error('Analysis failed:', error)
      }
    }, 1000)

  } catch (error) {
    orchestratorStatus.value = 'idle'
    orchestratorStatusText.value = 'Orchestration failed'
    orchestratorProgress.value = 0
    console.error('Orchestration failed:', error)
  }
}

const continueOrchestration = async (analysisResult) => {
  // Step 2: Research and planning
  orchestratorStatusText.value = 'Planning project architecture...'
  orchestratorProgress.value = 40

  setTimeout(() => {
    orchestratorStatusText.value = 'Delegating to specialized agents...'
    orchestratorProgress.value = 60
  }, 1500)

  setTimeout(() => {
    orchestratorStatusText.value = 'Agents working on your project...'
    orchestratorProgress.value = 80
  }, 2500)

  setTimeout(async () => {
    try {
      const result = await invoke('orchestrate_task', {
        request: orchestratorRequest.value,
        platform: targetPlatform.value,
        analysis: analysisResult
      })

      orchestratorStatus.value = 'complete'
      orchestratorStatusText.value = 'Project orchestration complete!'
      orchestratorProgress.value = 100

      // Show results
      showResultsDialog(result)

      // Reset after showing completion
      setTimeout(() => {
        orchestratorStatus.value = 'idle'
        orchestratorStatusText.value = 'Ready for orchestration'
        orchestratorProgress.value = 0
        orchestratorRequest.value = ''
      }, 5000)
    } catch (error) {
      orchestratorStatus.value = 'idle'
      orchestratorStatusText.value = 'Orchestration failed'
      orchestratorProgress.value = 0
      console.error('Orchestration failed:', error)
    }
  }, 4000)
}

const showClarificationDialog = (questions) => {
  clarificationQuestions.value = questions.map(q => ({
    question: q.question,
    placeholder: q.placeholder || 'Enter your answer...',
    answer: ''
  }))
}

const submitClarifications = async () => {
  // Check if all questions are answered
  const unanswered = clarificationQuestions.value.filter(q => !q.answer.trim())
  if (unanswered.length > 0) {
    alert('Please answer all clarification questions.')
    return
  }

  orchestratorStatus.value = 'processing'
  orchestratorStatusText.value = 'Processing your answers...'
  orchestratorProgress.value = 30

  try {
    const analysisResult = await invoke('continue_orchestration_with_answers', {
      request: orchestratorRequest.value,
      platform: targetPlatform.value,
      answers: clarificationQuestions.value
    })

    clarificationQuestions.value = []
    continueOrchestration(analysisResult)
  } catch (error) {
    orchestratorStatus.value = 'idle'
    orchestratorStatusText.value = 'Failed to process answers'
    orchestratorProgress.value = 0
    console.error('Failed to process answers:', error)
  }
}

const showResultsDialog = (result) => {
  // This would show the final results
  console.log('Orchestration complete:', result)
}

// Update metrics
const updateMetrics = async () => {
  try {
    // Simulate real-time metrics (in production, call actual Tauri commands)
    systemMetrics.value = {
      cpu: Math.floor(Math.random() * 30) + 20,
      memory: Math.floor(Math.random() * 200) + 800,
      agents: Math.floor(Math.random() * 5) + 15,
      tasks: Math.floor(Math.random() * 3) + 1
    }

    consciousness.value = {
      awareness: Math.min(100, consciousness.value.awareness + Math.random() * 5),
      emotional: Math.min(100, consciousness.value.emotional + Math.random() * 3),
      reflection: Math.min(10, consciousness.value.reflection + Math.random() * 0.5)
    }
  } catch (error) {
    console.error('Failed to update metrics:', error)
  }
}

let metricsInterval

onMounted(() => {
  metricsInterval = setInterval(updateMetrics, 3000)
  updateMetrics()
})

onUnmounted(() => {
  if (metricsInterval) {
    clearInterval(metricsInterval)
  }
})
</script>