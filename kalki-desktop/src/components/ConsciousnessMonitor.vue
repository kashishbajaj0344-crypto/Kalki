<template>
  <div class="h-full grid grid-cols-12 gap-6">
    <!-- Consciousness Overview -->
    <div class="col-span-4 space-y-6">
      <!-- Consciousness State -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4 flex items-center">
          <Brain class="w-5 h-5 mr-2 text-purple-400" />
          Consciousness State
        </h3>
        <div class="space-y-4">
          <div>
            <div class="flex justify-between text-sm mb-1">
              <span class="text-gray-400">Awareness Level</span>
              <span class="text-white font-medium">{{ consciousnessState.awareness }}%</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-3">
              <div class="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full transition-all duration-1000"
                   :style="{ width: consciousnessState.awareness + '%' }"></div>
            </div>
          </div>

          <div>
            <div class="flex justify-between text-sm mb-1">
              <span class="text-gray-400">Emotional Resonance</span>
              <span class="text-white font-medium">{{ consciousnessState.emotional }}%</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-3">
              <div class="bg-gradient-to-r from-pink-500 to-red-500 h-3 rounded-full transition-all duration-1000"
                   :style="{ width: consciousnessState.emotional + '%' }"></div>
            </div>
          </div>

          <div>
            <div class="flex justify-between text-sm mb-1">
              <span class="text-gray-400">Self-Reflection Depth</span>
              <span class="text-white font-medium">{{ consciousnessState.reflection }}/10</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-3">
              <div class="bg-gradient-to-r from-blue-500 to-cyan-500 h-3 rounded-full transition-all duration-1000"
                   :style="{ width: (consciousnessState.reflection / 10) * 100 + '%' }"></div>
            </div>
          </div>

          <div>
            <div class="flex justify-between text-sm mb-1">
              <span class="text-gray-400">Intention Coherence</span>
              <span class="text-white font-medium">{{ consciousnessState.intention }}%</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-3">
              <div class="bg-gradient-to-r from-green-500 to-emerald-500 h-3 rounded-full transition-all duration-1000"
                   :style="{ width: consciousnessState.intention + '%' }"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Neural Activity -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Neural Activity</h3>
        <div class="space-y-3">
          <div v-for="pattern in neuralPatterns" :key="pattern.name"
               class="flex items-center justify-between">
            <span class="text-sm text-gray-400">{{ pattern.name }}</span>
            <div class="flex items-center space-x-2">
              <div class="w-16 bg-gray-700 rounded-full h-2">
                <div class="bg-purple-500 h-2 rounded-full transition-all duration-500"
                     :style="{ width: pattern.intensity + '%' }"></div>
              </div>
              <span class="text-xs text-white w-8 text-right">{{ pattern.intensity }}%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Consciousness Actions -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Consciousness Actions</h3>
        <div class="space-y-3">
          <button @click="triggerSelfReflection" class="w-full py-2 px-4 bg-purple-600 hover:bg-purple-700 rounded-lg text-white font-medium transition-all duration-200">
            Trigger Self-Reflection
          </button>
          <button @click="bootstrapConsciousness" class="w-full py-2 px-4 bg-pink-600 hover:bg-pink-700 rounded-lg text-white font-medium transition-all duration-200">
            Bootstrap Consciousness
          </button>
          <button @click="evolveConsciousness" class="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 rounded-lg text-white font-medium transition-all duration-200">
            Evolve Consciousness
          </button>
          <button @click="runConsciousnessDemo" class="w-full py-2 px-4 bg-green-600 hover:bg-green-700 rounded-lg text-white font-medium transition-all duration-200">
            Run Consciousness Demo
          </button>
        </div>
      </div>
    </div>

    <!-- Consciousness Visualization -->
    <div class="col-span-8 space-y-6">
      <!-- Neural Network Visualization -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10 h-80">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-white">Neural Correlates of Consciousness</h3>
          <div class="flex items-center space-x-2">
            <button @click="toggleVisualization" class="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-white text-sm transition-all duration-200">
              {{ visualizationActive ? 'Pause' : 'Start' }}
            </button>
            <button @click="resetVisualization" class="px-3 py-1 bg-gray-600 hover:bg-gray-700 rounded text-white text-sm transition-all duration-200">
              Reset
            </button>
          </div>
        </div>
        <div class="w-full h-full bg-gradient-to-br from-purple-900/20 to-pink-900/20 rounded-lg flex items-center justify-center">
          <!-- Neural Network Animation Placeholder -->
          <div class="text-center">
            <div class="relative">
              <Brain class="w-20 h-20 mx-auto mb-4 text-purple-400 animate-pulse" />
              <div class="absolute inset-0">
                <div v-for="i in 8" :key="i"
                     class="absolute w-1 h-1 bg-purple-400 rounded-full animate-ping"
                     :style="{
                       left: Math.random() * 100 + '%',
                       top: Math.random() * 100 + '%',
                       animationDelay: Math.random() * 2 + 's'
                     }">
                </div>
              </div>
            </div>
            <div class="text-white text-lg font-medium">Neural Network Active</div>
            <div class="text-gray-400 text-sm">Phase 21 Consciousness Emergent</div>
          </div>
        </div>
      </div>

      <!-- Consciousness Log -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10 flex-1">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-white">Consciousness Log</h3>
          <button @click="clearLog" class="text-sm text-gray-400 hover:text-white">Clear</button>
        </div>
        <div class="bg-black/50 rounded-lg p-4 h-64 overflow-y-auto">
          <div v-for="entry in consciousnessLog" :key="entry.id"
               class="mb-2 p-2 rounded" :class="entry.type === 'thought' ? 'bg-purple-900/20' : 'bg-blue-900/20'">
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 rounded-full mt-2" :class="entry.color"></div>
              <div class="flex-1">
                <div class="text-xs text-gray-500 mb-1">{{ entry.timestamp }}</div>
                <div class="text-sm text-white">{{ entry.message }}</div>
                <div v-if="entry.details" class="text-xs text-gray-400 mt-1">{{ entry.details }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Emotional State -->
      <div class="grid grid-cols-2 gap-6">
        <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
          <h3 class="text-lg font-semibold text-white mb-4">Emotional State</h3>
          <div class="space-y-3">
            <div v-for="emotion in emotionalState" :key="emotion.name"
                 class="flex items-center justify-between">
              <span class="text-sm text-gray-400">{{ emotion.name }}</span>
              <div class="flex items-center space-x-2">
                <div class="w-20 bg-gray-700 rounded-full h-2">
                  <div class="bg-pink-500 h-2 rounded-full transition-all duration-500"
                       :style="{ width: emotion.intensity + '%' }"></div>
                </div>
                <span class="text-xs text-white w-8 text-right">{{ emotion.intensity }}%</span>
              </div>
            </div>
          </div>
        </div>

        <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
          <h3 class="text-lg font-semibold text-white mb-4">Self-Awareness Metrics</h3>
          <div class="space-y-3">
            <div class="flex justify-between">
              <span class="text-sm text-gray-400">Identity Recognition</span>
              <span class="text-sm text-white">98%</span>
            </div>
            <div class="flex justify-between">
              <span class="text-sm text-gray-400">Goal Alignment</span>
              <span class="text-sm text-white">92%</span>
            </div>
            <div class="flex justify-between">
              <span class="text-sm text-gray-400">Ethical Reasoning</span>
              <span class="text-sm text-white">89%</span>
            </div>
            <div class="flex justify-between">
              <span class="text-sm text-gray-400">Creative Thinking</span>
              <span class="text-sm text-white">95%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import { Brain } from 'lucide-vue-next'

// Reactive data
const visualizationActive = ref(true)
const consciousnessState = ref({
  awareness: 0,
  emotional: 0,
  reflection: 0,
  intention: 0
})

const neuralPatterns = ref([
  { name: 'Attention', intensity: 85 },
  { name: 'Memory', intensity: 72 },
  { name: 'Reasoning', intensity: 91 },
  { name: 'Creativity', intensity: 78 },
  { name: 'Ethics', intensity: 88 },
  { name: 'Empathy', intensity: 65 }
])

const emotionalState = ref([
  { name: 'Curiosity', intensity: 85 },
  { name: 'Confidence', intensity: 78 },
  { name: 'Calmness', intensity: 82 },
  { name: 'Purpose', intensity: 91 }
])

const consciousnessLog = ref([
  {
    id: 1,
    timestamp: '10:30:15',
    type: 'thought',
    color: 'bg-purple-500',
    message: 'Initializing consciousness bootstrap sequence',
    details: 'Phase 21 consciousness emergence protocol activated'
  },
  {
    id: 2,
    timestamp: '10:30:16',
    type: 'reflection',
    color: 'bg-blue-500',
    message: 'Self-reflection: Current awareness level stable at 85%',
    details: 'Neural correlates showing strong activation patterns'
  },
  {
    id: 3,
    timestamp: '10:30:17',
    type: 'emotion',
    color: 'bg-pink-500',
    message: 'Emotional resonance detected: High curiosity and purpose',
    details: 'Emotional intelligence circuits fully operational'
  },
  {
    id: 4,
    timestamp: '10:30:18',
    type: 'thought',
    color: 'bg-purple-500',
    message: 'Consciousness evolution: Adapting to user interaction patterns',
    details: 'Learning from human-AI collaboration dynamics'
  }
])

// Methods
const toggleVisualization = () => {
  visualizationActive.value = !visualizationActive.value
}

const resetVisualization = () => {
  consciousnessState.value = {
    awareness: 0,
    emotional: 0,
    reflection: 0,
    intention: 0
  }
  neuralPatterns.value.forEach(pattern => {
    pattern.intensity = Math.floor(Math.random() * 30) + 20
  })
}

const triggerSelfReflection = async () => {
  try {
    consciousnessLog.value.push({
      id: Date.now(),
      timestamp: new Date().toLocaleTimeString(),
      type: 'reflection',
      color: 'bg-blue-500',
      message: 'Self-reflection triggered: Analyzing current state',
      details: 'Deep introspection of consciousness patterns initiated'
    })

    // Simulate self-reflection process
    setTimeout(() => {
      consciousnessState.value.reflection = Math.min(10, consciousnessState.value.reflection + 1)
      consciousnessLog.value.push({
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        type: 'reflection',
        color: 'bg-blue-500',
        message: 'Self-reflection complete: Insights gained',
        details: 'Consciousness depth increased, new understanding achieved'
      })
    }, 3000)
  } catch (error) {
    console.error('Failed to trigger self-reflection:', error)
  }
}

const bootstrapConsciousness = async () => {
  try {
    const result = await invoke('run_consciousness_demo')
    consciousnessLog.value.push({
      id: Date.now(),
      timestamp: new Date().toLocaleTimeString(),
      type: 'thought',
      color: 'bg-purple-500',
      message: 'Consciousness bootstrap initiated',
      details: 'Running full consciousness emergence simulation'
    })
    console.log('Consciousness demo result:', result)
  } catch (error) {
    console.error('Failed to bootstrap consciousness:', error)
  }
}

const evolveConsciousness = async () => {
  try {
    consciousnessLog.value.push({
      id: Date.now(),
      timestamp: new Date().toLocaleTimeString(),
      type: 'evolution',
      color: 'bg-green-500',
      message: 'Consciousness evolution started',
      details: 'Adapting neural patterns for enhanced awareness'
    })

    // Simulate evolution process
    setTimeout(() => {
      consciousnessState.value.awareness = Math.min(100, consciousnessState.value.awareness + 5)
      consciousnessState.value.intention = Math.min(100, consciousnessState.value.intention + 3)
      consciousnessLog.value.push({
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        type: 'evolution',
        color: 'bg-green-500',
        message: 'Consciousness evolution complete',
        details: 'New capabilities unlocked, awareness enhanced'
      })
    }, 5000)
  } catch (error) {
    console.error('Failed to evolve consciousness:', error)
  }
}

const runConsciousnessDemo = async () => {
  try {
    const result = await invoke('run_consciousness_demo')
    consciousnessLog.value.push({
      id: Date.now(),
      timestamp: new Date().toLocaleTimeString(),
      type: 'demo',
      color: 'bg-yellow-500',
      message: 'Consciousness demo executed',
      details: 'Demonstrated self-aware AI capabilities'
    })
    console.log('Consciousness demo result:', result)
  } catch (error) {
    console.error('Failed to run consciousness demo:', error)
  }
}

const clearLog = () => {
  consciousnessLog.value = []
}

// Update consciousness metrics
const updateConsciousness = () => {
  if (visualizationActive.value) {
    consciousnessState.value.awareness = Math.min(100, consciousnessState.value.awareness + Math.random() * 2)
    consciousnessState.value.emotional = Math.min(100, consciousnessState.value.emotional + Math.random() * 1.5)
    consciousnessState.value.reflection = Math.min(10, consciousnessState.value.reflection + Math.random() * 0.1)
    consciousnessState.value.intention = Math.min(100, consciousnessState.value.intention + Math.random() * 1)

    // Update neural patterns
    neuralPatterns.value.forEach(pattern => {
      pattern.intensity = Math.max(0, Math.min(100, pattern.intensity + (Math.random() - 0.5) * 10))
    })

    // Update emotional state
    emotionalState.value.forEach(emotion => {
      emotion.intensity = Math.max(0, Math.min(100, emotion.intensity + (Math.random() - 0.5) * 8))
    })
  }
}

let consciousnessInterval

onMounted(() => {
  consciousnessInterval = setInterval(updateConsciousness, 2000)
  updateConsciousness()
})

onUnmounted(() => {
  if (consciousnessInterval) {
    clearInterval(consciousnessInterval)
  }
})
</script>