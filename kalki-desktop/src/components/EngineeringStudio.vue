<template>
  <div class="h-full grid grid-cols-12 gap-6">
    <!-- Tools Panel -->
    <div class="col-span-3 space-y-6">
      <!-- Engineering Tools -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Engineering Tools</h3>
        <div class="space-y-3">
          <button
            v-for="tool in engineeringTools"
            :key="tool.id"
            @click="selectTool(tool)"
            :class="[
              'w-full text-left p-3 rounded-lg transition-all duration-200',
              selectedTool?.id === tool.id
                ? 'bg-purple-600 text-white'
                : 'bg-white/10 hover:bg-white/20 text-gray-300'
            ]"
          >
            <div class="flex items-center space-x-3">
              <component :is="tool.icon" class="w-5 h-5" />
              <div>
                <div class="text-sm font-medium">{{ tool.name }}</div>
                <div class="text-xs text-gray-400">{{ tool.description }}</div>
              </div>
            </div>
          </button>
        </div>
      </div>

      <!-- Project Templates -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Project Templates</h3>
        <div class="space-y-2">
          <button
            v-for="template in projectTemplates"
            :key="template.id"
            @click="loadTemplate(template)"
            class="w-full text-left p-3 rounded-lg bg-white/10 hover:bg-white/20 transition-all duration-200"
          >
            <div class="text-sm font-medium text-white">{{ template.name }}</div>
            <div class="text-xs text-gray-400">{{ template.description }}</div>
          </button>
        </div>
      </div>

      <!-- Current Project -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Current Project</h3>
        <div v-if="currentProject" class="space-y-3">
          <div>
            <div class="text-sm font-medium text-white">{{ currentProject.name }}</div>
            <div class="text-xs text-gray-400">{{ currentProject.type }}</div>
          </div>
          <div class="space-y-2">
            <div class="flex justify-between text-xs">
              <span class="text-gray-400">Progress</span>
              <span class="text-white">{{ currentProject.progress }}%</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-2">
              <div class="bg-purple-500 h-2 rounded-full transition-all duration-500"
                   :style="{ width: currentProject.progress + '%' }"></div>
            </div>
          </div>
          <div class="flex space-x-2">
            <button @click="saveProject" class="flex-1 text-xs py-2 bg-blue-600 hover:bg-blue-700 rounded text-white transition-all duration-200">
              Save
            </button>
            <button @click="exportProject" class="flex-1 text-xs py-2 bg-green-600 hover:bg-green-700 rounded text-white transition-all duration-200">
              Export
            </button>
          </div>
        </div>
        <div v-else class="text-center text-gray-400 py-8">
          <Wrench class="w-8 h-8 mx-auto mb-2 opacity-50" />
          <div class="text-sm">No active project</div>
        </div>
      </div>
    </div>

    <!-- Main Workspace -->
    <div class="col-span-9 space-y-6">
      <!-- 3D Viewer -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl border border-white/10 h-96">
        <div class="p-4 border-b border-white/10">
          <div class="flex items-center justify-between">
            <h3 class="text-lg font-semibold text-white">3D Design Workspace</h3>
            <div class="flex items-center space-x-2">
              <button @click="zoomIn" class="p-2 bg-white/10 hover:bg-white/20 rounded text-white transition-all duration-200">
                <ZoomIn class="w-4 h-4" />
              </button>
              <button @click="zoomOut" class="p-2 bg-white/10 hover:bg-white/20 rounded text-white transition-all duration-200">
                <ZoomOut class="w-4 h-4" />
              </button>
              <button @click="resetView" class="p-2 bg-white/10 hover:bg-white/20 rounded text-white transition-all duration-200">
                <RotateCcw class="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
        <div class="flex-1 p-4">
          <!-- 3D Canvas Placeholder -->
          <div class="w-full h-full bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg flex items-center justify-center">
            <div v-if="selectedTool?.id === 'robotics'" class="text-center">
              <Bot class="w-16 h-16 mx-auto mb-4 text-purple-400" />
              <div class="text-white text-lg font-medium">Robotics Simulation Active</div>
              <div class="text-gray-400 text-sm">20-DOF Exoskeleton Model Loaded</div>
            </div>
            <div v-else-if="selectedTool?.id === 'cad'" class="text-center">
              <Box class="w-16 h-16 mx-auto mb-4 text-blue-400" />
              <div class="text-white text-lg font-medium">CAD Integration Active</div>
              <div class="text-gray-400 text-sm">3D Modeling Environment Ready</div>
            </div>
            <div v-else class="text-center">
              <Wrench class="w-16 h-16 mx-auto mb-4 text-gray-400" />
              <div class="text-gray-400 text-lg">Select a tool to begin</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Parameters Panel -->
      <div class="grid grid-cols-2 gap-6">
        <!-- Design Parameters -->
        <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
          <h3 class="text-lg font-semibold text-white mb-4">Design Parameters</h3>
          <div class="space-y-4">
            <div v-for="param in designParameters" :key="param.id">
              <label class="block text-sm font-medium text-gray-300 mb-1">{{ param.name }}</label>
              <input
                v-model="param.value"
                :type="param.type"
                :min="param.min"
                :max="param.max"
                :step="param.step"
                class="w-full bg-black/50 border border-gray-600 rounded-lg px-3 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
              />
            </div>
          </div>
          <button @click="applyParameters" class="w-full mt-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white font-medium transition-all duration-200">
            Apply Parameters
          </button>
        </div>

        <!-- Simulation Results -->
        <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
          <h3 class="text-lg font-semibold text-white mb-4">Simulation Results</h3>
          <div v-if="simulationResults" class="space-y-4">
            <div v-for="result in simulationResults" :key="result.id" class="flex justify-between">
              <span class="text-gray-400">{{ result.name }}</span>
              <span class="text-white font-medium">{{ result.value }}</span>
            </div>
          </div>
          <div v-else class="text-center text-gray-400 py-8">
            <Activity class="w-8 h-8 mx-auto mb-2 opacity-50" />
            <div class="text-sm">Run simulation to see results</div>
          </div>
          <button @click="runSimulation" class="w-full mt-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-white font-medium transition-all duration-200">
            Run Simulation
          </button>
        </div>
      </div>

      <!-- Engineering Tasks -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-white">Engineering Tasks</h3>
          <button @click="createTask" class="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white text-sm font-medium transition-all duration-200">
            New Task
          </button>
        </div>
        <div class="space-y-3">
          <div v-for="task in engineeringTasks" :key="task.id"
               class="flex items-center justify-between p-3 rounded-lg bg-white/5">
            <div class="flex items-center space-x-3">
              <div class="w-2 h-2 rounded-full" :class="task.statusColor"></div>
              <div>
                <div class="text-sm font-medium text-white">{{ task.name }}</div>
                <div class="text-xs text-gray-400">{{ task.description }}</div>
              </div>
            </div>
            <div class="flex items-center space-x-2">
              <span class="text-xs text-gray-400">{{ task.progress }}%</span>
              <button @click="viewTask(task)" class="text-xs px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white transition-all duration-200">
                View
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import {
  Wrench,
  Bot,
  Box,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Activity
} from 'lucide-vue-next'

// Reactive data
const selectedTool = ref(null)
const currentProject = ref({
  name: 'Iron Man Suit v2.0',
  type: 'Flight Exoskeleton',
  progress: 75
})

const simulationResults = ref(null)

const engineeringTools = [
  {
    id: 'robotics',
    name: 'Robotics Simulation',
    description: 'Physics-based robot simulation',
    icon: Bot
  },
  {
    id: 'cad',
    name: 'CAD Integration',
    description: '3D modeling and design',
    icon: Box
  },
  {
    id: 'kinematics',
    name: 'Kinematics Analysis',
    description: 'Motion planning and analysis',
    icon: Activity
  },
  {
    id: 'control',
    name: 'Control Systems',
    description: 'PID controllers and automation',
    icon: Wrench
  }
]

const projectTemplates = [
  { id: 'exoskeleton', name: 'Powered Exoskeleton', description: 'Human augmentation suit' },
  { id: 'drone', name: 'Autonomous Drone', description: 'Flying robot platform' },
  { id: 'robotic_arm', name: 'Robotic Arm', description: 'Industrial manipulator' },
  { id: 'mobile_robot', name: 'Mobile Robot', description: 'Autonomous navigation' }
]

const designParameters = [
  { id: 'mass', name: 'Total Mass (kg)', type: 'number', value: 40, min: 10, max: 200, step: 1 },
  { id: 'dof', name: 'Degrees of Freedom', type: 'number', value: 20, min: 6, max: 50, step: 1 },
  { id: 'power', name: 'Peak Power (kW)', type: 'number', value: 205, min: 10, max: 500, step: 5 },
  { id: 'thrust', name: 'Thrust-to-Weight Ratio', type: 'number', value: 2.0, min: 1.0, max: 5.0, step: 0.1 }
]

const engineeringTasks = [
  {
    id: 1,
    name: 'Suit Structure Design',
    description: 'Design mechanical exoskeleton structure',
    status: 'completed',
    statusColor: 'bg-green-500',
    progress: 100
  },
  {
    id: 2,
    name: 'Propulsion System',
    description: 'Design flight propulsion and control',
    status: 'in_progress',
    statusColor: 'bg-yellow-500',
    progress: 75
  },
  {
    id: 3,
    name: 'Power Management',
    description: 'Implement power distribution system',
    status: 'pending',
    statusColor: 'bg-gray-500',
    progress: 0
  }
]

// Methods
const selectTool = (tool) => {
  selectedTool.value = tool
}

const loadTemplate = async (template) => {
  try {
    currentProject.value = {
      name: template.name,
      type: template.description,
      progress: 0
    }
    // In real implementation, load template data
    console.log('Loading template:', template.id)
  } catch (error) {
    console.error('Failed to load template:', error)
  }
}

const saveProject = async () => {
  try {
    // In real implementation, save project data
    console.log('Saving project...')
  } catch (error) {
    console.error('Failed to save project:', error)
  }
}

const exportProject = async () => {
  try {
    // In real implementation, export project
    console.log('Exporting project...')
  } catch (error) {
    console.error('Failed to export project:', error)
  }
}

const zoomIn = () => {
  // Implement zoom in
  console.log('Zooming in...')
}

const zoomOut = () => {
  // Implement zoom out
  console.log('Zooming out...')
}

const resetView = () => {
  // Implement reset view
  console.log('Resetting view...')
}

const applyParameters = async () => {
  try {
    // In real implementation, apply parameters to simulation
    console.log('Applying parameters...')
  } catch (error) {
    console.error('Failed to apply parameters:', error)
  }
}

const runSimulation = async () => {
  try {
    // Simulate running engineering simulation
    simulationResults.value = [
      { id: 'thrust', name: 'Thrust Generated', value: '3924 N' },
      { id: 'stability', name: 'Stability Margin', value: '85%' },
      { id: 'efficiency', name: 'Power Efficiency', value: '78%' },
      { id: 'weight', name: 'Total Weight', value: '40 kg' }
    ]
  } catch (error) {
    console.error('Failed to run simulation:', error)
  }
}

const createTask = () => {
  // Implement create new task
  console.log('Creating new task...')
}

const viewTask = (task) => {
  // Implement view task details
  console.log('Viewing task:', task.id)
}

onMounted(() => {
  // Select first tool by default
  if (engineeringTools.length > 0) {
    selectedTool.value = engineeringTools[0]
  }
})
</script>