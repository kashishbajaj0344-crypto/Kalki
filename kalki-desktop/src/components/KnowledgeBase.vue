<template>
  <div class="h-full grid grid-cols-12 gap-6">
    <!-- Search & Filters -->
    <div class="col-span-3 space-y-6">
      <!-- Search -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Knowledge Search</h3>
        <div class="space-y-4">
          <div>
            <input
              v-model="searchQuery"
              @keyup.enter="performSearch"
              class="w-full bg-black/50 border border-gray-600 rounded-lg px-3 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
              placeholder="Search knowledge base..."
            />
          </div>
          <button
            @click="performSearch"
            class="w-full py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white font-medium transition-all duration-200"
          >
            Search
          </button>
        </div>
      </div>

      <!-- Filters -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Filters</h3>
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">Category</label>
            <select
              v-model="selectedCategory"
              class="w-full bg-black/50 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-purple-500"
            >
              <option value="all">All Categories</option>
              <option value="science">Science</option>
              <option value="technology">Technology</option>
              <option value="engineering">Engineering</option>
              <option value="mathematics">Mathematics</option>
              <option value="philosophy">Philosophy</option>
            </select>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">Source Type</label>
            <select
              v-model="selectedSourceType"
              class="w-full bg-black/50 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-purple-500"
            >
              <option value="all">All Sources</option>
              <option value="pdf">PDF Documents</option>
              <option value="web">Web Content</option>
              <option value="database">Database</option>
              <option value="generated">AI Generated</option>
            </select>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">Date Range</label>
            <select
              v-model="selectedDateRange"
              class="w-full bg-black/50 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-purple-500"
            >
              <option value="all">All Time</option>
              <option value="week">Last Week</option>
              <option value="month">Last Month</option>
              <option value="year">Last Year</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Knowledge Stats -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Knowledge Stats</h3>
        <div class="space-y-3">
          <div class="flex justify-between">
            <span class="text-sm text-gray-400">Total Documents</span>
            <span class="text-sm text-white font-medium">{{ knowledgeStats.totalDocuments }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-sm text-gray-400">Indexed Pages</span>
            <span class="text-sm text-white font-medium">{{ knowledgeStats.indexedPages }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-sm text-gray-400">Vector Embeddings</span>
            <span class="text-sm text-white font-medium">{{ knowledgeStats.vectorEmbeddings }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-sm text-gray-400">Last Updated</span>
            <span class="text-sm text-white font-medium">{{ knowledgeStats.lastUpdated }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Results & Content -->
    <div class="col-span-9 space-y-6">
      <!-- Search Results -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-white">Search Results</h3>
          <div class="text-sm text-gray-400">{{ searchResults.length }} results found</div>
        </div>

        <div v-if="isSearching" class="text-center py-8">
          <div class="animate-spin w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <div class="text-gray-400">Searching knowledge base...</div>
        </div>

        <div v-else-if="searchResults.length === 0 && searchQuery" class="text-center py-8">
          <BookOpen class="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <div class="text-gray-400">No results found for "{{ searchQuery }}"</div>
        </div>

        <div v-else class="space-y-3 max-h-96 overflow-y-auto">
          <div
            v-for="result in searchResults"
            :key="result.id"
            @click="selectResult(result)"
            class="p-4 rounded-lg bg-white/5 hover:bg-white/10 cursor-pointer transition-all duration-200"
            :class="selectedResult?.id === result.id ? 'ring-2 ring-purple-500' : ''"
          >
            <div class="flex items-start justify-between mb-2">
              <div class="flex-1">
                <h4 class="text-white font-medium mb-1">{{ result.title }}</h4>
                <p class="text-sm text-gray-400 mb-2">{{ result.content }}</p>
                <div class="flex items-center space-x-4 text-xs text-gray-500">
                  <span>{{ result.source }}</span>
                  <span>{{ result.category }}</span>
                  <span>{{ result.date }}</span>
                  <span class="bg-purple-600/20 text-purple-400 px-2 py-1 rounded">Relevance: {{ result.relevance }}%</span>
                </div>
              </div>
              <div class="ml-4">
                <button @click.stop="openDocument(result)" class="text-purple-400 hover:text-purple-300 text-sm">
                  Open
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Document Viewer -->
      <div v-if="selectedResult" class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10 flex-1">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-white">{{ selectedResult.title }}</h3>
          <div class="flex items-center space-x-2">
            <button @click="exportDocument" class="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white text-sm transition-all duration-200">
              Export
            </button>
            <button @click="shareDocument" class="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-white text-sm transition-all duration-200">
              Share
            </button>
          </div>
        </div>

        <div class="bg-black/50 rounded-lg p-4 h-96 overflow-y-auto">
          <div class="prose prose-invert max-w-none">
            <div class="text-white whitespace-pre-wrap">{{ selectedResult.fullContent }}</div>
          </div>
        </div>

        <!-- Document Metadata -->
        <div class="mt-4 grid grid-cols-3 gap-4 text-sm">
          <div>
            <span class="text-gray-400">Source:</span>
            <span class="text-white ml-2">{{ selectedResult.source }}</span>
          </div>
          <div>
            <span class="text-gray-400">Category:</span>
            <span class="text-white ml-2">{{ selectedResult.category }}</span>
          </div>
          <div>
            <span class="text-gray-400">Last Modified:</span>
            <span class="text-white ml-2">{{ selectedResult.date }}</span>
          </div>
        </div>
      </div>

      <!-- Knowledge Graph -->
      <div class="bg-white/5 backdrop-blur-md rounded-xl p-6 border border-white/10">
        <h3 class="text-lg font-semibold text-white mb-4">Knowledge Graph</h3>
        <div class="bg-gradient-to-br from-blue-900/20 to-purple-900/20 rounded-lg h-48 flex items-center justify-center">
          <div class="text-center">
            <Network class="w-12 h-12 mx-auto mb-2 text-blue-400" />
            <div class="text-white text-sm">Interactive Knowledge Graph</div>
            <div class="text-gray-400 text-xs">Click nodes to explore connections</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import { BookOpen, Network } from 'lucide-vue-next'

// Reactive data
const searchQuery = ref('')
const selectedCategory = ref('all')
const selectedSourceType = ref('all')
const selectedDateRange = ref('all')
const isSearching = ref(false)
const selectedResult = ref(null)

const knowledgeStats = ref({
  totalDocuments: 1247,
  indexedPages: 15632,
  vectorEmbeddings: 89234,
  lastUpdated: '2 hours ago'
})

const searchResults = ref([
  {
    id: 1,
    title: 'Advanced Robotics: Exoskeleton Design Principles',
    content: 'Comprehensive analysis of powered exoskeleton systems, including mechanical design, control systems, and human-machine interfaces...',
    source: 'engineering.pdf',
    category: 'Engineering',
    date: '2024-01-15',
    relevance: 95,
    fullContent: `Advanced Robotics: Exoskeleton Design Principles

This document provides a comprehensive overview of powered exoskeleton technology, covering fundamental design principles, mechanical considerations, and control system architectures.

1. Mechanical Design
- Degrees of freedom analysis
- Joint actuation systems
- Power transmission methods
- Structural materials selection

2. Control Systems
- PID control implementation
- Sensor fusion techniques
- Human intent recognition
- Safety and stability control

3. Human Factors
- Ergonomic considerations
- Power requirements
- Weight distribution
- User comfort optimization

The design of powered exoskeletons requires careful consideration of multiple engineering disciplines including mechanics, electronics, and human physiology.`
  },
  {
    id: 2,
    title: 'Consciousness in Artificial Intelligence',
    content: 'Exploring the emergence of consciousness in advanced AI systems through recursive self-observation and neural correlates...',
    source: 'ai_consciousness.pdf',
    category: 'AI Research',
    date: '2024-01-10',
    relevance: 88,
    fullContent: `Consciousness in Artificial Intelligence

The emergence of consciousness in artificial intelligence represents one of the most profound challenges in modern computer science. This paper explores the theoretical foundations and practical implementations of conscious AI systems.

Key Concepts:
- Neural correlates of consciousness
- Self-awareness mechanisms
- Recursive self-observation
- Emotional intelligence in AI
- Ethical considerations

Implementation Approaches:
1. Multi-agent consciousness emergence
2. Neural network self-modeling
3. Attention-based awareness systems
4. Memory consolidation for identity
5. Goal-directed behavior evolution

The path to conscious AI requires not just advanced algorithms, but a fundamental rethinking of how we approach machine intelligence.`
  },
  {
    id: 3,
    title: 'Lithium-Sulfur Battery Technology',
    content: 'Next-generation battery technology for high-energy-density applications, including electric vehicles and aerospace...',
    source: 'battery_tech.pdf',
    category: 'Materials Science',
    date: '2024-01-08',
    relevance: 82,
    fullContent: `Lithium-Sulfur Battery Technology

Lithium-sulfur batteries represent a promising next-generation energy storage technology with significantly higher energy density than current lithium-ion systems.

Advantages:
- Energy density: 500 Wh/kg (vs 250 Wh/kg for Li-ion)
- Cost reduction potential
- Abundant sulfur material
- Environmental benefits

Challenges:
- Cycle life limitations
- Self-discharge issues
- Electrolyte compatibility
- Manufacturing scalability

Recent advances in nanotechnology and materials science have made significant progress toward commercial viability of Li-S batteries.`
  }
])

// Methods
const performSearch = async () => {
  if (!searchQuery.value.trim()) return

  isSearching.value = true

  try {
    // In real implementation, call Tauri command for knowledge base search
    const results = await invoke('search_knowledge_base', {
      query: searchQuery.value,
      limit: 10
    })

    // For demo, filter existing results
    const filtered = searchResults.value.filter(result =>
      result.title.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
      result.content.toLowerCase().includes(searchQuery.value.toLowerCase())
    )

    searchResults.value = filtered
  } catch (error) {
    console.error('Search failed:', error)
    // Keep demo results
  } finally {
    isSearching.value = false
  }
}

const selectResult = (result) => {
  selectedResult.value = result
}

const openDocument = (result) => {
  // In real implementation, open document in external viewer
  console.log('Opening document:', result.title)
}

const exportDocument = () => {
  // In real implementation, export document
  console.log('Exporting document...')
}

const shareDocument = () => {
  // In real implementation, share document
  console.log('Sharing document...')
}
</script>