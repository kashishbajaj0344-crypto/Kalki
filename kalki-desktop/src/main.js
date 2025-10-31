import { createApp } from 'vue'
import App from './App.vue'
import './style.css'

// Create and mount the app
const app = createApp(App)

// Global error handler
app.config.errorHandler = (err, instance, info) => {
  console.error('Global error:', err)
  console.error('Component instance:', instance)
  console.error('Error info:', info)
}

// Mount the app
app.mount('#app')