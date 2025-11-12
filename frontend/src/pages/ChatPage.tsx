import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { kalkiAPI } from '../lib/api'
import { Send, Loader2, Brain } from 'lucide-react'

export default function ChatPage() {
  const [messages, setMessages] = useState<any[]>([])
  const [input, setInput] = useState('')

  const chatMutation = useMutation({
    mutationFn: kalkiAPI.chat,
    onSuccess: (data) => {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response,
        confidence: data.confidence,
        enhancements: data.enhancements_used,
        reasoning: data.reasoning,
        next_steps: data.next_steps,
      }])
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    setMessages(prev => [...prev, { role: 'user', content: input }])
    chatMutation.mutate({ user_input: input })
    setInput('')
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="bg-white border-b px-6 py-4">
        <h1 className="text-2xl font-bold text-gray-900">Construction Copilot Chat</h1>
        <p className="text-sm text-gray-600">AI-powered construction guidance with 10 intelligence enhancements</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center py-12">
            <Brain size={48} className="mx-auto text-gray-400 mb-4" />
            <h2 className="text-xl font-semibold text-gray-700 mb-2">Welcome to KALKI Construction Copilot</h2>
            <p className="text-gray-600">Ask me anything about your construction project!</p>
            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl mx-auto">
              <button
                onClick={() => setInput("I want to build an ADU in my backyard")}
                className="p-3 text-left border border-gray-200 rounded-lg hover:border-primary-300 hover:bg-primary-50 transition"
              >
                <div className="font-medium text-gray-900">Build an ADU</div>
                <div className="text-sm text-gray-600">Backyard accessory dwelling unit</div>
              </button>
              <button
                onClick={() => setInput("What permits do I need for a kitchen remodel?")}
                className="p-3 text-left border border-gray-200 rounded-lg hover:border-primary-300 hover:bg-primary-50 transition"
              >
                <div className="font-medium text-gray-900">Kitchen Remodel</div>
                <div className="text-sm text-gray-600">Permits and requirements</div>
              </button>
              <button
                onClick={() => setInput("How much does it cost to build a new house?")}
                className="p-3 text-left border border-gray-200 rounded-lg hover:border-primary-300 hover:bg-primary-50 transition"
              >
                <div className="font-medium text-gray-900">Cost Estimation</div>
                <div className="text-sm text-gray-600">Budget planning</div>
              </button>
              <button
                onClick={() => setInput("Generate a construction timeline for me")}
                className="p-3 text-left border border-gray-200 rounded-lg hover:border-primary-300 hover:bg-primary-50 transition"
              >
                <div className="font-medium text-gray-900">Timeline Planning</div>
                <div className="text-sm text-gray-600">Project scheduling</div>
              </button>
            </div>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-3xl rounded-lg px-4 py-3 ${
              msg.role === 'user'
                ? 'bg-primary-600 text-white'
                : 'bg-white border border-gray-200'
            }`}>
              <p className="whitespace-pre-wrap">{msg.content}</p>
              
              {msg.confidence && (
                <div className="mt-2 pt-2 border-t border-gray-200 text-xs text-gray-600">
                  <div className="flex justify-between items-center">
                    <span>Confidence: {(msg.confidence * 100).toFixed(0)}%</span>
                    {msg.enhancements && msg.enhancements.length > 0 && (
                      <span className="text-primary-600">✨ {msg.enhancements.join(', ')}</span>
                    )}
                  </div>
                </div>
              )}

              {msg.next_steps && msg.next_steps.length > 0 && (
                <div className="mt-3 p-2 bg-blue-50 rounded border-l-4 border-blue-400">
                  <div className="font-medium text-sm text-gray-900 mb-1">Next Steps:</div>
                  <ul className="text-sm text-gray-700 space-y-1">
                    {msg.next_steps.map((step: string, i: number) => (
                      <li key={i}>• {step}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}

        {chatMutation.isPending && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 rounded-lg px-4 py-3 flex items-center gap-2">
              <Loader2 className="animate-spin" size={16} />
              <span className="text-gray-600">KALKI is thinking...</span>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t bg-white px-6 py-4">
        <form onSubmit={handleSubmit} className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask me anything about construction..."
            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            disabled={chatMutation.isPending}
          />
          <button
            type="submit"
            disabled={chatMutation.isPending || !input.trim()}
            className="px-6 py-3 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <Send size={18} />
            Send
          </button>
        </form>
      </div>
    </div>
  )
}
