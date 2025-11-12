
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { kalkiAPI } from '../lib/api'

export default function ConsensusPage() {
  const [decision, setDecision] = useState('')
  const [context, setContext] = useState('')
  const [requireUnanimous, setRequireUnanimous] = useState(false)
  const [result, setResult] = useState<any>(null)

  const mutation = useMutation({
    mutationFn: kalkiAPI.getConsensus,
    onSuccess: (data) => setResult(data),
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    let contextObj = {}
    try { contextObj = context ? JSON.parse(context) : {} } catch {}
    mutation.mutate({
      decision,
      context: contextObj,
      require_unanimous: requireUnanimous,
    })
  }

  return (
    <div className="p-6 max-w-xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Multi-Agent Consensus</h1>
      <form onSubmit={handleSubmit} className="space-y-4 bg-white p-4 rounded shadow">
        <div>
          <label className="block font-medium mb-1">Decision to Validate</label>
          <input type="text" value={decision} onChange={e => setDecision(e.target.value)} className="w-full border rounded px-3 py-2" required />
        </div>
        <div>
          <label className="block font-medium mb-1">Context (JSON, optional)</label>
          <textarea value={context} onChange={e => setContext(e.target.value)} className="w-full border rounded px-3 py-2" rows={2} placeholder='{"project": "ADU", "budget": 500000}' />
        </div>
        <div className="flex items-center gap-2">
          <input type="checkbox" checked={requireUnanimous} onChange={e => setRequireUnanimous(e.target.checked)} id="unanimous" />
          <label htmlFor="unanimous">Require Unanimous Agreement</label>
        </div>
        <button type="submit" className="bg-primary-600 text-white px-6 py-2 rounded font-medium" disabled={mutation.isPending}>
          {mutation.isPending ? 'Validating...' : 'Get Consensus'}
        </button>
      </form>

      {result && (
        <div className="mt-6 bg-gray-50 p-4 rounded shadow">
          <h2 className="text-lg font-semibold mb-2">Consensus Result</h2>
          <div className="mb-2"><b>Agreement:</b> {result.agreement * 100}%</div>
          <div className="mb-2"><b>Recommendation:</b> {result.recommendation}</div>
          <div className="mb-2"><b>Conflicts:</b> {result.conflicts?.join(', ')}</div>
          <div className="mb-2"><b>Voting Breakdown:</b> <pre className="inline">{JSON.stringify(result.voting_breakdown, null, 2)}</pre></div>
          <div className="mb-2"><b>Individual Analyses:</b> <pre className="inline">{JSON.stringify(result.individual_analyses, null, 2)}</pre></div>
        </div>
      )}
    </div>
  )
}
