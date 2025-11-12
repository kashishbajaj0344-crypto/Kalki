
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { kalkiAPI } from '../lib/api'

export default function PropertyPage() {
  const [address, setAddress] = useState('')
  const [propertyType, setPropertyType] = useState<'residential' | 'commercial'>('residential')
  const [projectIntent, setProjectIntent] = useState('')
  const [result, setResult] = useState<any>(null)

  const mutation = useMutation({
    mutationFn: kalkiAPI.analyzeProperty,
    onSuccess: (data) => setResult(data),
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    mutation.mutate({
      address,
      property_type: propertyType,
      project_intent: projectIntent,
    })
  }

  return (
    <div className="p-6 max-w-xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Property Intelligence</h1>
      <form onSubmit={handleSubmit} className="space-y-4 bg-white p-4 rounded shadow">
        <div>
          <label className="block font-medium mb-1">Address</label>
          <input type="text" value={address} onChange={e => setAddress(e.target.value)} className="w-full border rounded px-3 py-2" required />
        </div>
        <div>
          <label className="block font-medium mb-1">Property Type</label>
          <select value={propertyType} onChange={e => setPropertyType(e.target.value as any)} className="w-full border rounded px-3 py-2">
            <option value="residential">Residential</option>
            <option value="commercial">Commercial</option>
          </select>
        </div>
        <div>
          <label className="block font-medium mb-1">Project Intent</label>
          <input type="text" value={projectIntent} onChange={e => setProjectIntent(e.target.value)} className="w-full border rounded px-3 py-2" required />
        </div>
        <button type="submit" className="bg-primary-600 text-white px-6 py-2 rounded font-medium" disabled={mutation.isPending}>
          {mutation.isPending ? 'Analyzing...' : 'Analyze'}
        </button>
      </form>

      {result && (
        <div className="mt-6 bg-gray-50 p-4 rounded shadow">
          <h2 className="text-lg font-semibold mb-2">Analysis Result</h2>
          <div className="mb-2"><b>Complexity Score:</b> {result.complexity_score}</div>
          <div className="mb-2"><b>Zoning Info:</b> <pre className="inline">{JSON.stringify(result.zoning_info, null, 2)}</pre></div>
          {result.setbacks && <div className="mb-2"><b>Setbacks:</b> <pre className="inline">{JSON.stringify(result.setbacks, null, 2)}</pre></div>}
          <div className="mb-2"><b>Permit Requirements:</b> {result.permit_requirements?.join(', ')}</div>
          <div className="mb-2"><b>Estimated Timeline:</b> {result.estimated_timeline}</div>
          <div className="mb-2"><b>Risks:</b> {result.risks?.join(', ')}</div>
        </div>
      )}
    </div>
  )
}
