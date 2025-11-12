
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { kalkiAPI } from '../lib/api'

export default function RoadmapPage() {
  const [projectType, setProjectType] = useState<'adu' | 'remodel' | 'new_construction'>('adu')
  const [propertyData, setPropertyData] = useState('')
  const [userPreferences, setUserPreferences] = useState('')
  const [result, setResult] = useState<any>(null)

  const mutation = useMutation({
    mutationFn: kalkiAPI.generateRoadmap,
    onSuccess: (data) => setResult(data),
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    let propertyObj = {}
    let preferencesObj = {}
    try { propertyObj = propertyData ? JSON.parse(propertyData) : {} } catch {}
    try { preferencesObj = userPreferences ? JSON.parse(userPreferences) : {} } catch {}
    mutation.mutate({
      project_type: projectType,
      property_data: propertyObj,
      user_preferences: preferencesObj,
    })
  }

  return (
    <div className="p-6 max-w-xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Roadmap Generator</h1>
      <form onSubmit={handleSubmit} className="space-y-4 bg-white p-4 rounded shadow">
        <div>
          <label className="block font-medium mb-1">Project Type</label>
          <select value={projectType} onChange={e => setProjectType(e.target.value as any)} className="w-full border rounded px-3 py-2">
            <option value="adu">ADU</option>
            <option value="remodel">Remodel</option>
            <option value="new_construction">New Construction</option>
          </select>
        </div>
        <div>
          <label className="block font-medium mb-1">Property Data (JSON)</label>
          <textarea value={propertyData} onChange={e => setPropertyData(e.target.value)} className="w-full border rounded px-3 py-2" rows={3} placeholder='{"sqft": 2000, "location": "CA"}' />
        </div>
        <div>
          <label className="block font-medium mb-1">User Preferences (JSON, optional)</label>
          <textarea value={userPreferences} onChange={e => setUserPreferences(e.target.value)} className="w-full border rounded px-3 py-2" rows={2} placeholder='{"budget": 500000}' />
        </div>
        <button type="submit" className="bg-primary-600 text-white px-6 py-2 rounded font-medium" disabled={mutation.isPending}>
          {mutation.isPending ? 'Generating...' : 'Generate Roadmap'}
        </button>
      </form>

      {result && (
        <div className="mt-6 bg-gray-50 p-4 rounded shadow">
          <h2 className="text-lg font-semibold mb-2">Roadmap Result</h2>
          <div className="mb-2"><b>Total Weeks:</b> {result.total_weeks}</div>
          <div className="mb-2"><b>Total Cost:</b> ${result.total_cost}</div>
          <div className="mb-2"><b>Complexity Score:</b> {result.complexity_score}</div>
          <div className="mb-2"><b>Timeline:</b> <pre className="inline">{JSON.stringify(result.timeline, null, 2)}</pre></div>
          <div className="mb-2"><b>Steps:</b> <pre className="inline">{JSON.stringify(result.steps, null, 2)}</pre></div>
        </div>
      )}
    </div>
  )
}
