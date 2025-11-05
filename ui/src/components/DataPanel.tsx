'use client'

import { useState } from 'react'
import { 
  ChartBarIcon, 
  InformationCircleIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon
} from '@heroicons/react/24/outline'
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import { format } from 'date-fns'

interface DataPanelProps {
  selectedTime: Date
  satelliteData?: any[]
  gravityData?: any
  onRunCompare: (runId: string) => void
  selectedRun?: string | null
}

export function DataPanel({
  selectedTime,
  satelliteData = [],
  gravityData,
  onRunCompare,
  selectedRun
}: DataPanelProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'details' | 'comparison'>('overview')

  // Generate sample time series data
  const generateTimeSeriesData = () => {
    const data = []
    for (let i = 0; i < 24; i++) {
      data.push({
        time: format(new Date(selectedTime.getTime() - (23 - i) * 3600000), 'HH:mm'),
        gravity: Math.sin(i / 4) * 10 + Math.random() * 2,
        uncertainty: Math.abs(Math.cos(i / 4) * 2 + Math.random()),
        baseline: 220 + Math.sin(i / 3) * 5 + Math.random() * 2
      })
    }
    return data
  }

  const timeSeriesData = generateTimeSeriesData()

  // Calculate statistics
  const stats = {
    meanGravity: gravityData?.mean || -9.8,
    stdGravity: gravityData?.std || 0.015,
    maxAnomaly: gravityData?.maxAnomaly || 45.2,
    minAnomaly: gravityData?.minAnomaly || -38.7,
    coverage: satelliteData.length > 0 ? 87.3 : 0,
    dataQuality: 'Good'
  }

  const availableRuns = [
    { id: 'run-001', name: 'RL06.1 - January 2025', date: '2025-01-15' },
    { id: 'run-002', name: 'RL06.1 - December 2024', date: '2024-12-15' },
    { id: 'run-003', name: 'RL06.0 - November 2024', date: '2024-11-15' },
  ]

  return (
    <div className="card">
      {/* Tabs */}
      <div className="flex space-x-1 mb-4">
        <button
          onClick={() => setActiveTab('overview')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            activeTab === 'overview' 
              ? 'bg-primary-600 text-white' 
              : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
          }`}
        >
          Overview
        </button>
        <button
          onClick={() => setActiveTab('details')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            activeTab === 'details' 
              ? 'bg-primary-600 text-white' 
              : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
          }`}
        >
          Details
        </button>
        <button
          onClick={() => setActiveTab('comparison')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            activeTab === 'comparison' 
              ? 'bg-primary-600 text-white' 
              : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
          }`}
        >
          Comparison
        </button>
      </div>

      {/* Content */}
      {activeTab === 'overview' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold flex items-center">
            <ChartBarIcon className="h-5 w-5 mr-2 text-primary-400" />
            Gravity Field Statistics
          </h3>
          
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-700 rounded-lg p-3">
              <p className="text-xs text-gray-400 mb-1">Mean Gravity</p>
              <p className="text-lg font-semibold">{stats.meanGravity.toFixed(3)} mGal</p>
            </div>
            <div className="bg-slate-700 rounded-lg p-3">
              <p className="text-xs text-gray-400 mb-1">Std Deviation</p>
              <p className="text-lg font-semibold">{stats.stdGravity.toFixed(3)} mGal</p>
            </div>
            <div className="bg-slate-700 rounded-lg p-3">
              <p className="text-xs text-gray-400 mb-1">Max Anomaly</p>
              <p className="text-lg font-semibold flex items-center">
                <ArrowTrendingUpIcon className="h-4 w-4 mr-1 text-red-400" />
                {stats.maxAnomaly.toFixed(1)} mGal
              </p>
            </div>
            <div className="bg-slate-700 rounded-lg p-3">
              <p className="text-xs text-gray-400 mb-1">Min Anomaly</p>
              <p className="text-lg font-semibold flex items-center">
                <ArrowTrendingDownIcon className="h-4 w-4 mr-1 text-blue-400" />
                {stats.minAnomaly.toFixed(1)} mGal
              </p>
            </div>
          </div>

          <div>
            <h4 className="text-sm font-medium mb-2">Gravity Trend (24h)</h4>
            <ResponsiveContainer width="100%" height={150}>
              <AreaChart data={timeSeriesData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                <XAxis dataKey="time" stroke="#94a3b8" fontSize={10} />
                <YAxis stroke="#94a3b8" fontSize={10} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: 'none' }}
                  labelStyle={{ color: '#94a3b8' }}
                />
                <Area 
                  type="monotone" 
                  dataKey="gravity" 
                  stroke="#3b82f6" 
                  fill="#3b82f6" 
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="flex items-center justify-between p-3 bg-slate-700 rounded-lg">
            <div className="flex items-center">
              <InformationCircleIcon className="h-5 w-5 mr-2 text-green-400" />
              <span className="text-sm">Data Quality: {stats.dataQuality}</span>
            </div>
            <div className="text-sm text-gray-400">
              Coverage: {stats.coverage.toFixed(1)}%
            </div>
          </div>
        </div>
      )}

      {activeTab === 'details' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Satellite Details</h3>
          
          {satelliteData.length > 0 ? (
            <div className="space-y-3">
              {satelliteData.map((sat) => (
                <div key={sat.id} className="bg-slate-700 rounded-lg p-3">
                  <h4 className="font-medium mb-2">{sat.id}</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-gray-400">Position:</span>
                      <p>Lat: {sat.position.lat.toFixed(3)}°</p>
                      <p>Lon: {sat.position.lon.toFixed(3)}°</p>
                      <p>Alt: {sat.position.alt.toFixed(1)} km</p>
                    </div>
                    <div>
                      <span className="text-gray-400">Measurements:</span>
                      <p>Gravity: {(sat.gravity || 0).toFixed(6)} mGal</p>
                      <p>Quality: Good</p>
                      <p>SNR: 42.1 dB</p>
                    </div>
                  </div>
                </div>
              ))}
              
              <div>
                <h4 className="text-sm font-medium mb-2">Baseline Length</h4>
                <ResponsiveContainer width="100%" height={120}>
                  <LineChart data={timeSeriesData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                    <XAxis dataKey="time" stroke="#94a3b8" fontSize={10} />
                    <YAxis stroke="#94a3b8" fontSize={10} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1e293b', border: 'none' }}
                      labelStyle={{ color: '#94a3b8' }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="baseline" 
                      stroke="#10b981" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-400">
              No satellite data available for selected time
            </div>
          )}
        </div>
      )}

      {activeTab === 'comparison' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Run Comparison</h3>
          
          <div className="space-y-2">
            {availableRuns.map((run) => (
              <button
                key={run.id}
                onClick={() => onRunCompare(run.id)}
                className={`w-full text-left p-3 rounded-lg transition-colors ${
                  selectedRun === run.id
                    ? 'bg-primary-600 text-white'
                    : 'bg-slate-700 hover:bg-slate-600 text-gray-300'
                }`}
              >
                <div className="font-medium">{run.name}</div>
                <div className="text-sm opacity-75">Processed: {run.date}</div>
              </button>
            ))}
          </div>

          {selectedRun && (
            <div>
              <h4 className="text-sm font-medium mb-2">Difference Plot</h4>
              <ResponsiveContainer width="100%" height={150}>
                <AreaChart data={timeSeriesData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                  <XAxis dataKey="time" stroke="#94a3b8" fontSize={10} />
                  <YAxis stroke="#94a3b8" fontSize={10} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: 'none' }}
                    labelStyle={{ color: '#94a3b8' }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="uncertainty" 
                    stroke="#f59e0b" 
                    fill="#f59e0b" 
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
              
              <div className="grid grid-cols-2 gap-2 mt-3 text-sm">
                <div className="bg-slate-700 rounded p-2">
                  <p className="text-gray-400">Mean Diff</p>
                  <p className="font-medium">0.023 mGal</p>
                </div>
                <div className="bg-slate-700 rounded p-2">
                  <p className="text-gray-400">RMS Diff</p>
                  <p className="font-medium">0.041 mGal</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
