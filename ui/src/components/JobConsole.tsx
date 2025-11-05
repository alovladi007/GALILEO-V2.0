'use client'

import { useState, useEffect } from 'react'
import { useJobs } from '@/hooks/useJobs'
import { 
  PlayIcon, 
  StopIcon, 
  CheckCircleIcon, 
  XCircleIcon,
  ArrowPathIcon,
  ClockIcon
} from '@heroicons/react/24/outline'
import { format } from 'date-fns'
import toast from 'react-hot-toast'

export function JobConsole() {
  const [selectedTab, setSelectedTab] = useState<'active' | 'history'>('active')
  const { jobs, isLoading, createJob, cancelJob, refetch } = useJobs()
  const [isCreatingJob, setIsCreatingJob] = useState(false)

  const activeJobs = jobs?.filter(job => 
    ['pending', 'running'].includes(job.status)
  ) || []
  
  const completedJobs = jobs?.filter(job => 
    ['completed', 'failed', 'cancelled'].includes(job.status)
  ) || []

  const handleCreateTestJob = async () => {
    setIsCreatingJob(true)
    try {
      await createJob({
        type: 'process',
        config: {
          algorithm: 'variational',
          degree_max: 60,
          satellites: ['GRACE-A', 'GRACE-B']
        }
      })
      toast.success('Processing job created')
      refetch()
    } catch (error) {
      toast.error('Failed to create job')
    } finally {
      setIsCreatingJob(false)
    }
  }

  const handleCancelJob = async (jobId: string) => {
    try {
      await cancelJob(jobId)
      toast.success('Job cancelled')
      refetch()
    } catch (error) {
      toast.error('Failed to cancel job')
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <ArrowPathIcon className="h-5 w-5 text-blue-400 animate-spin" />
      case 'pending':
        return <ClockIcon className="h-5 w-5 text-yellow-400" />
      case 'completed':
        return <CheckCircleIcon className="h-5 w-5 text-green-400" />
      case 'failed':
        return <XCircleIcon className="h-5 w-5 text-red-400" />
      default:
        return <XCircleIcon className="h-5 w-5 text-gray-400" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'text-blue-400'
      case 'pending':
        return 'text-yellow-400'
      case 'completed':
        return 'text-green-400'
      case 'failed':
        return 'text-red-400'
      default:
        return 'text-gray-400'
    }
  }

  // Auto-refresh active jobs
  useEffect(() => {
    if (activeJobs.length > 0) {
      const interval = setInterval(() => {
        refetch()
      }, 5000) // Refresh every 5 seconds

      return () => clearInterval(interval)
    }
  }, [activeJobs.length, refetch])

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Processing Jobs</h3>
        <button
          onClick={() => refetch()}
          className="p-2 text-gray-400 hover:text-white transition-colors"
          title="Refresh"
        >
          <ArrowPathIcon className="h-4 w-4" />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex space-x-1 mb-4">
        <button
          onClick={() => setSelectedTab('active')}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            selectedTab === 'active' 
              ? 'bg-primary-600 text-white' 
              : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
          }`}
        >
          Active ({activeJobs.length})
        </button>
        <button
          onClick={() => setSelectedTab('history')}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            selectedTab === 'history' 
              ? 'bg-primary-600 text-white' 
              : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
          }`}
        >
          History ({completedJobs.length})
        </button>
      </div>

      {/* Job List */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {isLoading ? (
          <div className="text-center py-8">
            <div className="loading-spinner mx-auto mb-2" />
            <p className="text-sm text-gray-400">Loading jobs...</p>
          </div>
        ) : selectedTab === 'active' ? (
          activeJobs.length > 0 ? (
            activeJobs.map(job => (
              <div key={job.id} className="bg-slate-700 rounded-lg p-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    {getStatusIcon(job.status)}
                    <div>
                      <p className="font-medium">{job.job_type.toUpperCase()}</p>
                      <p className="text-xs text-gray-400">
                        ID: {job.id.slice(0, 8)}...
                      </p>
                      <p className="text-xs text-gray-400">
                        Started: {format(new Date(job.created_at), 'HH:mm:ss')}
                      </p>
                      {job.status === 'running' && (
                        <div className="mt-2">
                          <div className="w-32 h-1 bg-slate-600 rounded-full overflow-hidden">
                            <div className="h-full bg-primary-500 animate-pulse" style={{ width: '60%' }} />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                  {job.status === 'running' && (
                    <button
                      onClick={() => handleCancelJob(job.id)}
                      className="p-1 text-red-400 hover:text-red-300 transition-colors"
                      title="Cancel job"
                    >
                      <StopIcon className="h-4 w-4" />
                    </button>
                  )}
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-8">
              <p className="text-gray-400 mb-4">No active jobs</p>
              <button
                onClick={handleCreateTestJob}
                disabled={isCreatingJob}
                className="button-primary flex items-center space-x-2 mx-auto"
              >
                <PlayIcon className="h-4 w-4" />
                <span>{isCreatingJob ? 'Creating...' : 'Start Test Job'}</span>
              </button>
            </div>
          )
        ) : (
          completedJobs.length > 0 ? (
            completedJobs.slice(0, 10).map(job => (
              <div key={job.id} className="bg-slate-700 rounded-lg p-3">
                <div className="flex items-start space-x-3">
                  {getStatusIcon(job.status)}
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <p className="font-medium">{job.job_type.toUpperCase()}</p>
                      <span className={`text-xs ${getStatusColor(job.status)}`}>
                        {job.status.toUpperCase()}
                      </span>
                    </div>
                    <p className="text-xs text-gray-400">
                      ID: {job.id.slice(0, 8)}...
                    </p>
                    <p className="text-xs text-gray-400">
                      Completed: {format(new Date(job.completed_at || job.updated_at), 'MMM dd, HH:mm')}
                    </p>
                    {job.error_message && (
                      <p className="text-xs text-red-400 mt-1">
                        Error: {job.error_message}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-8 text-gray-400">
              No completed jobs
            </div>
          )
        )}
      </div>

      {/* Quick Actions */}
      <div className="mt-4 pt-4 border-t border-slate-600">
        <div className="grid grid-cols-2 gap-2">
          <button
            className="button-secondary text-sm py-2"
            onClick={() => {
              toast.success('Opening processing panel...')
            }}
          >
            New Process
          </button>
          <button
            className="button-secondary text-sm py-2"
            onClick={() => {
              toast.success('Opening job history...')
            }}
          >
            View All
          </button>
        </div>
      </div>
    </div>
  )
}
