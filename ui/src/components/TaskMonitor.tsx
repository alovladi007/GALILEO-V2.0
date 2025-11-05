'use client'

/**
 * Task Monitor Component
 *
 * Monitor and manage Celery background tasks.
 * Displays active, scheduled, and completed async operations.
 */

import React, { useState, useEffect } from 'react'
import { api } from '../lib/api-client-full'
import {
  Zap, Clock, CheckCircle, XCircle, Loader2,
  Server, Activity, Pause, Play, RotateCcw
} from 'lucide-react'

interface Task {
  task_id: string
  task_name: string
  status: string
  ready: boolean
  successful?: boolean
  failed?: boolean
  worker?: string
  result?: any
  error?: string
  progress?: number
  current?: number
  total?: number
}

interface WorkerStats {
  name: string
  pool: string
  max_concurrency: number
  active_tasks: number
  registered_tasks: number
  uptime: number
}

export function TaskMonitor() {
  const [activeTasks, setActiveTasks] = useState<Task[]>([])
  const [scheduledTasks, setScheduledTasks] = useState<Task[]>([])
  const [workerStats, setWorkerStats] = useState<WorkerStats[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedTask, setSelectedTask] = useState<string | null>(null)

  useEffect(() => {
    loadData()
    // Poll every 3 seconds
    const interval = setInterval(loadData, 3000)
    return () => clearInterval(interval)
  }, [])

  const loadData = async () => {
    try {
      const [activeRes, scheduledRes, workersRes] = await Promise.all([
        api.task.listActiveTasks(),
        api.task.listScheduledTasks(),
        api.task.getWorkerStats()
      ])

      setActiveTasks(activeRes.data.active_tasks || [])
      setScheduledTasks(scheduledRes.data.scheduled_tasks || [])
      setWorkerStats(workersRes.data.workers || [])
      setLoading(false)
    } catch (err: any) {
      console.error('Failed to load task data:', err)
      setLoading(false)
    }
  }

  const submitTestTask = async () => {
    try {
      await api.task.submitTask({
        task_name: 'simulation.propagate_formation',
        parameters: {
          n_satellites: 2,
          duration_days: 1,
          time_step: 60
        },
        priority: 5
      })
      loadData()
    } catch (err: any) {
      console.error('Failed to submit task:', err)
    }
  }

  const cancelTask = async (taskId: string) => {
    try {
      await api.task.cancelTask(taskId, false)
      loadData()
    } catch (err: any) {
      console.error('Failed to cancel task:', err)
    }
  }

  const retryTask = async (taskId: string) => {
    try {
      await api.task.retryTask(taskId)
      loadData()
    } catch (err: any) {
      console.error('Failed to retry task:', err)
    }
  }

  const getTaskIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'pending':
        return <Clock className="w-5 h-5 text-gray-400" />
      case 'started':
      case 'progress':
        return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'failure':
        return <XCircle className="w-5 h-5 text-red-500" />
      default:
        return <Activity className="w-5 h-5 text-gray-400" />
    }
  }

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Task Monitor
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Celery background task queue management
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={loadData}
            className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
          >
            Refresh
          </button>
          <button
            onClick={submitTestTask}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
          >
            <Play className="w-4 h-4" />
            Test Task
          </button>
        </div>
      </div>

      {/* Worker Stats */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {workerStats.map((worker) => (
          <div
            key={worker.name}
            className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4"
          >
            <div className="flex items-center gap-3 mb-3">
              <Server className="w-5 h-5 text-blue-500" />
              <div>
                <div className="font-medium text-gray-900 dark:text-white">
                  {worker.name.split('@')[1] || 'Worker'}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {worker.pool} pool
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <div className="text-gray-500 dark:text-gray-400">Active</div>
                <div className="text-xl font-bold text-gray-900 dark:text-white">
                  {worker.active_tasks}
                </div>
              </div>
              <div>
                <div className="text-gray-500 dark:text-gray-400">Capacity</div>
                <div className="text-xl font-bold text-gray-900 dark:text-white">
                  {worker.max_concurrency}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Task Lists */}
      <div className="grid grid-cols-2 gap-6">
        {/* Active Tasks */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-gray-900 dark:text-white">
                Active Tasks ({activeTasks.length})
              </h3>
              <Activity className="w-5 h-5 text-blue-500" />
            </div>
          </div>
          <div className="divide-y divide-gray-200 dark:divide-gray-700 max-h-96 overflow-y-auto">
            {activeTasks.length === 0 ? (
              <div className="p-8 text-center text-gray-500 dark:text-gray-400">
                No active tasks
              </div>
            ) : (
              activeTasks.map((task) => (
                <div
                  key={task.task_id}
                  className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700/50"
                >
                  <div className="flex items-start gap-3">
                    <Loader2 className="w-5 h-5 text-blue-500 animate-spin mt-0.5" />
                    <div className="flex-1">
                      <div className="font-medium text-gray-900 dark:text-white">
                        {task.task_name}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        ID: {task.task_id.slice(0, 8)}...
                      </div>
                      {task.worker && (
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          Worker: {task.worker.split('@')[1]}
                        </div>
                      )}
                      {task.progress !== undefined && (
                        <div className="mt-2">
                          <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
                            <span>Progress</span>
                            <span>{task.progress}%</span>
                          </div>
                          <div className="bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full transition-all"
                              style={{ width: `${task.progress}%` }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                    <button
                      onClick={() => cancelTask(task.task_id)}
                      className="px-2 py-1 bg-red-600 text-white rounded text-xs hover:bg-red-700 transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Scheduled Tasks */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-gray-900 dark:text-white">
                Scheduled Tasks ({scheduledTasks.length})
              </h3>
              <Clock className="w-5 h-5 text-orange-500" />
            </div>
          </div>
          <div className="divide-y divide-gray-200 dark:divide-gray-700 max-h-96 overflow-y-auto">
            {scheduledTasks.length === 0 ? (
              <div className="p-8 text-center text-gray-500 dark:text-gray-400">
                No scheduled tasks
              </div>
            ) : (
              scheduledTasks.map((task) => (
                <div
                  key={task.task_id}
                  className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700/50"
                >
                  <div className="flex items-start gap-3">
                    <Clock className="w-5 h-5 text-orange-500 mt-0.5" />
                    <div className="flex-1">
                      <div className="font-medium text-gray-900 dark:text-white">
                        {task.task_name}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        ID: {task.task_id.slice(0, 8)}...
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Task Submission Panel */}
      <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
        <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
          Quick Submit
        </h3>
        <div className="grid grid-cols-4 gap-3">
          <button className="px-3 py-2 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors text-sm">
            Simulation
          </button>
          <button className="px-3 py-2 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors text-sm">
            Inversion
          </button>
          <button className="px-3 py-2 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 rounded hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors text-sm">
            ML Training
          </button>
          <button className="px-3 py-2 bg-orange-50 dark:bg-orange-900/20 text-orange-700 dark:text-orange-300 rounded hover:bg-orange-100 dark:hover:bg-orange-900/30 transition-colors text-sm">
            Trade Study
          </button>
        </div>
      </div>
    </div>
  )
}

export default TaskMonitor
