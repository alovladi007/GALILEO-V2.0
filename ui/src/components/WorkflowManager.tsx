'use client'

/**
 * Workflow Manager Component
 *
 * Submit, monitor, and manage end-to-end mission workflows.
 * Supports 5 workflow templates: full_mission, simulation_only,
 * inversion_pipeline, trade_study, compliance_audit
 */

import React, { useState, useEffect } from 'react'
import { api } from '../lib/api-client-full'
import {
  PlayCircle, Clock, CheckCircle, XCircle, AlertCircle,
  Loader2, Eye, Download, X
} from 'lucide-react'

interface Workflow {
  workflow_id: string
  workflow_type: string
  status: string
  current_step: number
  total_steps: number
  progress_percent: number
  created_at: string
  started_at?: string
  completed_at?: string
  error?: string
}

interface WorkflowTemplate {
  type: string
  name: string
  description: string
  steps: Array<{
    service: string
    method: string
    description: string
  }>
  estimated_duration: string
  outputs: string[]
}

export function WorkflowManager() {
  const [templates, setTemplates] = useState<WorkflowTemplate[]>([])
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showSubmitDialog, setShowSubmitDialog] = useState(false)

  // Load templates and workflows
  useEffect(() => {
    loadTemplates()
    loadWorkflows()
    // Poll for updates every 5 seconds
    const interval = setInterval(loadWorkflows, 5000)
    return () => clearInterval(interval)
  }, [])

  const loadTemplates = async () => {
    try {
      const response = await api.workflow.listTemplates()
      setTemplates(response.data.templates || [])
    } catch (err: any) {
      console.error('Failed to load templates:', err)
    }
  }

  const loadWorkflows = async () => {
    try {
      const response = await api.workflow.listWorkflows({ limit: 50 })
      setWorkflows(response.data.workflows || [])
    } catch (err: any) {
      console.error('Failed to load workflows:', err)
    }
  }

  const submitWorkflow = async (workflowType: string) => {
    setLoading(true)
    setError(null)

    try {
      const response = await api.workflow.submitWorkflow({
        workflow_type: workflowType,
        parameters: {}, // Template-specific params would go here
        user_id: 'demo-user',
        priority: 5
      })

      setShowSubmitDialog(false)
      loadWorkflows()
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to submit workflow')
    } finally {
      setLoading(false)
    }
  }

  const executeWorkflow = async (workflowId: string) => {
    try {
      await api.workflow.executeWorkflow(workflowId)
      loadWorkflows()
    } catch (err: any) {
      console.error('Failed to execute workflow:', err)
    }
  }

  const cancelWorkflow = async (workflowId: string) => {
    try {
      await api.workflow.cancelWorkflow(workflowId)
      loadWorkflows()
    } catch (err: any) {
      console.error('Failed to cancel workflow:', err)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <Clock className="w-5 h-5 text-gray-400" />
      case 'running':
        return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />
      case 'cancelled':
        return <AlertCircle className="w-5 h-5 text-yellow-500" />
      default:
        return <Clock className="w-5 h-5 text-gray-400" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
      case 'completed':
        return 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
      case 'failed':
        return 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
      case 'cancelled':
        return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300'
      default:
        return 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
    }
  }

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Workflow Manager
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Submit and monitor end-to-end mission workflows
          </p>
        </div>
        <button
          onClick={() => setShowSubmitDialog(true)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
        >
          <PlayCircle className="w-5 h-5" />
          New Workflow
        </button>
      </div>

      {/* Workflow Templates Grid */}
      {showSubmitDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-4xl w-full max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                Select Workflow Template
              </h3>
              <button
                onClick={() => setShowSubmitDialog(false)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {templates.map((template) => (
                <div
                  key={template.type}
                  className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:border-blue-500 transition-colors cursor-pointer"
                  onClick={() => {
                    setSelectedTemplate(template.type)
                    submitWorkflow(template.type)
                  }}
                >
                  <h4 className="font-semibold text-gray-900 dark:text-white">
                    {template.name}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    {template.description}
                  </p>
                  <div className="mt-3 text-xs text-gray-500 dark:text-gray-400">
                    <div className="flex items-center gap-2">
                      <Clock className="w-3 h-3" />
                      <span>{template.estimated_duration}</span>
                    </div>
                    <div className="mt-2">
                      {template.steps.length} steps
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {error && (
              <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                <p className="text-red-700 dark:text-red-300 text-sm">{error}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Active Workflows */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <h3 className="font-semibold text-gray-900 dark:text-white">
            Active Workflows ({workflows.length})
          </h3>
        </div>

        <div className="divide-y divide-gray-200 dark:divide-gray-700">
          {workflows.length === 0 ? (
            <div className="p-8 text-center text-gray-500 dark:text-gray-400">
              No workflows yet. Click "New Workflow" to get started.
            </div>
          ) : (
            workflows.map((workflow) => (
              <div key={workflow.workflow_id} className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4 flex-1">
                    {getStatusIcon(workflow.status)}

                    <div className="flex-1">
                      <div className="flex items-center gap-3">
                        <h4 className="font-medium text-gray-900 dark:text-white">
                          {workflow.workflow_type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </h4>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(workflow.status)}`}>
                          {workflow.status}
                        </span>
                      </div>

                      <div className="flex items-center gap-4 mt-2 text-sm text-gray-600 dark:text-gray-400">
                        <span>ID: {workflow.workflow_id.slice(-8)}</span>
                        <span>•</span>
                        <span>Step {workflow.current_step} of {workflow.total_steps}</span>
                        <span>•</span>
                        <span>Created {new Date(workflow.created_at).toLocaleString()}</span>
                      </div>

                      {/* Progress Bar */}
                      {workflow.status === 'running' && (
                        <div className="mt-3 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${workflow.progress_percent}%` }}
                          />
                        </div>
                      )}

                      {workflow.error && (
                        <div className="mt-2 text-sm text-red-600 dark:text-red-400">
                          Error: {workflow.error}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2">
                    {workflow.status === 'pending' && (
                      <button
                        onClick={() => executeWorkflow(workflow.workflow_id)}
                        className="px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700 transition-colors"
                      >
                        Execute
                      </button>
                    )}
                    {workflow.status === 'running' && (
                      <button
                        onClick={() => cancelWorkflow(workflow.workflow_id)}
                        className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700 transition-colors"
                      >
                        Cancel
                      </button>
                    )}
                    {workflow.status === 'completed' && (
                      <button
                        className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 transition-colors flex items-center gap-1"
                      >
                        <Download className="w-4 h-4" />
                        Results
                      </button>
                    )}
                    <button className="px-3 py-1 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded text-sm hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors flex items-center gap-1">
                      <Eye className="w-4 h-4" />
                      Details
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export default WorkflowManager
