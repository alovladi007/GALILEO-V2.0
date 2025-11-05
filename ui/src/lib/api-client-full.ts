/**
 * Comprehensive API Client for GeoSense Platform
 *
 * Provides typed client functions for all 11 backend services:
 * - Simulation, Inversion, Control, Emulator, Calibration
 * - ML, TradeStudy, Compliance, Workflow, Database, Task
 *
 * Covers 104+ REST endpoints with full TypeScript support
 */

import axios, { AxiosInstance } from 'axios'
import { getSession } from 'next-auth/react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5050'

// Create axios instance with interceptors
const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: API_URL,
    headers: {
      'Content-Type': 'application/json',
    },
  })

  // Add auth token to requests
  client.interceptors.request.use(async (config) => {
    const session = await getSession()

    if (session?.accessToken) {
      config.headers.Authorization = `Bearer ${session.accessToken}`
    } else {
      config.headers.Authorization = `Bearer demo-token-for-testing`
    }

    return config
  })

  // Handle responses and errors
  client.interceptors.response.use(
    (response) => response,
    (error) => {
      if (error.response?.status === 401) {
        window.location.href = '/auth/signin'
      }
      return Promise.reject(error)
    }
  )

  return client
}

export const apiClient = createApiClient()

// =================================================================
// Simulation Service API
// =================================================================

export const simulationApi = {
  propagateFormation: (data: any) =>
    apiClient.post('/api/simulation/propagate-formation', data),

  propagateSingle: (data: any) =>
    apiClient.post('/api/simulation/propagate-single', data),

  computeBaselines: (data: any) =>
    apiClient.post('/api/simulation/compute-baselines', data),

  generateMeasurements: (data: any) =>
    apiClient.post('/api/simulation/generate-measurements', data),

  analyzeOrbit: (data: any) =>
    apiClient.post('/api/simulation/analyze-orbit', data),
}

// =================================================================
// Inversion Service API
// =================================================================

export const inversionApi = {
  estimateGravity: (data: any) =>
    apiClient.post('/api/inversion/estimate-gravity', data),

  validateResults: (data: any) =>
    apiClient.post('/api/inversion/validate-results', data),

  solveVariational: (data: any) =>
    apiClient.post('/api/inversion/solve-variational', data),

  solveSphericalHarmonic: (data: any) =>
    apiClient.post('/api/inversion/solve-spherical-harmonic', data),

  solveMascon: (data: any) =>
    apiClient.post('/api/inversion/solve-mascon', data),

  gridField: (data: any) =>
    apiClient.post('/api/inversion/grid-field', data),

  exportCoefficients: (data: any) =>
    apiClient.post('/api/inversion/export-coefficients', data),
}

// =================================================================
// Control Service API
// =================================================================

export const controlApi = {
  computeStationKeeping: (data: any) =>
    apiClient.post('/api/control/station-keeping', data),

  planManeuvers: (data: any) =>
    apiClient.post('/api/control/plan-maneuvers', data),

  computeFormationControl: (data: any) =>
    apiClient.post('/api/control/formation-control', data),

  validateManeuvers: (data: any) =>
    apiClient.post('/api/control/validate-maneuvers', data),

  optimizeManeuvers: (data: any) =>
    apiClient.post('/api/control/optimize-maneuvers', data),

  computeFuelBudget: (data: any) =>
    apiClient.post('/api/control/fuel-budget', data),
}

// =================================================================
// Emulator Service API
// =================================================================

export const emulatorApi = {
  createEmulator: (data: any) =>
    apiClient.post('/api/emulator/create', data),

  startEmulator: (emulatorId = 'default') =>
    apiClient.post(`/api/emulator/${emulatorId}/start`),

  stopEmulator: (emulatorId = 'default') =>
    apiClient.post(`/api/emulator/${emulatorId}/stop`),

  getCurrentState: (emulatorId = 'default') =>
    apiClient.get(`/api/emulator/${emulatorId}/state`),

  getSignalHistory: (emulatorId = 'default', params?: any) =>
    apiClient.get(`/api/emulator/${emulatorId}/history`, { params }),

  injectEvent: (emulatorId: string, data: any) =>
    apiClient.post(`/api/emulator/${emulatorId}/inject-event`, data),
}

// =================================================================
// Calibration Service API
// =================================================================

export const calibrationApi = {
  computeAllanDeviation: (data: any) =>
    apiClient.post('/api/calibration/allan-deviation', data),

  computePhaseFromRange: (data: any) =>
    apiClient.post('/api/calibration/phase-from-range', data),

  computeNoiseBudget: (data: any) =>
    apiClient.post('/api/calibration/noise-budget', data),

  assessMeasurementQuality: (data: any) =>
    apiClient.post('/api/calibration/measurement-quality', data),

  identifyNoiseSource: (data: any) =>
    apiClient.post('/api/calibration/identify-noise', data),
}

// =================================================================
// ML Service API
// =================================================================

export const mlApi = {
  // PINN operations
  createPinnModel: (data: any) =>
    apiClient.post('/api/ml/pinn/create', data),

  trainPinn: (data: any) =>
    apiClient.post('/api/ml/pinn/train', data),

  pinnInference: (data: any) =>
    apiClient.post('/api/ml/pinn/inference', data),

  savePinnModel: (data: any) =>
    apiClient.post('/api/ml/pinn/save', data),

  loadPinnModel: (data: any) =>
    apiClient.post('/api/ml/pinn/load', data),

  // U-Net operations
  createUnetModel: (data: any) =>
    apiClient.post('/api/ml/unet/create', data),

  trainUnet: (data: any) =>
    apiClient.post('/api/ml/unet/train', data),

  unetInference: (data: any) =>
    apiClient.post('/api/ml/unet/inference', data),

  unetUncertainty: (data: any) =>
    apiClient.post('/api/ml/unet/uncertainty', data),

  saveUnetModel: (data: any) =>
    apiClient.post('/api/ml/unet/save', data),

  loadUnetModel: (data: any) =>
    apiClient.post('/api/ml/unet/load', data),

  listModels: () =>
    apiClient.get('/api/ml/models/list'),
}

// =================================================================
// Trade Study Service API
// =================================================================

export const tradeStudyApi = {
  runBaselineStudy: (data: any) =>
    apiClient.post('/api/trade-study/baseline', data),

  runOrbitStudy: (data: any) =>
    apiClient.post('/api/trade-study/orbit', data),

  runOpticalStudy: (data: any) =>
    apiClient.post('/api/trade-study/optical', data),

  findParetoFront: (data: any) =>
    apiClient.post('/api/trade-study/pareto', data),

  sensitivityAnalysis: (data: any) =>
    apiClient.post('/api/trade-study/sensitivity', data),

  compareDesigns: (data: any) =>
    apiClient.post('/api/trade-study/compare', data),
}

// =================================================================
// Compliance Service API
// =================================================================

export const complianceApi = {
  // Audit logging
  logAuditEvent: (data: any) =>
    apiClient.post('/api/compliance/audit/log', data),

  queryAuditLogs: (params?: any) =>
    apiClient.get('/api/compliance/audit/query', { params }),

  verifyAuditChain: () =>
    apiClient.get('/api/compliance/audit/verify'),

  // Authorization
  createPolicy: (data: any) =>
    apiClient.post('/api/compliance/authz/policy', data),

  checkPermission: (data: any) =>
    apiClient.post('/api/compliance/authz/check-permission', data),

  assignRole: (data: any) =>
    apiClient.post('/api/compliance/authz/assign-role', data),

  listPolicies: () =>
    apiClient.get('/api/compliance/authz/policies'),

  // Secrets management
  storeSecret: (data: any) =>
    apiClient.post('/api/compliance/secrets/store', data),

  retrieveSecret: (data: any) =>
    apiClient.post('/api/compliance/secrets/retrieve', data),

  rotateSecret: (data: any) =>
    apiClient.post('/api/compliance/secrets/rotate', data),

  listSecrets: (params?: any) =>
    apiClient.get('/api/compliance/secrets/list', { params }),

  // Data retention
  createRetentionPolicy: (data: any) =>
    apiClient.post('/api/compliance/retention/policy', data),

  applyLegalHold: (data: any) =>
    apiClient.post('/api/compliance/retention/legal-hold', data),

  releaseLegalHold: (data: any) =>
    apiClient.post('/api/compliance/retention/release-hold', data),

  listRetentionPolicies: () =>
    apiClient.get('/api/compliance/retention/policies'),

  listLegalHolds: (params?: any) =>
    apiClient.get('/api/compliance/retention/legal-holds', { params }),
}

// =================================================================
// Workflow Service API
// =================================================================

export const workflowApi = {
  listTemplates: () =>
    apiClient.get('/api/workflow/templates'),

  getTemplate: (workflowType: string) =>
    apiClient.get(`/api/workflow/templates/${workflowType}`),

  submitWorkflow: (data: any) =>
    apiClient.post('/api/workflow/submit', data),

  executeWorkflow: (workflowId: string) =>
    apiClient.post(`/api/workflow/${workflowId}/execute`),

  getWorkflowStatus: (workflowId: string) =>
    apiClient.get(`/api/workflow/${workflowId}/status`),

  listWorkflows: (params?: any) =>
    apiClient.get('/api/workflow/list', { params }),

  cancelWorkflow: (workflowId: string) =>
    apiClient.post(`/api/workflow/${workflowId}/cancel`),

  getWorkflowOutputs: (workflowId: string) =>
    apiClient.get(`/api/workflow/${workflowId}/outputs`),
}

// =================================================================
// Database Service API
// =================================================================

export const databaseApi = {
  // Users
  createUser: (data: any) =>
    apiClient.post('/api/db/users', data),

  getUser: (username: string) =>
    apiClient.get(`/api/db/users/${username}`),

  listUsers: (params?: any) =>
    apiClient.get('/api/db/users', { params }),

  // Jobs
  createJob: (data: any) =>
    apiClient.post('/api/db/jobs', data),

  updateJobStatus: (jobId: string, data: any) =>
    apiClient.put(`/api/db/jobs/${jobId}/status`, data),

  getJob: (jobId: string) =>
    apiClient.get(`/api/db/jobs/${jobId}`),

  listJobs: (params?: any) =>
    apiClient.get('/api/db/jobs', { params }),

  // Observations
  createObservation: (data: any) =>
    apiClient.post('/api/db/observations', data),

  bulkCreateObservations: (data: any) =>
    apiClient.post('/api/db/observations/bulk', data),

  queryObservations: (params?: any) =>
    apiClient.get('/api/db/observations', { params }),

  // Products
  createProduct: (data: any) =>
    apiClient.post('/api/db/products', data),

  queryProducts: (params?: any) =>
    apiClient.get('/api/db/products', { params }),

  // Baseline vectors
  createBaselineVector: (data: any) =>
    apiClient.post('/api/db/baselines', data),

  bulkCreateBaselineVectors: (data: any) =>
    apiClient.post('/api/db/baselines/bulk', data),

  queryBaselineVectors: (params?: any) =>
    apiClient.get('/api/db/baselines', { params }),

  // Audit logs
  createAuditLog: (data: any) =>
    apiClient.post('/api/db/audit-logs', data),

  queryAuditLogs: (params?: any) =>
    apiClient.get('/api/db/audit-logs', { params }),
}

// =================================================================
// Task Service API
// =================================================================

export const taskApi = {
  submitTask: (data: any) =>
    apiClient.post('/api/tasks/submit', data),

  submitChain: (data: any) =>
    apiClient.post('/api/tasks/submit-chain', data),

  submitGroup: (data: any) =>
    apiClient.post('/api/tasks/submit-group', data),

  getTaskStatus: (taskId: string) =>
    apiClient.get(`/api/tasks/${taskId}/status`),

  getTaskResult: (taskId: string, params?: any) =>
    apiClient.get(`/api/tasks/${taskId}/result`, { params }),

  listActiveTasks: () =>
    apiClient.get('/api/tasks/active'),

  listScheduledTasks: () =>
    apiClient.get('/api/tasks/scheduled'),

  cancelTask: (taskId: string, params?: any) =>
    apiClient.post(`/api/tasks/${taskId}/cancel`, null, { params }),

  retryTask: (taskId: string) =>
    apiClient.post(`/api/tasks/${taskId}/retry`),

  getWorkerStats: () =>
    apiClient.get('/api/tasks/workers/stats'),

  pingWorkers: () =>
    apiClient.get('/api/tasks/workers/ping'),
}

// =================================================================
// Unified API Export
// =================================================================

export const api = {
  simulation: simulationApi,
  inversion: inversionApi,
  control: controlApi,
  emulator: emulatorApi,
  calibration: calibrationApi,
  ml: mlApi,
  tradeStudy: tradeStudyApi,
  compliance: complianceApi,
  workflow: workflowApi,
  database: databaseApi,
  task: taskApi,

  // Legacy ops endpoints (for backward compatibility)
  healthCheck: () => apiClient.get('/health'),
}

export default api
