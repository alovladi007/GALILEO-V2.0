import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

interface UseGravityDataProps {
  time: Date
  runId?: string | null
}

export function useGravityData({ time, runId }: UseGravityDataProps) {
  return useQuery({
    queryKey: ['gravityData', time.toISOString(), runId],
    queryFn: async () => {
      // Simulate API call with mock gravity field data
      // In production, this would fetch actual gravity field data
      
      const gridSize = 180
      const grid = []
      
      for (let lat = 0; lat < gridSize; lat++) {
        const row = []
        for (let lon = 0; lon < gridSize * 2; lon++) {
          // Generate realistic gravity anomaly values
          const latRad = (lat - 90) * Math.PI / 180
          const lonRad = (lon - 180) * Math.PI / 180
          
          const value = 
            Math.sin(latRad * 2) * 30 +
            Math.cos(lonRad * 3) * 20 +
            Math.sin(latRad * lonRad) * 10 +
            (Math.random() - 0.5) * 5
          
          row.push(value)
        }
        grid.push(row)
      }
      
      return {
        grid,
        bounds: {
          north: 90,
          south: -90,
          east: 180,
          west: -180
        },
        colorScale: {
          min: -50,
          max: 50
        },
        mean: 0.023,
        std: 15.7,
        maxAnomaly: 48.3,
        minAnomaly: -42.1,
        timestamp: time.toISOString(),
        runId: runId || 'current'
      }
    },
    staleTime: 60000, // Consider data stale after 1 minute
  })
}
