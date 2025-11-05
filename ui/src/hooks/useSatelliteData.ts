import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

interface UseSatelliteDataProps {
  satellites: string[]
  time: Date
}

export function useSatelliteData({ satellites, time }: UseSatelliteDataProps) {
  return useQuery({
    queryKey: ['satelliteData', satellites, time.toISOString()],
    queryFn: async () => {
      // Simulate API call with mock data
      // In production, this would call the actual API
      return satellites.map(id => ({
        id,
        position: {
          lat: (Math.random() - 0.5) * 140,
          lon: (Math.random() - 0.5) * 360,
          alt: 450 + Math.random() * 20
        },
        velocity: {
          x: Math.random() * 7.5,
          y: Math.random() * 7.5,
          z: Math.random() * 0.1
        },
        gravity: 9.8 + (Math.random() - 0.5) * 0.001
      }))
    },
    enabled: satellites.length > 0,
    staleTime: 10000, // Consider data stale after 10 seconds
    refetchInterval: 30000, // Refetch every 30 seconds
  })
}
