import { useSession } from 'next-auth/react'
import { useEffect, useState } from 'react'

export function useAuth() {
  const { data: session, status } = useSession()
  const [user, setUser] = useState<any>(null)
  const [isAuthenticated, setIsAuthenticated] = useState(false)

  useEffect(() => {
    if (session?.user) {
      setUser(session.user)
      setIsAuthenticated(true)
    } else {
      setUser(null)
      setIsAuthenticated(false)
    }
  }, [session])

  return {
    user,
    isAuthenticated,
    isLoading: status === 'loading',
    session
  }
}
