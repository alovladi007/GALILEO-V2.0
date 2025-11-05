# Web UI Documentation

## Overview

The Gravity Processing Web UI provides a comprehensive interface for visualizing and analyzing satellite gravity field data. Built with Next.js 14, CesiumJS, and Deck.gl, it offers real-time 3D visualization, time-series analysis, and job monitoring capabilities.

## Technology Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **3D Visualization**: CesiumJS + Resium
- **Data Overlays**: Deck.gl
- **Styling**: Tailwind CSS
- **State Management**: Zustand + React Query
- **Authentication**: NextAuth.js (OAuth2)
- **Charts**: Recharts
- **Testing**: Playwright + Jest

## Features

### 1. Globe Visualization
- **3D Earth Rendering**: Interactive Cesium globe with terrain
- **Satellite Tracking**: Real-time satellite position and orbit visualization
- **Baseline Vectors**: Inter-satellite range measurements
- **Gravity Field Overlay**: Color-coded gravity anomaly visualization
- **Uncertainty Maps**: Statistical uncertainty visualization

### 2. Time Controls
- **Timeline Slider**: Navigate through temporal data
- **Playback Controls**: Play/pause animation with adjustable speed
- **Date Range Selection**: Focus on specific time periods
- **Frame Rate**: 1x, 2x, 5x, 10x speed options

### 3. Data Analysis Panel
- **Overview Tab**: 
  - Gravity field statistics
  - Mean, standard deviation, min/max anomalies
  - 24-hour trend charts
  - Data quality indicators
- **Details Tab**:
  - Individual satellite information
  - Position, velocity, measurements
  - Baseline length graphs
  - Signal-to-noise ratios
- **Comparison Tab**:
  - Multiple processing run comparison
  - Difference plots
  - RMS statistics

### 4. Job Console
- **Active Jobs**: Monitor running processing tasks
- **Job History**: View completed and failed jobs
- **Quick Actions**: Start new processing jobs
- **Progress Tracking**: Real-time status updates
- **Error Reporting**: Detailed failure information

## Project Structure

```
ui/
├── src/
│   ├── app/                    # Next.js app router
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Main dashboard
│   │   ├── globals.css         # Global styles
│   │   ├── providers.tsx       # Context providers
│   │   └── api/               # API routes
│   │       └── auth/          # NextAuth endpoints
│   ├── components/            # React components
│   │   ├── GlobeVisualization.tsx
│   │   ├── Navigation.tsx
│   │   ├── TimeControls.tsx
│   │   ├── DataPanel.tsx
│   │   └── JobConsole.tsx
│   ├── hooks/                 # Custom React hooks
│   │   ├── useAuth.ts
│   │   ├── useSatelliteData.ts
│   │   ├── useGravityData.ts
│   │   └── useJobs.ts
│   ├── lib/                   # Utilities
│   │   └── api-client.ts
│   └── types/                 # TypeScript definitions
├── public/                    # Static assets
├── tests/                     # Test files
│   └── e2e/                  # Playwright tests
└── config files              # Various configurations
```

## Getting Started

### Prerequisites
- Node.js 20+
- npm or yarn
- Docker (optional)

### Installation

```bash
# Install dependencies
npm install

# Set environment variables
cp .env.example .env.local

# Run development server
npm run dev
```

### Environment Variables

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_CESIUM_ION_TOKEN=your-cesium-token
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-key
```

## Component Documentation

### GlobeVisualization

Main 3D visualization component using CesiumJS.

```typescript
<GlobeVisualization
  satelliteData={satellites}      // Array of satellite positions
  gravityData={field}            // Gravity field grid data
  showGravityOverlay={true}      // Toggle gravity visualization
  showUncertainty={false}        // Toggle uncertainty layer
  selectedSatellites={['GRACE-A']} // Active satellites
  selectedTime={new Date()}      // Current visualization time
/>
```

### TimeControls

Temporal navigation and animation controls.

```typescript
<TimeControls
  selectedTime={currentTime}
  timeRange={{ start, end }}
  isPlaying={false}
  playbackSpeed={1}
  onTimeChange={(time) => setTime(time)}
  onPlayPause={() => togglePlay()}
  onSpeedChange={(speed) => setSpeed(speed)}
/>
```

### DataPanel

Multi-tab data analysis interface.

```typescript
<DataPanel
  selectedTime={currentTime}
  satelliteData={satellites}
  gravityData={field}
  onRunCompare={(runId) => compareRun(runId)}
  selectedRun={comparisonRun}
/>
```

### JobConsole

Processing job monitoring and management.

```typescript
<JobConsole />  // Self-contained with internal state
```

## API Integration

### Authentication

```typescript
// Sign in
import { signIn } from 'next-auth/react'
await signIn('credentials', { 
  username: 'user', 
  password: 'pass' 
})

// Sign out
import { signOut } from 'next-auth/react'
await signOut()

// Get session
import { useSession } from 'next-auth/react'
const { data: session } = useSession()
```

### Data Fetching

```typescript
// Using React Query hooks
const { data: satellites } = useSatelliteData({
  satellites: ['GRACE-A', 'GRACE-B'],
  time: new Date()
})

const { data: gravity } = useGravityData({
  time: new Date(),
  runId: 'run-001'
})

const { jobs, createJob, cancelJob } = useJobs()
```

## Styling

### Tailwind Classes

Custom utility classes defined in `globals.css`:

- `.glass-effect` - Glassmorphism effect
- `.button-primary` - Primary action button
- `.button-secondary` - Secondary action button
- `.card` - Card container
- `.time-slider` - Custom time slider
- `.loading-spinner` - Loading animation

### Theme Colors

```css
primary: Blue palette (#3b82f6)
gravity: {
  low: #2563eb (blue)
  medium: #10b981 (green)
  high: #ef4444 (red)
}
```

## Performance Optimization

### Code Splitting
- Dynamic imports for heavy components (Cesium)
- Route-based code splitting
- Lazy loading of visualizations

### Rendering
- React.memo for expensive components
- useMemo/useCallback for computations
- Virtual scrolling for large lists

### Data Management
- React Query for caching
- Optimistic updates
- Background refetching
- Stale-while-revalidate

### Bundle Size
- Tree shaking
- Minification
- Compression
- CDN for static assets

## Testing

### Unit Tests
```bash
npm test
```

### E2E Tests
```bash
npm run test:e2e
```

### Test Coverage
```bash
npm run test:coverage
```

## Accessibility

### Features
- ARIA labels and roles
- Keyboard navigation
- Focus management
- Screen reader support
- High contrast mode
- Reduced motion support

### WCAG Compliance
- Level AA compliance
- Color contrast ratios
- Alternative text
- Semantic HTML
- Error identification

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
- Mobile browsers (iOS Safari, Chrome Mobile)

## Deployment

### Production Build
```bash
npm run build
npm start
```

### Docker Deployment
```bash
docker build -t gravity-ui .
docker run -p 3000:3000 gravity-ui
```

### Environment-specific Builds
```bash
# Staging
npm run build:staging

# Production
npm run build:production
```

## Monitoring

### Performance Metrics
- Core Web Vitals (LCP, FID, CLS)
- Custom performance marks
- Real User Monitoring (RUM)
- Error tracking with Sentry

### Analytics
- Page views
- User interactions
- Feature usage
- API performance

## Security

### Best Practices
- Content Security Policy (CSP)
- HTTPS enforcement
- XSS protection
- CSRF tokens
- Input sanitization
- Rate limiting

### Authentication
- JWT tokens with expiry
- Secure cookie storage
- OAuth2 flow
- Session management
- Role-based access control

## Troubleshooting

### Common Issues

1. **Cesium not loading**
   - Check Ion token configuration
   - Verify WebGL support
   - Clear browser cache

2. **Authentication failures**
   - Verify API connectivity
   - Check token expiry
   - Validate credentials

3. **Performance issues**
   - Reduce satellite count
   - Lower visualization quality
   - Enable hardware acceleration

4. **Data not updating**
   - Check WebSocket connection
   - Verify API responses
   - Review console errors

## Future Enhancements

- [ ] VR/AR support for immersive visualization
- [ ] Machine learning predictions
- [ ] Collaborative features
- [ ] Mobile native apps
- [ ] Advanced filtering and search
- [ ] Custom visualization presets
- [ ] Export capabilities (PDF, PNG, CSV)
- [ ] Real-time collaboration tools
