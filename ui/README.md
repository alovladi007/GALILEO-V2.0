# GeoSense Platform - UI

Next.js web application with CesiumJS 3D globe visualization.

## Quick Start

```bash
# Install dependencies
npm install

# Set up environment variables
cp .env.local.example .env.local
# Edit .env.local and add your Cesium Ion token

# Run development server
npm run dev
```

Visit http://localhost:3000

## Features

- **3D Globe Viewer**: CesiumJS-powered Earth visualization
- **Orbit Visualization**: Real-time satellite orbit rendering
- **Gravity Mapping**: Geophysical gravity anomaly visualization
- **API Integration**: Connects to FastAPI backend on port 5050

## Get Cesium Ion Token

1. Visit https://ion.cesium.com/
2. Sign up for a free account
3. Go to Access Tokens
4. Copy your default token
5. Add to `.env.local`:
   ```
   NEXT_PUBLIC_CESIUM_ION_TOKEN=your_token_here
   ```

## Build for Production

```bash
npm run build
npm start
```

## Tech Stack

- **Next.js 14**: React framework
- **TypeScript**: Type safety
- **TailwindCSS**: Styling
- **CesiumJS**: 3D globe visualization
- **Resium**: React components for Cesium
