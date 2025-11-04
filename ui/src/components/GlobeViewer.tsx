'use client';

import { useEffect, useRef, useState } from 'react';
import * as Cesium from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';

interface SatellitePosition {
  time: Date;
  position: [number, number, number]; // [x, y, z] in meters (ECEF)
  velocity: [number, number, number];
}

interface GravityMeasurement {
  position: [number, number, number];
  anomaly: number; // mGal
}

interface GlobeViewerProps {
  satellitePositions?: SatellitePosition[];
  gravityData?: GravityMeasurement[];
  showGrid?: boolean;
}

export default function GlobeViewer({
  satellitePositions = [],
  gravityData = [],
  showGrid = false,
}: GlobeViewerProps) {
  const viewerContainerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Cesium.Viewer | null>(null);
  const [selectedPoint, setSelectedPoint] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!viewerContainerRef.current) return;

    const initCesium = async () => {
      try {
        // Set Cesium Ion token
        const token = process.env.NEXT_PUBLIC_CESIUM_ION_TOKEN || 'your-token-here';
        Cesium.Ion.defaultAccessToken = token;

        // Set Cesium base URL for assets (Workers, etc.)
        (window as any).CESIUM_BASE_URL = '/';

        // Configure Cesium to load assets from public directory
        if ((Cesium as any).buildModuleUrl) {
          (Cesium as any).buildModuleUrl.setBaseUrl('/');
        }

        // Create the Viewer
        const viewer = new Cesium.Viewer(viewerContainerRef.current!, {
          terrain: Cesium.Terrain.fromWorldTerrain(),
          timeline: true,
          animation: true,
          baseLayerPicker: true,
          geocoder: true,
          homeButton: true,
          sceneModePicker: true,
          navigationHelpButton: true,
          fullscreenButton: true,
        });

        viewerRef.current = viewer;

        // Add satellite positions if provided
        if (satellitePositions.length > 0) {
          satellitePositions.forEach((satPos, index) => {
            viewer.entities.add({
              name: `Satellite ${index + 1}`,
              position: Cesium.Cartesian3.fromArray(satPos.position),
              point: {
                pixelSize: 10,
                color: Cesium.Color.YELLOW,
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 2,
              },
            });
          });
        }

        // Add gravity measurements if provided
        if (gravityData.length > 0) {
          gravityData.forEach((measurement, index) => {
            const normalized = (measurement.anomaly + 100) / 200;
            const clamped = Math.max(0, Math.min(1, normalized));
            const color = Cesium.Color.fromHsl((1 - clamped) * 0.6, 0.8, 0.5);

            viewer.entities.add({
              name: `Gravity Anomaly ${index}`,
              position: Cesium.Cartesian3.fromArray(measurement.position),
              point: {
                pixelSize: 8,
                color: color,
                outlineColor: Cesium.Color.WHITE,
                outlineWidth: 1,
              },
            });
          });
        }

        setIsLoading(false);
      } catch (err) {
        console.error('Failed to initialize Cesium:', err);
        setError(err instanceof Error ? err.message : 'Failed to initialize 3D viewer');
        setIsLoading(false);
      }
    };

    initCesium();

    // Cleanup
    return () => {
      if (viewerRef.current && !viewerRef.current.isDestroyed()) {
        viewerRef.current.destroy();
      }
    };
  }, [satellitePositions, gravityData]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Initializing Cesium...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900">
        <div className="text-center max-w-md">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-xl font-bold text-white mb-2">Failed to Load 3D Viewer</h2>
          <p className="text-gray-400 mb-4">{error}</p>
          <p className="text-sm text-gray-500">
            Please check your Cesium Ion token configuration in .env.local
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-screen">
      <div ref={viewerContainerRef} className="w-full h-full" />

      {/* Info panel */}
      {selectedPoint !== null && gravityData[selectedPoint] && (
        <div className="absolute top-4 right-4 bg-white p-4 rounded-lg shadow-lg">
          <h3 className="font-bold mb-2">Gravity Measurement</h3>
          <div className="text-sm space-y-1">
            <div>
              <span className="font-medium">Anomaly:</span>{' '}
              {gravityData[selectedPoint].anomaly.toFixed(2)} mGal
            </div>
            <div>
              <span className="font-medium">Position:</span>
              <div className="pl-4">
                X: {gravityData[selectedPoint].position[0].toFixed(2)} m<br />
                Y: {gravityData[selectedPoint].position[1].toFixed(2)} m<br />
                Z: {gravityData[selectedPoint].position[2].toFixed(2)} m
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
