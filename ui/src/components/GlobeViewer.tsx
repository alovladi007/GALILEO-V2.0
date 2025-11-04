'use client';

import { useEffect, useRef, useState } from 'react';
import {
  Viewer,
  Entity,
  PointGraphics,
  PathGraphics,
  ModelGraphics,
} from 'resium';
import { Cartesian3, Color, Ion } from 'cesium';

// Set your Cesium Ion access token here
Ion.defaultAccessToken = process.env.NEXT_PUBLIC_CESIUM_ION_TOKEN || 'your-token-here';

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
  const viewerRef = useRef(null);
  const [selectedPoint, setSelectedPoint] = useState<number | null>(null);

  useEffect(() => {
    // Initialize viewer with terrain
    if (viewerRef.current) {
      // Configure terrain provider
      // Configure imagery layers
    }
  }, []);

  // Convert ECEF to Cartesian3
  const toCartesian3 = (position: [number, number, number]) => {
    return new Cartesian3(position[0], position[1], position[2]);
  };

  // Color mapping for gravity anomalies
  const getColorForAnomaly = (anomaly: number): Color => {
    // Map anomaly to color scale (blue = negative, red = positive)
    const normalized = (anomaly + 100) / 200; // Assuming ±100 mGal range
    const clamped = Math.max(0, Math.min(1, normalized));
    
    return Color.fromHsl(
      (1 - clamped) * 0.6, // Hue: blue to red
      0.8, // Saturation
      0.5  // Lightness
    );
  };

  return (
    <div className="w-full h-screen">
      <Viewer
        ref={viewerRef}
        full
        timeline={true}
        animation={true}
        baseLayerPicker={true}
        geocoder={true}
        homeButton={true}
        sceneModePicker={true}
        navigationHelpButton={true}
        className="cesium-viewer"
      >
        {/* Satellite orbit path */}
        {satellitePositions.length > 0 && (
          <Entity
            name="Satellite Orbit"
            availability={undefined}
            position={toCartesian3(satellitePositions[0].position)}
          >
            <PathGraphics
              material={Color.YELLOW}
              width={2}
              resolution={60}
              leadTime={0}
              trailTime={satellitePositions.length * 60}
            />
            <PointGraphics
              pixelSize={10}
              color={Color.YELLOW}
              outlineColor={Color.BLACK}
              outlineWidth={2}
            />
          </Entity>
        )}

        {/* Gravity measurements visualization */}
        {gravityData.map((measurement, index) => (
          <Entity
            key={`gravity-${index}`}
            name={`Gravity Anomaly ${index}`}
            position={toCartesian3(measurement.position)}
            onClick={() => setSelectedPoint(index)}
          >
            <PointGraphics
              pixelSize={8}
              color={getColorForAnomaly(measurement.anomaly)}
              outlineColor={Color.WHITE}
              outlineWidth={1}
            />
          </Entity>
        ))}

        {/* Grid overlay for model discretization */}
        {showGrid && (
          // Grid entities would be generated here
          <></>
        )}
      </Viewer>

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
