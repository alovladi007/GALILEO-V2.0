import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'GeoSense Platform - Mission Control',
  description: 'AI-Enhanced Space-Based Geophysical Sensing Platform',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
