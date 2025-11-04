/** @type {import('next').NextConfig} */
const CopyWebpackPlugin = require('copy-webpack-plugin');
const path = require('path');

const nextConfig = {
  reactStrictMode: true,
  webpack: (config, { isServer }) => {
    // CesiumJS configuration
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };

      // Copy Cesium assets to public output
      config.plugins.push(
        new CopyWebpackPlugin({
          patterns: [
            {
              from: path.join(__dirname, 'node_modules/cesium/Build/Cesium/Workers'),
              to: '../public/Workers',
            },
            {
              from: path.join(__dirname, 'node_modules/cesium/Build/Cesium/ThirdParty'),
              to: '../public/ThirdParty',
            },
            {
              from: path.join(__dirname, 'node_modules/cesium/Build/Cesium/Assets'),
              to: '../public/Assets',
            },
            {
              from: path.join(__dirname, 'node_modules/cesium/Build/Cesium/Widgets'),
              to: '../public/Widgets',
            },
          ],
        })
      );
    }

    // Handle CesiumJS static files
    config.module.rules.push({
      test: /\.(glb|gltf)$/,
      use: {
        loader: 'file-loader',
      },
    });

    // Ignore cesium source maps
    config.ignoreWarnings = [/Failed to parse source map/];

    return config;
  },
};

module.exports = nextConfig;
