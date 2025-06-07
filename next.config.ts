/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true, // This will allow deployment even with ESLint errors
  },
  typescript: {
    ignoreBuildErrors: true, // This will allow deployment even with TypeScript errors
  },
}

module.exports = nextConfig