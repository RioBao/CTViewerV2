import { defineConfig } from 'vite';

export default defineConfig({
  base: '/CTViewerV2/',
  assetsInclude: ['**/*.wgsl'],
  build: {
    target: 'esnext',
  },
});
