import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
	plugins: [
		sveltekit(),
		viteStaticCopy({
			targets: [
				{
					src: 'node_modules/onnxruntime-web/dist/*.jsep.*',

					dest: 'wasm'
				}
			]
		})
	],
	define: {
		APP_VERSION: JSON.stringify(process.env.npm_package_version),
		APP_BUILD_HASH: JSON.stringify(process.env.APP_BUILD_HASH || 'dev-build')
	},
	build: {
		sourcemap: true
	},
	worker: {
		format: 'es'
	},
	server: {
    		host: true,
    			allowedHosts: ['ai.jb.go.kr', 'ai1.jb.go.kr'],
    	proxy: {
      		'/v1': {
        			target: 'http://127.0.0.1:8080',  // 백엔드
        			changeOrigin: true,
        			ws: true,
      			},
      		'/api': {
        			target: 'http://127.0.0.1:8080',
        			changeOrigin: true,
        			ws: true,
      			}
    		}
  	},
  	preview: {
    		allowedHosts: ['ai.jb.go.kr', 'ai1.jb.go.kr'],
  	}
});
