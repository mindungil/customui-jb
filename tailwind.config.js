import typography from '@tailwindcss/typography';
import containerQueries from '@tailwindcss/container-queries';

/** @type {import('tailwindcss').Config} */
export default {
	darkMode: 'class',
	content: ['./src/**/*.{html,js,svelte,ts}'],
	theme: {
		extend: {
			fontSize: {
				'xs': ['0.875rem', { lineHeight: '1.25rem' }],     // 14px
				'sm': ['1rem', { lineHeight: '1.5rem' }],          // 16px
				'base': ['1.2rem', { lineHeight: '1.75rem' }],     // 19.2px
				'lg': ['1.25rem', { lineHeight: '1.75rem' }],      // 20px
				'xl': ['1.5rem', { lineHeight: '2rem' }],          // 24px
				'2xl': ['1.875rem', { lineHeight: '2.25rem' }],    // 30px
				'3xl': ['2.25rem', { lineHeight: '2.5rem' }],      // 36px
				'4xl': ['3rem', { lineHeight: '1' }],              // 48px
				'5xl': ['3.75rem', { lineHeight: '1' }],           // 60px
				'6xl': ['4.5rem', { lineHeight: '1' }],            // 72px
				'7xl': ['6rem', { lineHeight: '1' }],              // 96px
				'8xl': ['8rem', { lineHeight: '1' }],              // 128px
				'9xl': ['9rem', { lineHeight: '1' }],              // 144px
			},
                        colors: {
				gray: {
					50: 'var(--color-gray-50, #f9f9f9)',
					100: 'var(--color-gray-100, #ececec)',
					200: 'var(--color-gray-200, #e3e3e3)',
					300: 'var(--color-gray-300, #cdcdcd)',
					400: 'var(--color-gray-400, #b4b4b4)',
					500: 'var(--color-gray-500, #9b9b9b)',
					600: 'var(--color-gray-600, #676767)',
					700: 'var(--color-gray-700, #4e4e4e)',
					800: 'var(--color-gray-800, #333)',
					850: 'var(--color-gray-850, #262626)',
					900: 'var(--color-gray-900, #171717)',
					950: 'var(--color-gray-950, #0d0d0d)'
				}
			},
			typography: {
				DEFAULT: {
					css: {
						fontSize: '1.2rem',
						pre: false,
						code: false,
						'pre code': false,
						'code::before': false,
						'code::after': false
					}
				}
			},
			padding: {
				'safe-bottom': 'env(safe-area-inset-bottom)'
			},
			transitionProperty: {
				width: 'width'
			}
		}
	},
	plugins: [typography, containerQueries]
};
