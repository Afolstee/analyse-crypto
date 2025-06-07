module.exports = {
    extends: [
      'next/core-web-vitals', // Recommended for Next.js projects
      'plugin:@typescript-eslint/recommended', // TypeScript support
    ],
    rules: {
      // Ignore unused variables prefixed with _
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
        },
      ],
      // Warn about missing dependencies in useEffect/useCallback
      'react-hooks/exhaustive-deps': 'warn',
      // Warn about using `any` type
      '@typescript-eslint/no-explicit-any': 'warn',
    },
  };