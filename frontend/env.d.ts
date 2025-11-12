/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  // add more env variables here as needed
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

/// <reference types="vite/client" />

// Extend ImportMetaEnv if you have custom env variables
interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  // add more env variables here as needed
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
