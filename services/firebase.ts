// We are defining local types and mocks because the 'firebase' module is missing
// or incompatible in this environment, causing "Module has no exported member" errors.

// @ts-ignore
import { initializeApp } from 'firebase/app'; 
// @ts-ignore
import { getAuth, signInAnonymously } from 'firebase/auth';

export interface FirebaseOptions {
  apiKey?: string;
  authDomain?: string;
  projectId?: string;
  storageBucket?: string;
  messagingSenderId?: string;
  appId?: string;
  [key: string]: any;
}

export interface FirebaseApp {
  name: string;
  options: FirebaseOptions;
  automaticDataCollectionEnabled: boolean;
}

export interface Auth {
  app: FirebaseApp;
  currentUser: { uid: string } | null;
}

interface FirebaseInitResult {
  success: boolean;
  app?: FirebaseApp;
  auth?: Auth;
  error?: any;
  message: string;
  data?: any;
}

/**
 * Initializes the Firebase application with the provided configuration.
 * Mock implementation to allow compilation without valid firebase module.
 */
export const initFirebase = async (config: FirebaseOptions): Promise<FirebaseInitResult> => {
  try {
    console.log("Mocking Firebase initialization...");
    
    // Simulate async operation
    await new Promise(resolve => setTimeout(resolve, 500));

    const app: FirebaseApp = {
      name: "[DEFAULT]",
      options: config,
      automaticDataCollectionEnabled: true
    };

    // Initialize Auth service mock
    const auth: Auth = {
      app,
      currentUser: null
    };

    return {
      success: true,
      app,
      auth,
      message: "Firebase SDK inicializado correctamente (Mock)."
    };
  } catch (error: any) {
    console.error("Firebase initialization error:", error);
    return {
      success: false,
      error: error,
      message: error.message || "Error desconocido al inicializar Firebase."
    };
  }
};

/**
 * Función que realiza una prueba de conexión REAL de Auth.
 * Inicializa la app y realiza un signInAnonymously() con el SDK real.
 */
export const testRealAuthConnection = async (config: FirebaseOptions): Promise<FirebaseInitResult> => {
  try {
    if (!config.apiKey || !config.projectId) {
      throw new Error("La configuración de Firebase está incompleta.");
    }

    // Usamos un nombre de app único (timestamp) para evitar el error 'app/duplicate-app'
    const uniqueAppName = `test-auth-${Date.now()}`;

    // 1. Inicialización REAL (Usa initializeApp del SDK real)
    const app = initializeApp(config, uniqueAppName); 
    const auth = getAuth(app);

    // 2. Inicio de Sesión Anónimo REAL
    const userCredential = await signInAnonymously(auth);

    // 3. Devolver resultado exitoso con el UID real
    return {
      success: true,
      app: app as any,
      auth: auth as any,
      message: "Conexión Real exitosa.",
      data: {
        uid: userCredential.user.uid,
        isAnonymous: userCredential.user.isAnonymous,
        appName: app.name
      }
    };
  } catch (error: any) {
    // Si falla, devuelve el código de error real de Firebase
    return {
      success: false,
      message: `Fallo de conexión REAL: ${error.code || error.message}`
    };
  }
};

/**
 * Simulates signing in a user.
 */
export const mockSignIn = async (uid: string, app: FirebaseApp): Promise<Auth> => {
  await new Promise(resolve => setTimeout(resolve, 400));
  
  return {
    app,
    currentUser: { uid }
  };
};

/**
 * Simulates signing out.
 */
export const mockSignOut = async (app: FirebaseApp): Promise<Auth> => {
  await new Promise(resolve => setTimeout(resolve, 200));
  
  return {
    app,
    currentUser: null
  };
};

export const getConfigDisplay = (config: any) => {
  if (!config || !config.apiKey) return { ...config };
  
  // Return a masked version of the config for display purposes
  return {
    ...config,
    apiKey: `${config.apiKey.substring(0, 6)}...${config.apiKey.substring(config.apiKey.length - 4)}`
  };
};