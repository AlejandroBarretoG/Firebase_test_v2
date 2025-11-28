import React, { useState, useEffect } from 'react';
import { initFirebase, getConfigDisplay, mockSignIn, testRealAuthConnection, FirebaseApp, Auth } from './services/firebase';
import { mockWriteUserData, mockGetUserData } from './services/firestore_mock';
import { runGeminiTests } from './services/gemini';
import { StatusCard } from './components/StatusCard';
import { FirebaseWizard } from './components/FirebaseWizard';
import { ShieldCheck, Server, Database, Settings, XCircle, Code2, ChevronDown, ChevronUp, Bot, Sparkles, KeyRound, Cpu, UserCircle, HelpCircle } from 'lucide-react';

interface TestStep {
  id: string;
  title: string;
  description: string;
  status: 'idle' | 'loading' | 'success' | 'error';
  details?: string;
}

const DEFAULT_FIREBASE_CONFIG = {
  apiKey: "AIzaSyCjk5g2nAAClXrTY4LOSxAzS0YNE8lsSgw",
  authDomain: "studio-5674530481-7e956.firebaseapp.com",
  projectId: "studio-5674530481-7e956",
  storageBucket: "studio-5674530481-7e956.firebasestorage.app",
  messagingSenderId: "651553916706",
  appId: "1:651553916706:web:79ce4d5791126f3288877b"
};

type AppMode = 'firebase' | 'gemini';

const GEMINI_MODELS = [
  { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash (Recomendado)' },
  { id: 'gemini-2.5-flash-lite-preview-02-05', name: 'Gemini 2.5 Flash Lite' },
  { id: 'gemini-3-pro-preview', name: 'Gemini 3.0 Pro' },
];

/**
 * Extractor inteligente de configuración.
 * Capaz de procesar JSON puro, Objetos JS, o bloques de código completo copiados de Firebase Console.
 */
const safeJsonParse = (input: string) => {
  // 1. Limpieza previa: Quitar comentarios de una línea (//) y de bloque (/* */)
  // Esto es crucial porque JSON.parse ni new Function soportan comentarios bien si hay saltos de línea raros
  const cleanInput = input
    .replace(/\/\/.*$/gm, '') 
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .trim();

  let objectString = cleanInput;

  // 2. Detectar si es un bloque de código buscando la asignación de variable
  // Buscamos "firebaseConfig =" o "const firebaseConfig ="
  const configMarker = "firebaseConfig";
  const assignmentIndex = cleanInput.indexOf(configMarker);
  
  if (assignmentIndex !== -1) {
    // Buscar el primer '{' después de la asignación
    const searchFrom = cleanInput.substring(assignmentIndex);
    const firstBraceIndex = searchFrom.indexOf('{');
    
    if (firstBraceIndex !== -1) {
      // Algoritmo de balanceo de llaves para extraer SOLO el objeto, ignorando el resto del código (imports, init, etc)
      let openCount = 0;
      let endIndex = -1;
      
      for (let i = firstBraceIndex; i < searchFrom.length; i++) {
        if (searchFrom[i] === '{') openCount++;
        else if (searchFrom[i] === '}') openCount--;
        
        if (openCount === 0) {
          endIndex = i;
          break;
        }
      }

      if (endIndex !== -1) {
        // Extraemos exactamente desde { hasta }
        objectString = searchFrom.substring(firstBraceIndex, endIndex + 1);
      }
    }
  } else {
    // Si no hay variable explícita, intentamos encontrar el primer objeto {...} 
    // útil si copiaron solo el objeto pero quedaron caracteres sueltos
    const first = cleanInput.indexOf('{');
    const last = cleanInput.lastIndexOf('}');
    if (first !== -1 && last > first) {
       objectString = cleanInput.substring(first, last + 1);
    }
  }

  try {
    // 3. Evaluar como expresión JavaScript
    // new Function es seguro aquí porque corre en el cliente y permite sintaxis de objeto JS (claves sin comillas)
    const func = new Function(`return ${objectString}`);
    const result = func();
    if (result && typeof result === 'object') return result;
    throw new Error("El resultado no es un objeto válido.");
  } catch (jsError) {
    // 4. Fallback final: JSON.parse estricto
    // Si falló lo anterior, probamos el parseo estricto por si acaso
    return JSON.parse(input);
  }
};

const App: React.FC = () => {
  const [mode, setMode] = useState<AppMode>('firebase');
  
  // Firebase State
  const [firebaseInstance, setFirebaseInstance] = useState<FirebaseApp | null>(null);
  const [firebaseAuth, setFirebaseAuth] = useState<Auth | null>(null);
  const [firebaseConfigInput, setFirebaseConfigInput] = useState<string>(JSON.stringify(DEFAULT_FIREBASE_CONFIG, null, 2));
  const [testUid, setTestUid] = useState<string>('test-user-123');
  const [showConfig, setShowConfig] = useState(true);
  const [showWizard, setShowWizard] = useState(false);
  
  // Gemini State
  const [geminiModel, setGeminiModel] = useState<string>(GEMINI_MODELS[0].id);
  
  // Shared Steps State
  const [firebaseSteps, setFirebaseSteps] = useState<TestStep[]>([
    { id: 'config', title: 'Validación de Configuración', description: 'Analizando el JSON proporcionado.', status: 'idle' },
    { id: 'init', title: 'Inicialización del SDK', description: 'Ejecutando initializeApp() con la configuración.', status: 'idle' },
    { id: 'auth_module', title: 'Servicio de Autenticación', description: 'Verificando la instanciación del módulo Auth.', status: 'idle' },
    { id: 'real_auth_test', title: 'Prueba de Conexión REAL (Auth)', description: 'Intentando signInAnonymously() contra el servidor.', status: 'idle' },
    { id: 'auth_login', title: 'Simulación de Login', description: 'Estableciendo usuario activo (UID).', status: 'idle' },
    { id: 'db_write', title: 'Escritura Protegida (BD)', description: 'Guardando datos en /users/{uid}/data.', status: 'idle' },
    { id: 'db_read', title: 'Lectura Protegida (BD)', description: 'Recuperando datos propios del usuario.', status: 'idle' }
  ]);

  const [geminiSteps, setGeminiSteps] = useState<TestStep[]>([
    { id: 'connect', title: 'Verificación de API Key', description: 'Intentando establecer conexión inicial con Gemini.', status: 'idle' },
    { id: 'text', title: 'Generación de Texto', description: 'Prompt simple "Hola mundo".', status: 'idle' },
    { id: 'stream', title: 'Prueba de Streaming', description: 'Verificando recepción de chunks en tiempo real.', status: 'idle' },
    { id: 'tokens', title: 'Conteo de Tokens', description: 'Verificando endpoint countTokens.', status: 'idle' },
    { id: 'vision', title: 'Capacidad Multimodal', description: 'Analizando imagen de prueba (Pixel).', status: 'idle' },
    { id: 'system', title: 'Instrucciones del Sistema', description: 'Probando comportamiento de systemInstruction.', status: 'idle' },
    { id: 'embed', title: 'Embeddings', description: 'Generando vector con text-embedding-004.', status: 'idle' }
  ]);

  const updateStep = (mode: AppMode, id: string, updates: Partial<TestStep>) => {
    if (mode === 'firebase') {
      setFirebaseSteps(prev => prev.map(step => step.id === id ? { ...step, ...updates } : step));
    } else {
      setGeminiSteps(prev => prev.map(step => step.id === id ? { ...step, ...updates } : step));
    }
  };

  const runFirebaseTests = async () => {
    setFirebaseSteps(prev => prev.map(s => ({ ...s, status: 'idle', details: undefined })));
    setFirebaseInstance(null);
    setFirebaseAuth(null);
    
    // 1. Check Config
    updateStep('firebase', 'config', { status: 'loading' });
    await new Promise(resolve => setTimeout(resolve, 400)); 
    
    let parsedConfig: any;
    try {
      // Usamos safeJsonParse para permitir formatos más flexibles (como JS objects o código copiado)
      parsedConfig = safeJsonParse(firebaseConfigInput);
      
      if (!parsedConfig.apiKey || !parsedConfig.projectId) throw new Error("Faltan campos requeridos (apiKey, projectId).");
      
      updateStep('firebase', 'config', { 
        status: 'success', 
        details: JSON.stringify(getConfigDisplay(parsedConfig), null, 2) 
      });
    } catch (e: any) {
      updateStep('firebase', 'config', { status: 'error', details: `Formato Inválido: ${e.message}` });
      return;
    }

    // 2. Initialize App
    updateStep('firebase', 'init', { status: 'loading' });
    await new Promise(resolve => setTimeout(resolve, 600));

    const result = await initFirebase(parsedConfig);
    
    if (result.success && result.app) {
      setFirebaseInstance(result.app);
      updateStep('firebase', 'init', { 
        status: 'success', 
        details: `App Name: "${result.app.name}"\nAutomatic Data Collection: ${result.app.automaticDataCollectionEnabled}`
      });
    } else {
      updateStep('firebase', 'init', { status: 'error', details: result.error?.message || result.message });
      return;
    }

    // 3. Check Auth Module
    updateStep('firebase', 'auth_module', { status: 'loading' });
    await new Promise(resolve => setTimeout(resolve, 600));
    
    if (result.auth) {
       updateStep('firebase', 'auth_module', { 
        status: 'success', 
        details: `Auth SDK preparado.`
      });
    } else {
       updateStep('firebase', 'auth_module', { status: 'error', details: 'No se pudo obtener la instancia de Auth.' });
       return;
    }

    // 4. Test Real Auth Connection
    updateStep('firebase', 'real_auth_test', { status: 'loading' });
    
    const realAuthResult = await testRealAuthConnection(parsedConfig);

    if (realAuthResult.success && realAuthResult.data) {
       updateStep('firebase', 'real_auth_test', { 
        status: 'success', 
        details: `Conexión REAL exitosa. UID generado:\n${realAuthResult.data.uid}\nModo: Anónimo: ${realAuthResult.data.isAnonymous}`
      });
    } else {
       updateStep('firebase', 'real_auth_test', { status: 'error', details: realAuthResult.message });
       // No bloqueamos los siguientes pasos simulados, aunque la conexión real falle, para seguir probando la lógica.
    }

    // 5. Simulate Login
    updateStep('firebase', 'auth_login', { status: 'loading' });
    if (!testUid.trim()) {
      updateStep('firebase', 'auth_login', { status: 'error', details: 'Se requiere un UID de prueba para simular el login.' });
      return;
    }

    try {
      const authResult = await mockSignIn(testUid, result.app!);
      setFirebaseAuth(authResult);
      updateStep('firebase', 'auth_login', { 
        status: 'success', 
        details: `Usuario autenticado (Simulado):\nUID: ${authResult.currentUser?.uid}\nEstado: Sesión Activa`
      });
    } catch (e: any) {
      updateStep('firebase', 'auth_login', { status: 'error', details: e.message });
      return;
    }

    // 6. DB Write (Protected by UID)
    updateStep('firebase', 'db_write', { status: 'loading' });
    try {
      const docId = 'profile_v1';
      const sampleData = { role: 'tester', lastLogin: new Date().toISOString() };
      const writeResult = await mockWriteUserData(testUid, docId, sampleData);
      
      updateStep('firebase', 'db_write', { 
        status: 'success', 
        details: `Escritura exitosa en ruta protegida:\n${writeResult.path}\nDatos: ${JSON.stringify(sampleData)}`
      });
    } catch (e: any) {
      updateStep('firebase', 'db_write', { status: 'error', details: `Error de escritura: ${e.message}` });
      return;
    }

    // 7. DB Read (Protected by UID)
    updateStep('firebase', 'db_read', { status: 'loading' });
    try {
      const docId = 'profile_v1';
      const data = await mockGetUserData(testUid, docId);
      
      if (data) {
        updateStep('firebase', 'db_read', { 
          status: 'success', 
          details: `Lectura exitosa. Verificando propiedad:\nOwner: ${data._meta.createdBy} (Coincide con UID)\nData: ${JSON.stringify(data)}`
        });
      } else {
        throw new Error("El documento no se encontró o devolvió null.");
      }
    } catch (e: any) {
      updateStep('firebase', 'db_read', { status: 'error', details: `Error de lectura: ${e.message}` });
    }
  };

  const runGeminiTestFlow = async () => {
    setGeminiSteps(prev => prev.map(s => ({ ...s, status: 'idle', details: undefined })));

    // 1. Connection
    updateStep('gemini', 'connect', { status: 'loading' });
    const connResult = await runGeminiTests.connect(geminiModel);
    if (connResult.success) {
      updateStep('gemini', 'connect', { status: 'success', details: JSON.stringify(connResult.data, null, 2) });
    } else {
      updateStep('gemini', 'connect', { status: 'error', details: connResult.message });
      return; // Stop if connection fails
    }

    // 2. Text Generation
    updateStep('gemini', 'text', { status: 'loading' });
    const textResult = await runGeminiTests.generateText(geminiModel);
    if (textResult.success) updateStep('gemini', 'text', { status: 'success', details: JSON.stringify(textResult.data, null, 2) });
    else updateStep('gemini', 'text', { status: 'error', details: textResult.message });

    // 3. Streaming
    updateStep('gemini', 'stream', { status: 'loading' });
    const streamResult = await runGeminiTests.streamText(geminiModel);
    if (streamResult.success) updateStep('gemini', 'stream', { status: 'success', details: JSON.stringify(streamResult.data, null, 2) });
    else updateStep('gemini', 'stream', { status: 'error', details: streamResult.message });

    // 4. Tokens
    updateStep('gemini', 'tokens', { status: 'loading' });
    const tokenResult = await runGeminiTests.countTokens(geminiModel);
    if (tokenResult.success) updateStep('gemini', 'tokens', { status: 'success', details: JSON.stringify(tokenResult.data, null, 2) });
    else updateStep('gemini', 'tokens', { status: 'error', details: tokenResult.message });

    // 5. Vision
    updateStep('gemini', 'vision', { status: 'loading' });
    const visionResult = await runGeminiTests.vision(geminiModel);
    if (visionResult.success) updateStep('gemini', 'vision', { status: 'success', details: JSON.stringify(visionResult.data, null, 2) });
    else updateStep('gemini', 'vision', { status: 'error', details: visionResult.message });

    // 6. System Instruction
    updateStep('gemini', 'system', { status: 'loading' });
    const sysResult = await runGeminiTests.systemInstruction(geminiModel);
    if (sysResult.success) updateStep('gemini', 'system', { status: 'success', details: JSON.stringify(sysResult.data, null, 2) });
    else updateStep('gemini', 'system', { status: 'error', details: sysResult.message });

    // 7. Embeddings
    updateStep('gemini', 'embed', { status: 'loading' });
    const embedResult = await runGeminiTests.embedding();
    if (embedResult.success) updateStep('gemini', 'embed', { status: 'success', details: JSON.stringify(embedResult.data, null, 2) });
    else updateStep('gemini', 'embed', { status: 'error', details: embedResult.message });
  };

  // Run initial firebase test on mount if mode is firebase (default)
  useEffect(() => {
    if (mode === 'firebase') {
      runFirebaseTests();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const currentSteps = mode === 'firebase' ? firebaseSteps : geminiSteps;
  const allSuccess = currentSteps.every(s => s.status === 'success');
  const hasError = currentSteps.some(s => s.status === 'error');
  const isLoading = currentSteps.some(s => s.status === 'loading');

  return (
    <div className="min-h-screen bg-slate-50 p-6 md:p-12 font-sans relative">
      <FirebaseWizard isOpen={showWizard} onClose={() => setShowWizard(false)} />

      <div className="max-w-3xl mx-auto">
        
        {/* Header with Tabs */}
        <div className="mb-8 text-center">
          <div className="flex justify-center mb-6">
            <div className="bg-white p-1 rounded-xl shadow-sm border border-slate-200 inline-flex">
              <button 
                onClick={() => setMode('firebase')}
                className={`flex items-center gap-2 px-6 py-2 rounded-lg text-sm font-medium transition-all ${
                  mode === 'firebase' ? 'bg-orange-100 text-orange-700' : 'text-slate-500 hover:bg-slate-50'
                }`}
              >
                <Database size={18} />
                Firebase
              </button>
              <button 
                onClick={() => setMode('gemini')}
                className={`flex items-center gap-2 px-6 py-2 rounded-lg text-sm font-medium transition-all ${
                  mode === 'gemini' ? 'bg-blue-100 text-blue-700' : 'text-slate-500 hover:bg-slate-50'
                }`}
              >
                <Sparkles size={18} />
                Gemini AI
              </button>
            </div>
          </div>
          
          <h1 className="text-3xl font-bold text-slate-900">
            {mode === 'firebase' ? 'Firebase Connection Test' : 'Gemini API Diagnostics'}
          </h1>
          <p className="text-slate-500 mt-2">
            {mode === 'firebase' 
              ? 'Herramienta de diagnóstico para verificar integración de Firebase SDK y simulación Auth/DB.' 
              : 'Suite de pruebas para validar conectividad y funciones de Gemini API.'}
          </p>
        </div>

        {/* Configuration Section */}
        <div className="mb-8 bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <div className="w-full flex items-center justify-between p-4 bg-slate-50 border-b border-slate-100">
            <button 
              onClick={() => setShowConfig(!showConfig)}
              className="flex items-center gap-2 text-slate-800 font-medium hover:text-slate-900 transition-colors"
            >
              {mode === 'firebase' ? <Code2 size={20} className="text-orange-500" /> : <KeyRound size={20} className="text-blue-500" />}
              {mode === 'firebase' ? 'Configuración Firebase & Auth' : 'Configuración Gemini'}
              {showConfig ? <ChevronUp size={16} className="text-slate-400" /> : <ChevronDown size={16} className="text-slate-400" />}
            </button>
            
            {mode === 'firebase' && showConfig && (
              <button 
                onClick={() => setShowWizard(true)}
                className="text-xs flex items-center gap-1.5 text-blue-600 hover:text-blue-700 font-medium bg-blue-50 px-3 py-1.5 rounded-full hover:bg-blue-100 transition-colors"
              >
                <HelpCircle size={14} />
                ¿Cómo obtengo esto?
              </button>
            )}
          </div>
          
          {showConfig && (
            <div className="p-4">
              {mode === 'firebase' ? (
                <div className="space-y-4">
                  <div>
                    <p className="text-sm text-slate-500 mb-2">
                      Pega tu objeto <code>firebaseConfig</code> aquí. Acepta JSON estándar o formato JS de consola.
                    </p>
                    <textarea
                      value={firebaseConfigInput}
                      onChange={(e) => setFirebaseConfigInput(e.target.value)}
                      className="w-full h-32 p-4 font-mono text-xs md:text-sm bg-slate-900 text-green-400 rounded-lg border border-slate-300 outline-none resize-y"
                      spellCheck={false}
                    />
                    <div className="mt-2 text-right">
                       <button 
                         onClick={() => setFirebaseConfigInput(JSON.stringify(DEFAULT_FIREBASE_CONFIG, null, 2))}
                         className="text-xs text-slate-400 hover:text-slate-600 underline"
                       >
                         Restaurar defecto
                       </button>
                    </div>
                  </div>
                  
                  <div className="pt-4 border-t border-slate-100">
                     <label className="block text-sm font-medium text-slate-700 mb-1 flex items-center gap-2">
                       <UserCircle size={16} />
                       UID de Prueba (Login Simulado)
                     </label>
                     <div className="flex gap-2">
                       <input 
                         type="text" 
                         value={testUid}
                         onChange={(e) => setTestUid(e.target.value)}
                         placeholder="Ej: test-user-123"
                         className="flex-1 p-2 font-mono text-sm bg-slate-50 border border-slate-300 rounded-lg focus:ring-2 focus:ring-orange-500 outline-none"
                       />
                     </div>
                     <p className="text-xs text-slate-400 mt-1">
                       Se usará este ID para simular permisos de lectura/escritura en rutas protegidas: <code>/users/{'{uid}'}/...</code>
                     </p>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">
                      Modelo para Pruebas
                    </label>
                    <div className="relative">
                      <select
                        value={geminiModel}
                        onChange={(e) => setGeminiModel(e.target.value)}
                        className="w-full p-3 pr-10 appearance-none bg-white border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-sm font-medium text-slate-700"
                      >
                        {GEMINI_MODELS.map(m => (
                          <option key={m.id} value={m.id}>{m.name}</option>
                        ))}
                      </select>
                      <Cpu className="absolute right-3 top-3 text-slate-400 pointer-events-none" size={18} />
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Status Banner */}
        <div className={`mb-8 p-4 rounded-xl border shadow-sm flex items-center gap-4 transition-colors duration-500 ${
          hasError 
            ? 'bg-red-50 border-red-100 text-red-800' 
            : allSuccess && !isLoading && stepsAreComplete(currentSteps)
              ? 'bg-emerald-50 border-emerald-100 text-emerald-800'
              : 'bg-white border-slate-200 text-slate-600'
        }`}>
          {hasError ? (
            <div className="bg-red-100 p-2 rounded-full"><XCircle size={24} /></div>
          ) : allSuccess && !isLoading && stepsAreComplete(currentSteps) ? (
            <div className="bg-emerald-100 p-2 rounded-full"><ShieldCheck size={24} /></div>
          ) : (
             <div className="bg-slate-100 p-2 rounded-full"><Server size={24} className={isLoading ? "animate-pulse" : ""} /></div>
          )}
          
          <div>
            <h2 className="font-bold text-lg">
              {hasError ? 'Diagnóstico Fallido' : allSuccess && !isLoading && stepsAreComplete(currentSteps) ? 'Sistema Operativo' : 'Estado del Diagnóstico'}
            </h2>
            <p className="text-sm opacity-90">
              {hasError 
                ? 'Se encontraron problemas durante la ejecución.' 
                : allSuccess && !isLoading && stepsAreComplete(currentSteps)
                  ? 'Todas las pruebas pasaron exitosamente.' 
                  : 'Listo para iniciar pruebas.'}
            </p>
          </div>
        </div>

        {/* Steps List */}
        <div className="space-y-4">
          {currentSteps.map((step) => (
            <StatusCard
              key={step.id}
              title={step.title}
              description={step.description}
              status={step.status}
              details={step.details}
            />
          ))}
        </div>

        {/* Actions */}
        <div className="mt-8 flex justify-center">
          <button
            onClick={mode === 'firebase' ? runFirebaseTests : runGeminiTestFlow}
            disabled={isLoading}
            className={`flex items-center gap-2 px-6 py-3 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium shadow-lg hover:shadow-xl ${
              mode === 'firebase' 
                ? 'bg-orange-600 hover:bg-orange-700 shadow-orange-200' 
                : 'bg-blue-600 hover:bg-blue-700 shadow-blue-200'
            }`}
          >
            {isLoading ? <Server size={18} className="animate-spin" /> : mode === 'firebase' ? <Database size={18} /> : <Bot size={18} />}
            {isLoading ? 'Ejecutando...' : 'Iniciar Diagnóstico'}
          </button>
        </div>

      </div>
    </div>
  );
};

// Helper to check if tests actually ran (not just initial idle state)
function stepsAreComplete(steps: TestStep[]) {
  return steps.every(s => s.status !== 'idle' && s.status !== 'loading');
}

export default App;