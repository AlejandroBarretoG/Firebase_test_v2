import React, { useState } from 'react';
import { X, ChevronRight, ChevronLeft, LayoutDashboard, Loader2, ExternalLink, Maximize2, Minimize2, Settings, PlusCircle, FileSearch } from 'lucide-react';

interface FirebaseWizardProps {
  isOpen: boolean;
  onClose: () => void;
}

const getImageUrl = (id: string) => `https://lh3.googleusercontent.com/d/${id}`;

// --- GUÍA 1: CREACIÓN E INSTALACIÓN (19 Pasos) ---
const SETUP_STEPS = [
  { title: "Crear Proyecto", desc: "Haz clic en 'Crear un proyecto' en la consola de Firebase.", imgId: "19-KnrkDwe36RKOntgZbrmoijNjQxW66M" },
  { title: "Nombrar Proyecto", desc: "Ingresa un nombre para tu proyecto (ej. Conexion-Tester-Suite).", imgId: "1NuV_zaGx7tR7Sya_pV_iVtYCuo5GAxgA" },
  { title: "Aceptar y Continuar", desc: "Haz clic en 'Continuar'.", imgId: "1caBaGGArmt2ORGo09TW4kHS7SkCOO5m_" },
  { title: "Desactivar Analytics", desc: "Recomendamos desactivar Google Analytics para este entorno de pruebas.", imgId: "14OJo6GGKlkERnCEpnsB94iy7NnGCMQfp" },
  { title: "Confirmar Creación", desc: "Haz clic en 'Crear proyecto'.", imgId: "17UG4mjZkAuAs1x8qMa6RotG_Qk2CsYfy" },
  { title: "Esperar", desc: "Espera unos segundos mientras Firebase configura los recursos.", imgId: "1gGDzGq6cXRaVIhVKeMHB075GcaaXP--G" },
  { title: "Proyecto Listo", desc: "Tu nueva proyecto está listo para usarse. Haz clic en 'Continuar'.", imgId: "19VJq1xFo41IZVSY8nlIWCdkGPKJae-fn" },
  { title: "Crear App", desc: "Haz clic en '+ Crear app'.", imgId: "1JFtUgLPRr_XN8W1R-wWCo2xqI1KXJt_B" },
  { title: "Seleccionar tipo de App", desc: "En el dashboard, selecciona el icono de Web (</>).", imgId: "11_vupCeD-IEGyRDG4b3NTlxCMNaVkprj" },
  { title: "Asignar Sobrenombre", desc: "Asigna un apodo a la aplicación (ej. web-tester).", imgId: "1IQkpWxaJjlukkm90U9P4v30_O-KznqC0" },
  { title: "Registrar App", desc: "Haz clic en 'Registrar app'.", imgId: "1HANYd5wraXTGU2CFpLSdBtaVGSzx-q8f" },
  { title: "Regresar a la Consola", desc: "La app a sido creada, Haz clic en 'Ir a la consola' para regresar.", imgId: "1gsJPgNqZl3aDt2WtUoGbIUY1fsGzbJhl" },
  { title: "Seleccionar Compilación", desc: "En el menú lateral izquierdo, ve a 'Compilación'.", imgId: "1cpsfK8rkurE5LLZCNhfm9Gndp6BsDndG" },
  { title: "Seleccionar Authentication", desc: "En el menú lateral izquierdo, ve a Compilación > Authentication.", imgId: "19Mnpyn1FbzN3eGqwUHOYMAGjGc8LXYDo" },
  { title: "Comenzar", desc: "Haz clic en 'Comenzar' para configurar los usuarios.", imgId: "1kdJ2lLHJIj0Bcoh4oWtPszLBAT0CRzkb" },
  { title: "Seleccionar 'Método de acceso'", desc: "Selecciona la pestaña 'Método de acceso'.", imgId: "1ildmHilvI1J_A3TguG6nN2QqJbjofKmK" },
  { title: "Proveedor Anónimo", desc: "En la lista de proveedores, selecciona 'Anónimo'.", imgId: "1G1ZmpNhgylHnaJpHly1VocsQEg8jw9HV" },
  { title: "Habilitar", desc: "Activa el interruptor 'Habilitar' para permitir sesiones sin contraseña.", imgId: "1hmMhVRsouhbtwXzHh4BtoUbcjsrjVAZ-" },
  { title: "Guardar", desc: "Haz clic en 'Guardar'.", imgId: "12IlGN-z6pilFQ4z7nx6eZnvz2BRTChmB" },
  { title: "Todo Listo", desc: "Has completado la configuración necesaria.", imgId: "1BhiJDsaiEQ617MGe2wBvUJcR_iu5Hrq7" },
];

// --- GUÍA 2: ENCONTRAR CONFIGURACIÓN (13 Pasos) ---
const FIND_CONFIG_STEPS = [
  { title: "Dashboard", desc: "En la lista de proyectos busca el proyecto recientemente creado, en este ejemplo 'Conexión-Tester-Suite'.", imgId: "1QP6FPgtJC78FQ0n5a6t1lObfkxGUluhN" },
  { title: "Menú Configuración", desc: "Desde la vista principal del proyecto, localiza el icono de engranaje", imgId: "1o00SFqjcErxuqYi-jfFCsN5WLH5nOYPn" },
  { title: "Configuración del Proyecto", desc: "Selecciona 'Configuración del proyecto' en el menú desplegable.", imgId: "1w0qXYIc1MQa3hf-cSvHpFfC0W81oJLmZ" },
  { title: "Desplazar Abajo", desc: "Desplázate hacia la parte inferior de la página hasta 'Tus apps'.", imgId: "1mkIE1dcTE4GnYQToBdPNsfi2_obsaCJI" },
  { title: "Presiona el boton 'Copiar'", desc: "Presiona el botón 'Copiar' para copiar el SDK.", imgId: "1Y3pRGqfDjIGWQ_EJlDPoNyBy4ZTNKY-r" },
  { title: "Seleccionar Compilación", desc: "En el menú lateral izquierdo, ve a 'Compilación'.", imgId: "1UJiGs-YGq41dU-eEXy7N1sJT1-QFF3vf" },
  { title: "Seleccionar Authentication", desc: "En el menú lateral izquierdo, ve a Compilación > Authentication.", imgId: "17YRXKpdwXKJLF-9JMlgCh4gdrIhixJYO" },
  { title: "Ver Código", desc: "Observa el código de configuración de Firebase.", imgId: "1ygZUv45SQOoTBr1OLmJsOiRtfFvNjI3n" },
  { title: "Seleccionar 'Método de acceso'", desc: "Selecciona la pestaña 'Método de acceso'.", imgId: "161Coywt9mCPeyAgxfXOFaox8Ylz0TqOD" },
  { title: "Revisar Proveedores", desc: "Confirma que 'Anónimo' esté en la lista y se encuentre 'Habilitado'.", imgId: "1HEdvQaJ4iBEdgxxDvo_TQ6Wk7--WW3Kj" },
  { title: "Firebase Connection Test", desc: "Regresa a la aplicación a la sección 'Firebase Connection Test'", imgId: "1kEVofJIfxzPhd7xENyJdyEjdey_M2wvb" },
  { title: "Elimina el texto", desc: "Elimina todo el texto del cuadro de texto.", imgId: "1KRPTRBc5g-pZJQOZWVRyrwCeQ7wn3m2r" },
  { title: "Pega el texto 'ctrl + v'", desc: "Con 'ctrl + v', pega el contenido del SDK en el cuadro de texto.", imgId: "1GJm_U6gRpGTovDh5UcDqcIZkGMceT34-" },
  { title: "Realiza una Prueba de Diagnóstico", desc: "Desplaza la pantalla hasta la parte de abajo y presiona el botón 'Iniciar Diagnóstico'.", imgId: "1zeFOuYbAyd3-V5fHXE-jgFsIaopTcfov" },
  { title: "Todo Listo", desc: "Revisa el resultado del analisis.", imgId: "1dcqG1pk-V0gEVKYlY_Brb_LZGTrjcaAQ" },
];

const StepImage = ({ imgId, alt, isFullscreen }: { imgId: string, alt: string, isFullscreen: boolean }) => {
  const [loaded, setLoaded] = useState(false);

  return (
    <div className={`relative w-full bg-slate-100 rounded-lg overflow-hidden border border-slate-200 flex items-center justify-center transition-all duration-300 ${
      isFullscreen ? 'h-full' : 'h-64 md:h-96'
    }`}>
      {!loaded && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400">
          <Loader2 className="w-8 h-8 animate-spin mb-2" />
          <span className="text-xs">Cargando imagen...</span>
        </div>
      )}
      <img 
        src={getImageUrl(imgId)} 
        alt={alt}
        className={`w-full h-full object-contain transition-opacity duration-300 ${loaded ? 'opacity-100' : 'opacity-0'}`}
        onLoad={() => setLoaded(true)}
      />
    </div>
  );
};

export const FirebaseWizard: React.FC<FirebaseWizardProps> = ({ isOpen, onClose }) => {
  const [activeGuide, setActiveGuide] = useState<'setup' | 'config'>('setup');
  const [currentStep, setCurrentStep] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  if (!isOpen) return null;

  const currentSteps = activeGuide === 'setup' ? SETUP_STEPS : FIND_CONFIG_STEPS;

  const nextStep = () => {
    if (currentStep < currentSteps.length - 1) setCurrentStep(prev => prev + 1);
    else onClose();
  };

  const prevStep = () => {
    if (currentStep > 0) setCurrentStep(prev => prev - 1);
  };

  const switchGuide = (guide: 'setup' | 'config') => {
    setActiveGuide(guide);
    setCurrentStep(0);
  };

  const stepData = currentSteps[currentStep];

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-900/80 backdrop-blur-sm transition-opacity animate-in fade-in duration-200">
      <div className={`bg-white shadow-2xl overflow-hidden flex flex-col transition-all duration-300 ${
        isFullscreen 
          ? 'fixed inset-0 w-full h-full rounded-none z-[110]' 
          : 'rounded-xl w-full max-w-4xl max-h-[90vh]'
      }`}>
        
        {/* Header */}
        <div className="border-b border-slate-100 bg-white shrink-0">
          <div className="p-4 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h2 className="font-bold text-slate-800 flex items-center gap-2 text-lg">
                <LayoutDashboard size={20} className="text-orange-500" />
                Asistente Firebase
              </h2>
              
              <a 
                href="https://console.firebase.google.com/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="hidden sm:flex items-center gap-1.5 text-xs font-medium text-blue-600 hover:text-blue-800 bg-blue-50 hover:bg-blue-100 px-3 py-1.5 rounded-full transition-colors"
              >
                Ir a Firebase Console <ExternalLink size={12} />
              </a>
            </div>

            <div className="flex items-center gap-2">
               <a 
                href="https://console.firebase.google.com/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="sm:hidden p-2 text-blue-600 hover:bg-blue-50 rounded-full"
              >
                <ExternalLink size={20} />
              </a>
              <button 
                onClick={() => setIsFullscreen(!isFullscreen)}
                className="p-2 hover:bg-slate-100 rounded-full text-slate-400 hover:text-slate-600 transition-colors hidden sm:block"
              >
                {isFullscreen ? <Minimize2 size={20} /> : <Maximize2 size={20} />}
              </button>
              <button onClick={onClose} className="p-2 hover:bg-slate-100 rounded-full text-slate-400 hover:text-slate-600 transition-colors">
                <X size={20} />
              </button>
            </div>
          </div>

          {/* Guide Selector Tabs */}
          <div className="flex px-4 gap-4 overflow-x-auto">
            <button
              onClick={() => switchGuide('setup')}
              className={`flex items-center gap-2 pb-3 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                activeGuide === 'setup' 
                  ? 'border-orange-500 text-orange-600' 
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              <PlusCircle size={16} />
              Guía de Instalación (Desde Cero)
            </button>
            <button
              onClick={() => switchGuide('config')}
              className={`flex items-center gap-2 pb-3 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                activeGuide === 'config' 
                  ? 'border-orange-500 text-orange-600' 
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              <FileSearch size={16} />
              Revisión de Configuración
            </button>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="w-full h-1 bg-slate-100 shrink-0">
          <div 
            className="h-full bg-orange-500 transition-all duration-300 ease-out"
            style={{ width: `${((currentStep + 1) / currentSteps.length) * 100}%` }}
          ></div>
        </div>

        {/* Content Area */}
        <div className="p-6 flex-1 overflow-y-auto bg-slate-50/50 flex flex-col">
          <div className="mb-4 text-center md:text-left shrink-0">
            <div className="flex justify-between items-start">
              <div>
                <div className="inline-block px-2 py-0.5 rounded bg-orange-100 text-orange-700 text-[10px] font-bold uppercase tracking-wider mb-2">
                  Paso {currentStep + 1} de {currentSteps.length}
                </div>
                <h3 className="text-xl md:text-2xl font-bold text-slate-900 mb-1">{stepData.title}</h3>
                <p className="text-slate-600 text-sm">{stepData.desc}</p>
              </div>
            </div>
          </div>

          <div className="flex-1 min-h-0 flex flex-col justify-center">
             <StepImage imgId={stepData.imgId} alt={stepData.title} isFullscreen={isFullscreen} />
          </div>
        </div>

        {/* Footer Actions */}
        <div className="p-4 border-t border-slate-200 bg-white flex justify-between items-center shrink-0">
          <button 
            onClick={prevStep}
            disabled={currentStep === 0}
            className={`flex items-center gap-1 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors ${
              currentStep === 0 ? 'text-slate-300 cursor-not-allowed' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
            }`}
          >
            <ChevronLeft size={18} /> Anterior
          </button>

          <span className="text-xs text-slate-400 hidden sm:block">
            {currentStep + 1} / {currentSteps.length}
          </span>

          <button 
            onClick={nextStep}
            className="flex items-center gap-1.5 px-6 py-2.5 bg-orange-600 text-white rounded-lg text-sm font-bold hover:bg-orange-700 shadow-md shadow-orange-200 hover:shadow-lg hover:shadow-orange-200/50 transition-all transform hover:-translate-y-0.5"
          >
            {currentStep === currentSteps.length - 1 ? 'Finalizar' : 'Siguiente'} 
            {currentStep !== currentSteps.length - 1 && <ChevronRight size={18} />}
          </button>
        </div>
      </div>
    </div>
  );
};