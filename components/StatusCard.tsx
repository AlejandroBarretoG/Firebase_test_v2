import React from 'react';
import { CheckCircle2, XCircle, Loader2, CircleDashed } from 'lucide-react';

interface StatusCardProps {
  title: string;
  description: string;
  status: 'idle' | 'loading' | 'success' | 'error';
  details?: string;
}

export const StatusCard: React.FC<StatusCardProps> = ({ title, description, status, details }) => {
  const getIcon = () => {
    switch (status) {
      case 'loading':
        return <Loader2 className="w-6 h-6 text-blue-500 animate-spin" />;
      case 'success':
        return <CheckCircle2 className="w-6 h-6 text-green-500" />;
      case 'error':
        return <XCircle className="w-6 h-6 text-red-500" />;
      default:
        return <CircleDashed className="w-6 h-6 text-slate-400" />;
    }
  };

  const getBorderColor = () => {
     switch (status) {
      case 'success': return 'border-green-200 bg-green-50';
      case 'error': return 'border-red-200 bg-red-50';
      case 'loading': return 'border-blue-200 bg-blue-50';
      default: return 'border-slate-200 bg-white';
    }
  };

  return (
    <div className={`p-4 rounded-lg border ${getBorderColor()} transition-all duration-300`}>
      <div className="flex items-start gap-4">
        <div className="mt-1 flex-shrink-0">
          {getIcon()}
        </div>
        <div className="flex-1">
          <h3 className="font-semibold text-slate-900">{title}</h3>
          <p className="text-sm text-slate-600 mt-1">{description}</p>
          
          {details && (
            <div className="mt-3 p-2 bg-black/5 rounded text-xs font-mono text-slate-700 overflow-auto whitespace-pre-wrap break-all">
              {details}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};