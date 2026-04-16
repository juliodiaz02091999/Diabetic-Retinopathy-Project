import React, { useState, useRef } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { predictionAPI } from '@/lib/api';
import { supabasePredictionAPI } from '@/lib/supabaseApi';
import { ThemeToggle } from '@/components/ui/ThemeToggle';
import {
  Eye, Upload, FileText, AlertTriangle, CheckCircle, Loader2,
  LogOut, UserPlus, Info, Zap, Target, Microscope,
  Activity, TrendingUp, Sparkles, Shield, X
} from 'lucide-react';

interface RETFoundPredictionResult {
  confidence_score: number;
  prediction_class: string;
  diagnosis: string;
  probabilities: Record<string, number>;
  individual_prediction: string;
  individual_confidence: number;
  individual_diagnosis: string;
  binary_prediction: string;
  binary_confidence: number;
  binary_diagnosis: string;
  clinical_recommendation: string;
  detailed_class: string;
  detailed_probabilities: Record<string, number>;
  model_used: string;
  checkpoint_loaded: boolean;
}

const Dashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<RETFoundPredictionResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const processFile = (file: File) => {
    if (!file.type.startsWith('image/')) { alert('Please select an image file'); return; }
    setSelectedFile(file);
    setPrediction(null);
    setSaveMessage(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(file);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    try {
      const response = await predictionAPI.predictRETFound(selectedFile);
      setPrediction(response.data);
      try {
        await supabasePredictionAPI.save({
          prediction_class: response.data.prediction_class,
          confidence_score: response.data.confidence_score,
        });
        setSaveMessage('Prediction saved to patient history');
      } catch {
        setSaveMessage('Create a patient profile to save predictions');
      }
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const isRETFound = (p: any): p is RETFoundPredictionResult => p !== null && 'detailed_class' in p;
  const isDR = prediction?.prediction_class === 'DR';

  return (
    <div className="min-h-screen bg-background flex">

      {/* ── Sidebar ── */}
      <aside className="w-64 sidebar flex flex-col fixed left-0 top-0 h-full z-40">
        {/* Brand */}
        <div className="px-5 py-5 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl surface flex items-center justify-center shrink-0">
              <Eye className="h-4 w-4 text-foreground" />
            </div>
            <div>
              <p className="text-sm font-semibold text-foreground leading-tight">RetinaScan AI</p>
              <p className="text-[11px] text-muted-foreground leading-tight">Professional Platform</p>
            </div>
          </div>
        </div>

        {/* User */}
        <div className="px-5 py-4 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg surface flex items-center justify-center text-xs font-bold text-foreground shrink-0">
              {user?.username?.[0]?.toUpperCase()}
            </div>
            <div className="min-w-0">
              <p className="text-sm font-medium text-foreground truncate">{user?.username}</p>
              <p className="text-[11px] text-muted-foreground">Medical Professional</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          <div className="nav-item-active">
            <Zap className="h-4 w-4 text-foreground shrink-0" />
            <div>
              <p className="text-sm font-medium text-foreground">AI Analysis</p>
              <p className="text-[11px] text-muted-foreground">Active</p>
            </div>
          </div>

          <button onClick={() => navigate('/patient-profile')} className="nav-item group">
            <UserPlus className="h-4 w-4 text-muted-foreground group-hover:text-foreground shrink-0 transition-colors" />
            <div>
              <p className="text-sm font-medium text-muted-foreground group-hover:text-foreground transition-colors">Patient Profile</p>
              <p className="text-[11px] text-muted-foreground/60">Manage patient data</p>
            </div>
          </button>

          <div className="nav-item opacity-40 cursor-default">
            <FileText className="h-4 w-4 text-muted-foreground shrink-0" />
            <div>
              <p className="text-sm font-medium text-muted-foreground">Clinical Reports</p>
              <p className="text-[11px] text-muted-foreground/60">Coming soon</p>
            </div>
          </div>
        </nav>

        {/* Footer */}
        <div className="px-3 py-4 border-t border-border space-y-1">
          <div className="px-1 pb-1">
            <ThemeToggle labeled className="w-full justify-center" />
          </div>
          <button
            onClick={logout}
            className="nav-item group text-muted-foreground hover:text-red-600 dark:hover:text-red-400"
          >
            <LogOut className="h-4 w-4 shrink-0 transition-colors" />
            <span className="text-sm font-medium transition-colors">Sign out</span>
          </button>
        </div>
      </aside>

      {/* ── Main ── */}
      <main className={`flex-1 ml-64 transition-all duration-500 ease-out ${prediction ? 'mr-[26rem]' : 'mr-0'}`}>
        <div className="p-8 max-w-3xl mx-auto">
          {/* Welcome */}
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-foreground tracking-[0.08em] uppercase mb-1">
              {user?.username}
            </h2>
            <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">Ready to analyze retinal images with advanced AI</p>
          </div>

          {/* Stat pills */}
          <div className="flex items-center gap-3 mb-8 flex-wrap">
            {[
              { icon: Activity, label: '98.7% Accuracy', color: 'text-blue-500 dark:text-blue-400' },
              { icon: Sparkles, label: 'RETFound Model', color: 'text-purple-500 dark:text-purple-400' },
              { icon: Shield, label: 'HIPAA Compliant', color: 'text-emerald-600 dark:text-emerald-400' },
              { icon: TrendingUp, label: 'Real-time', color: 'text-orange-500 dark:text-orange-400' },
            ].map(({ icon: Icon, label, color }) => (
              <div key={label} className="pill transition-all duration-200 hover:shadow-elev-sm hover:-translate-y-px">
                <Icon className={`h-3 w-3 ${color}`} />
                {label}
              </div>
            ))}
          </div>

          {/* Upload card */}
          <div className="surface-lift overflow-hidden">
            {/* Card header */}
            <div className="px-6 py-5 border-b border-border flex items-center gap-4">
              <div className="w-9 h-9 rounded-xl surface flex items-center justify-center shrink-0">
                <Upload className="h-4 w-4 text-foreground" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-foreground">RETFound AI Analysis</h3>
                <p className="text-xs text-muted-foreground">Foundation model for diabetic retinopathy detection</p>
              </div>
              <div className="ml-auto pill">
                <Zap className="h-3 w-3 text-blue-500 dark:text-blue-400" />
                RETFound · Quantized
              </div>
            </div>

            <div className="p-6 space-y-5">
              <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileSelect} className="hidden" />

              {/* Drop zone */}
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); setIsDragOver(true); }}
                onDragLeave={(e) => { e.preventDefault(); e.stopPropagation(); setIsDragOver(false); }}
                onDrop={(e) => {
                  e.preventDefault(); e.stopPropagation(); setIsDragOver(false);
                  const file = e.dataTransfer.files?.[0];
                  if (file) processFile(file);
                }}
                className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all duration-200 ${
                  isDragOver
                    ? 'border-foreground/30 bg-muted/60 scale-[1.01]'
                    : 'border-border hover:border-foreground/20 hover:bg-muted/30'
                }`}
              >
                {preview ? (
                  <div className="space-y-4">
                    <img src={preview} alt="Preview" className="max-h-56 mx-auto rounded-xl shadow-lg ring-1 ring-border" />
                    <p className="text-xs text-muted-foreground font-medium">{selectedFile?.name}</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className={`w-14 h-14 rounded-2xl mx-auto flex items-center justify-center surface transition-all duration-200 ${isDragOver ? 'scale-110' : ''}`}>
                      <Upload className="h-7 w-7 text-muted-foreground" />
                    </div>
                    <div>
                      <p className="text-base font-semibold text-foreground mb-1">
                        {isDragOver ? 'Drop image here' : 'Upload retinal image'}
                      </p>
                      <p className="text-sm text-muted-foreground">Drag & drop or click to browse · JPG, PNG up to 10 MB</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Analyze button */}
              <button
                onClick={handleAnalyze}
                disabled={!selectedFile || isAnalyzing}
                className="btn-primary w-full h-11"
              >
                {isAnalyzing ? (
                  <><Loader2 className="h-4 w-4 animate-spin" />Analyzing with AI...</>
                ) : (
                  <><Zap className="h-4 w-4" />Start RETFound Analysis</>
                )}
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* ── Results Panel ── */}
      <aside className={`fixed right-0 top-0 h-full w-[26rem] sidebar flex flex-col z-50 transform transition-transform duration-500 ease-out ${
        prediction ? 'translate-x-0' : 'translate-x-full'
      }`}>
        {/* Header */}
        <div className="px-5 py-5 border-b border-border flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg surface flex items-center justify-center">
              <FileText className="h-4 w-4 text-foreground" />
            </div>
            <div>
              <p className="text-sm font-semibold text-foreground">Analysis Results</p>
              <p className="text-[11px] text-muted-foreground">AI Diagnostic Report</p>
            </div>
          </div>
          <button
            onClick={() => setPrediction(null)}
            className="w-8 h-8 rounded-lg hover:bg-muted dark:hover:bg-white/[0.07] flex items-center justify-center transition-colors"
          >
            <X className="h-4 w-4 text-muted-foreground" />
          </button>
        </div>

        {/* Scrollable */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          {prediction && (
            <>
              {/* Main diagnosis */}
              <div className={`p-4 rounded-xl border flex items-start gap-4 ${
                isDR
                  ? 'bg-red-50 border-red-200 dark:bg-red-500/[0.07] dark:border-red-500/20'
                  : 'bg-emerald-50 border-emerald-200 dark:bg-emerald-500/[0.07] dark:border-emerald-500/20'
              }`}>
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 border ${
                  isDR
                    ? 'bg-red-100 border-red-200 dark:bg-red-500/15 dark:border-red-500/20'
                    : 'bg-emerald-100 border-emerald-200 dark:bg-emerald-500/15 dark:border-emerald-500/20'
                }`}>
                  {isDR
                    ? <AlertTriangle className="h-5 w-5 text-red-600 dark:text-red-400" />
                    : <CheckCircle className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <p className="text-sm font-semibold text-foreground">{prediction.diagnosis}</p>
                    <span className={isDR ? 'badge-dr' : 'badge-no-dr'}>{prediction.prediction_class}</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {isDR ? 'Recommend ophthalmologist consultation' : 'No signs of diabetic retinopathy detected'}
                  </p>
                </div>
              </div>

              {/* Confidence + Model */}
              <div className="grid grid-cols-2 gap-3">
                <div className="surface p-4 text-center">
                  <p className="text-2xl font-bold text-foreground mb-1">{prediction.confidence_score.toFixed(1)}%</p>
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-semibold">Confidence</p>
                </div>
                <div className="surface p-4 text-center">
                  <p className="text-sm font-bold text-foreground mb-1 truncate">{prediction.model_used || 'Current Model'}</p>
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-semibold">Model</p>
                </div>
              </div>

              {/* Clinical analysis */}
              {isRETFound(prediction) && (
                <>
                  <div className="surface p-4">
                    <div className="flex items-center gap-2 mb-4">
                      <Microscope className="h-4 w-4 text-muted-foreground" />
                      <p className="text-sm font-semibold text-foreground">Clinical Analysis</p>
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                      {[
                        { value: `${prediction.individual_confidence.toFixed(1)}%`, label: 'Individual', badge: prediction.individual_prediction },
                        { value: `${prediction.binary_confidence.toFixed(1)}%`, label: 'Screening', badge: prediction.binary_prediction },
                        { value: `${prediction.clinical_recommendation.split(' | ').length}`, label: 'Actions', badge: 'View' },
                      ].map(({ value, label, badge }) => (
                        <div key={label} className="surface-muted p-3 text-center rounded-xl">
                          <p className="text-lg font-bold text-foreground mb-0.5">{value}</p>
                          <p className="text-[10px] text-muted-foreground mb-2">{label}</p>
                          <span className="inline-block text-[10px] font-semibold px-2 py-0.5 rounded-full pill">{badge}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Stage probabilities */}
                  <div className="surface p-4">
                    <div className="flex items-center gap-2 mb-4">
                      <Target className="h-4 w-4 text-muted-foreground" />
                      <p className="text-sm font-semibold text-foreground">DR Stage Probabilities</p>
                    </div>
                    <div className="space-y-2.5">
                      {Object.entries(prediction.detailed_probabilities).map(([stage, prob]) => (
                        <div key={stage} className="flex items-center gap-3">
                          <p className="text-xs text-muted-foreground w-24 shrink-0 truncate">{stage.replace(' DR', '')}</p>
                          <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                            <div
                              className="h-full bg-foreground/40 rounded-full transition-all duration-700"
                              style={{ width: `${prob}%` }}
                            />
                          </div>
                          <p className="text-xs font-semibold text-foreground w-12 text-right">{prob.toFixed(1)}%</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}

              {/* Save message */}
              {saveMessage && (
                <div className={saveMessage.includes('saved') ? 'alert-success' : 'alert-info'}>
                  <Info className="h-4 w-4 shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p>{saveMessage}</p>
                    {saveMessage.includes('Create') && (
                      <button
                        onClick={() => navigate('/patient-profile')}
                        className="mt-1.5 text-xs font-semibold underline underline-offset-2 hover:opacity-70 transition-opacity"
                      >
                        Create profile →
                      </button>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </aside>
    </div>
  );
};

export default Dashboard;
