import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ThemeToggle } from '@/components/ui/ThemeToggle';
import { supabasePatientAPI, supabasePredictionAPI } from '@/lib/supabaseApi';
import {
  User, UserPlus, Calendar, Download, AlertCircle, CheckCircle,
  Loader2, ArrowLeft, Eye, Activity, FileText, Clock, BarChart3, TrendingUp
} from 'lucide-react';

interface Patient { name: string; age: number; gender: string; contact: string; }
interface Prediction {
  patient_name: string; patient_id: number; prediction_class: string;
  confidence_score: number; prediction_date: string;
}

const PatientProfile = () => {
  const { } = useAuth();
  const navigate = useNavigate();
  const [patient, setPatient] = useState<Patient | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [showAddForm, setShowAddForm] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [formData, setFormData] = useState({ name: '', age: '', gender: 'Male', contact_info: '' });

  useEffect(() => { loadPatientData(); }, []);

  const loadPatientData = async () => {
    setLoading(true);
    try {
      const patientData = await supabasePatientAPI.getMyPatient();
      setPatient(patientData);
      if (patientData) {
        const predictionsData = await supabasePredictionAPI.getAll();
        setPredictions(predictionsData);
      }
    } catch (error) {
      console.error('Failed to load patient data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSavePatient = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      await supabasePatientAPI.create({
        name: formData.name,
        age: parseInt(formData.age),
        gender: formData.gender,
        contact_info: formData.contact_info,
      });
      await loadPatientData();
      setShowAddForm(false);
      setFormData({ name: '', age: '', gender: 'Male', contact_info: '' });
    } catch (error) {
      console.error('Failed to save patient:', error);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="page flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 rounded-2xl surface flex items-center justify-center mx-auto mb-4">
            <Loader2 className="h-6 w-6 text-foreground animate-spin" />
          </div>
          <p className="text-sm text-muted-foreground">Loading patient data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      {/* Subtle top glow – dark only */}
      <div className="fixed top-0 left-1/2 -translate-x-1/2 w-[800px] h-[200px] opacity-0 dark:opacity-100 bg-blue-600/[0.04] rounded-full blur-[120px] pointer-events-none transition-opacity duration-500" />

      <div className="max-w-5xl mx-auto px-8 py-8 relative z-10">
        {/* Header row */}
        <div className="flex items-start justify-between mb-8">
          <div>
            <button
              onClick={() => navigate('/dashboard')}
              className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-5 group"
            >
              <ArrowLeft className="h-4 w-4 group-hover:-translate-x-0.5 transition-transform" />
              Back to Dashboard
            </button>
            <div className="flex items-center gap-4">
              <div className="w-11 h-11 rounded-2xl surface flex items-center justify-center">
                <User className="h-5 w-5 text-foreground" />
              </div>
              <div>
                <h1 className="text-2xl font-semibold text-foreground tracking-tight">Patient Profile</h1>
                <p className="text-sm text-muted-foreground">Manage patient information and prediction history</p>
              </div>
            </div>
          </div>
          <ThemeToggle />
        </div>

        {patient ? (
          <>
            {/* Patient details + stats */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-5">
              <div className="lg:col-span-2 surface-lift p-6">
                <div className="flex items-center gap-2 mb-5">
                  <User className="h-4 w-4 text-muted-foreground" />
                  <p className="text-sm font-semibold text-foreground">Patient Information</p>
                </div>
                <div className="grid grid-cols-2 gap-5">
                  {[
                    { label: 'Name', value: patient.name },
                    { label: 'Age', value: `${patient.age} years` },
                    { label: 'Gender', value: patient.gender },
                    { label: 'Contact', value: patient.contact },
                  ].map(({ label, value }) => (
                    <div key={label}>
                      <p className="field-label mb-1">{label}</p>
                      <p className="text-lg font-semibold text-foreground">{value}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-4">
                <div className="surface p-5 text-center">
                  <div className="w-10 h-10 rounded-xl mx-auto mb-3 flex items-center justify-center
                    bg-blue-50 border border-blue-200 dark:bg-blue-500/10 dark:border-blue-500/20">
                    <Activity className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <p className="text-3xl font-bold text-foreground mb-0.5">{predictions.length}</p>
                  <p className="text-xs text-muted-foreground font-medium">Total Predictions</p>
                </div>
                <div className="surface p-5 text-center">
                  <div className="w-10 h-10 rounded-xl mx-auto mb-3 flex items-center justify-center
                    bg-emerald-50 border border-emerald-200 dark:bg-emerald-500/10 dark:border-emerald-500/20">
                    <TrendingUp className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                  </div>
                  <p className="text-3xl font-bold text-foreground mb-0.5">
                    {predictions.length > 0
                      ? Math.round(predictions.reduce((acc, p) => acc + p.confidence_score, 0) / predictions.length)
                      : 0}%
                  </p>
                  <p className="text-xs text-muted-foreground font-medium">Avg Confidence</p>
                </div>
              </div>
            </div>
          </>
        ) : (
          /* No patient */
          <div className="surface p-6 mb-5">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-9 h-9 rounded-xl flex items-center justify-center
                bg-amber-50 border border-amber-200 dark:bg-amber-500/10 dark:border-amber-500/20">
                <AlertCircle className="h-4 w-4 text-amber-600 dark:text-amber-400" />
              </div>
              <div>
                <p className="text-sm font-semibold text-foreground">No Patient Profile</p>
                <p className="text-xs text-muted-foreground">Create a profile to save and track predictions</p>
              </div>
            </div>

            {!showAddForm ? (
              <button onClick={() => setShowAddForm(true)} className="btn-primary h-9 px-4">
                <UserPlus className="h-4 w-4" />
                Add Patient Profile
              </button>
            ) : (
              <form onSubmit={handleSavePatient} className="space-y-4 mt-2">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="field-label">Name *</label>
                    <input value={formData.name} onChange={(e) => setFormData({ ...formData, name: e.target.value })} placeholder="Patient name" required className="field-input" />
                  </div>
                  <div className="space-y-1.5">
                    <label className="field-label">Age *</label>
                    <input type="number" value={formData.age} onChange={(e) => setFormData({ ...formData, age: e.target.value })} placeholder="Age" required className="field-input" />
                  </div>
                  <div className="space-y-1.5">
                    <label className="field-label">Gender</label>
                    <Select value={formData.gender} onValueChange={(v) => setFormData({ ...formData, gender: v })}>
                      <SelectTrigger className="field-input h-10">
                        <SelectValue placeholder="Select gender" />
                      </SelectTrigger>
                      <SelectContent className="bg-card border-border text-foreground">
                        <SelectItem value="Male">Male</SelectItem>
                        <SelectItem value="Female">Female</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1.5">
                    <label className="field-label">Contact Info</label>
                    <input value={formData.contact_info} onChange={(e) => setFormData({ ...formData, contact_info: e.target.value })} placeholder="Phone or email" className="field-input" />
                  </div>
                </div>
                <div className="flex gap-3 pt-1">
                  <button type="submit" disabled={saving} className="btn-primary h-9 px-4">
                    {saving ? <><Loader2 className="h-4 w-4 animate-spin" />Saving...</> : <><CheckCircle className="h-4 w-4" />Save Patient</>}
                  </button>
                  <button type="button" onClick={() => setShowAddForm(false)} className="btn-secondary h-9 px-4">
                    Cancel
                  </button>
                </div>
              </form>
            )}
          </div>
        )}

        {/* Prediction History */}
        {patient && (
          <div className="surface overflow-hidden">
            <div className="px-6 py-5 border-b border-border flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg surface flex items-center justify-center">
                  <Calendar className="h-4 w-4 text-muted-foreground" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-foreground">Prediction History</p>
                  <p className="text-xs text-muted-foreground">{predictions.length} prediction(s) recorded</p>
                </div>
              </div>
              {predictions.length > 0 && (
                <button onClick={() => alert('Report download feature is under development')} className="btn-secondary h-8 px-3 text-xs">
                  <Download className="h-3.5 w-3.5" />
                  Download Report
                </button>
              )}
            </div>

            <div className="p-4">
              {predictions.length > 0 ? (
                <div className="space-y-3">
                  {predictions.map((prediction, index) => {
                    const isDR = prediction.prediction_class === 'DR';
                    return (
                      <div key={index} className="flex items-center justify-between px-4 py-4 rounded-xl surface-lift cursor-default group">
                        <div className="flex items-center gap-4">
                          <div className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 border ${
                            isDR
                              ? 'bg-red-50 border-red-200 dark:bg-red-500/10 dark:border-red-500/20'
                              : 'bg-emerald-50 border-emerald-200 dark:bg-emerald-500/10 dark:border-emerald-500/20'
                          }`}>
                            <Eye className={`h-4 w-4 ${isDR ? 'text-red-600 dark:text-red-400' : 'text-emerald-600 dark:text-emerald-400'}`} />
                          </div>
                          <div>
                            <div className="flex items-center gap-2 mb-1">
                              <span className={isDR ? 'badge-dr' : 'badge-no-dr'}>
                                {isDR ? 'DR Detected' : 'No DR'}
                              </span>
                              <div className="flex items-center gap-1 text-muted-foreground text-xs">
                                <Clock className="h-3 w-3" />
                                {new Date(prediction.prediction_date).toLocaleDateString()}
                              </div>
                            </div>
                            <p className="text-xs text-muted-foreground">
                              Confidence: <span className="text-foreground font-semibold">{prediction.confidence_score.toFixed(1)}%</span>
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="text-right">
                            <p className="text-xl font-bold text-foreground">{prediction.confidence_score.toFixed(1)}%</p>
                            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Confidence</p>
                          </div>
                          <BarChart3 className="h-4 w-4 text-muted-foreground/40 group-hover:text-muted-foreground transition-colors" />
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-14">
                  <div className="w-14 h-14 rounded-2xl surface flex items-center justify-center mx-auto mb-4">
                    <FileText className="h-7 w-7 text-muted-foreground" />
                  </div>
                  <p className="text-sm font-medium text-muted-foreground mb-1">No predictions recorded yet</p>
                  <p className="text-xs text-muted-foreground/60">Start analyzing images to build prediction history</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PatientProfile;
