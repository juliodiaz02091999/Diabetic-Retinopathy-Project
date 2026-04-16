import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { ThemeToggle } from '@/components/ui/ThemeToggle';
import { UserPlus, AlertCircle, Loader2, CheckCircle, Zap, Shield, Sparkles } from 'lucide-react';

const Register = () => {
  const [formData, setFormData] = useState({ name: '', username: '', email: '', password: '' });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const { register } = useAuth();
  const navigate = useNavigate();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    try {
      await register(formData);
      setSuccess(true);
      setTimeout(() => navigate('/login'), 2000);
    } catch (err: any) {
      setError(err.message || 'Registration failed');
    } finally {
      setIsLoading(false);
    }
  };

  if (success) {
    return (
      <div className="page flex items-center justify-center p-6 relative overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] opacity-0 dark:opacity-100 bg-emerald-500/[0.05] rounded-full blur-[120px] pointer-events-none" />
        <div className="w-full max-w-sm relative z-10">
          <div className="surface p-8 text-center">
            <div className="w-14 h-14 rounded-2xl mx-auto mb-4 flex items-center justify-center
              bg-emerald-50 border-emerald-200 border
              dark:bg-emerald-500/10 dark:border-emerald-500/20">
              <CheckCircle className="h-7 w-7 text-emerald-600 dark:text-emerald-400" />
            </div>
            <h2 className="text-base font-semibold text-foreground mb-2">Account Created</h2>
            <p className="text-sm text-muted-foreground">Redirecting to sign in...</p>
            <div className="w-7 h-7 border-2 border-border border-t-foreground rounded-full animate-spin mx-auto mt-5" />
          </div>
        </div>
      </div>
    );
  }

  const fields = [
    { id: 'name', label: 'Full Name', type: 'text', placeholder: 'Dr. John Smith' },
    { id: 'username', label: 'Username', type: 'text', placeholder: 'drjohnsmith' },
    { id: 'email', label: 'Email', type: 'email', placeholder: 'you@hospital.com' },
    { id: 'password', label: 'Password', type: 'password', placeholder: '••••••••' },
  ];

  return (
    <div className="page flex items-center justify-center p-6 relative overflow-hidden">
      {/* Ambient glow – dark only */}
      <div className="absolute -top-40 left-1/2 -translate-x-1/2 w-[600px] h-[400px] opacity-0 dark:opacity-100 bg-blue-600/[0.07] rounded-full blur-[120px] pointer-events-none transition-opacity duration-500" />
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[800px] h-[300px] opacity-0 dark:opacity-100 bg-indigo-500/[0.05] rounded-full blur-[100px] pointer-events-none transition-opacity duration-500" />

      <div className="absolute top-4 right-4 z-20">
        <ThemeToggle labeled />
      </div>

      <div className="w-full max-w-sm relative z-10">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-[var(--radius)] surface mb-5">
            <UserPlus className="h-7 w-7 text-foreground" />
          </div>
          <h1 className="text-2xl font-bold text-foreground tracking-[0.15em] uppercase mb-1">RetinaScan AI</h1>
          <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">Create your professional account</p>
        </div>

        {/* Card */}
        <div className="surface p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="alert-error">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {fields.map(({ id, label, type, placeholder }) => (
              <div key={id} className="space-y-1.5">
                <label htmlFor={id} className="field-label">{label}</label>
                <input
                  id={id}
                  name={id}
                  type={type}
                  placeholder={placeholder}
                  value={formData[id as keyof typeof formData]}
                  onChange={handleChange}
                  required
                  className="field-input"
                />
              </div>
            ))}

            <button
              type="submit"
              disabled={isLoading}
              className="btn-primary w-full h-10 mt-1"
            >
              {isLoading ? (
                <><Loader2 className="h-4 w-4 animate-spin" />Creating account...</>
              ) : (
                'Create account'
              )}
            </button>
          </form>

          <p className="mt-5 text-center text-sm text-muted-foreground">
            Already have an account?{' '}
            <Link to="/login" className="text-foreground font-medium hover:opacity-70 transition-opacity">
              Sign in
            </Link>
          </p>
        </div>

        {/* Feature pills */}
        <div className="flex items-center justify-center gap-3 mt-5">
          {[
            { icon: Zap, label: 'AI-Powered' },
            { icon: Shield, label: 'HIPAA' },
            { icon: Sparkles, label: 'Clinical' },
          ].map(({ icon: Icon, label }) => (
            <div key={label} className="pill">
              <Icon className="h-3 w-3" />
              {label}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Register;
