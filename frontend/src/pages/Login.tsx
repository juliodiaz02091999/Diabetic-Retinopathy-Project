import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { ThemeToggle } from '@/components/ui/ThemeToggle';
import { Eye, AlertCircle, Loader2, Zap, Shield, Sparkles } from 'lucide-react';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    try {
      await login(email, password);
      navigate('/dashboard');
    } catch (err: any) {
      setError(err.message || 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page flex items-center justify-center p-6 relative overflow-hidden">
      {/* Ambient glow – only visible in dark mode */}
      <div className="absolute -top-40 left-1/2 -translate-x-1/2 w-[600px] h-[400px] opacity-0 dark:opacity-100 bg-blue-600/[0.07] rounded-full blur-[120px] pointer-events-none transition-opacity duration-500" />
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[800px] h-[300px] opacity-0 dark:opacity-100 bg-indigo-500/[0.05] rounded-full blur-[100px] pointer-events-none transition-opacity duration-500" />

      {/* Theme toggle – top right */}
      <div className="absolute top-4 right-4 z-20">
        <ThemeToggle labeled />
      </div>

      <div className="w-full max-w-sm relative z-10">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-[var(--radius)] surface mb-5">
            <Eye className="h-7 w-7 text-foreground" />
          </div>
          <h1 className="text-2xl font-bold text-foreground tracking-[0.15em] uppercase mb-1">RetinaScan AI</h1>
          <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">Advanced retinopathy screening</p>
        </div>

        {/* Card */}
        <div className="surface p-6">
          <div className="mb-5">
            <h2 className="text-base font-semibold text-foreground mb-1">Welcome back</h2>
            <p className="text-sm text-muted-foreground">Sign in to your account to continue</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="alert-error">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <div className="space-y-1.5">
              <label htmlFor="email" className="field-label">Email</label>
              <input
                id="email"
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="field-input"
              />
            </div>

            <div className="space-y-1.5">
              <label htmlFor="password" className="field-label">Password</label>
              <input
                id="password"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="field-input"
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="btn-primary w-full h-10 mt-1"
            >
              {isLoading ? (
                <><Loader2 className="h-4 w-4 animate-spin" />Signing in...</>
              ) : (
                'Sign in'
              )}
            </button>
          </form>

          <p className="mt-5 text-center text-sm text-muted-foreground">
            No account?{' '}
            <Link to="/register" className="text-foreground font-medium hover:opacity-70 transition-opacity">
              Create one
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

export default Login;
