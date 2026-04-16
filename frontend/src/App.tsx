import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import PatientProfile from './pages/PatientProfile';
import './globals.css';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
    },
  },
});

// Error Boundary component
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-[#0a0a0b]">
          <div className="text-center max-w-sm mx-auto p-6">
            <div className="w-12 h-12 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center justify-center mx-auto mb-4">
              <span className="text-red-400 text-xl">!</span>
            </div>
            <h1 className="text-lg font-semibold text-white mb-2">Something went wrong</h1>
            <p className="text-sm text-zinc-500 mb-5">An unexpected error occurred. Please reload the page.</p>
            <button
              onClick={() => window.location.reload()}
              className="h-9 px-4 rounded-xl bg-white text-zinc-950 text-sm font-semibold hover:bg-zinc-100 transition-all"
            >
              Reload
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Protected Route component
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, isLoading } = useAuth();
  
  console.log('ProtectedRoute render:', { user, isLoading });
  
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0a0a0b]">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-zinc-700 border-t-white rounded-full animate-spin mx-auto mb-4" />
          <p className="text-sm text-zinc-500">Loading...</p>
        </div>
      </div>
    );
  }
  
  if (!user) {
    console.log('ProtectedRoute: No user, redirecting to login');
    return <Navigate to="/login" replace />;
  }
  
  console.log('ProtectedRoute: User authenticated, rendering children');
  return <>{children}</>;
};

// Public Route component (redirect to dashboard if authenticated)
const PublicRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, isLoading } = useAuth();
  
  console.log('PublicRoute render:', { user, isLoading });
  
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0a0a0b]">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-zinc-700 border-t-white rounded-full animate-spin mx-auto mb-4" />
          <p className="text-sm text-zinc-500">Loading...</p>
        </div>
      </div>
    );
  }
  
  if (user) {
    console.log('PublicRoute: User authenticated, redirecting to dashboard');
    return <Navigate to="/dashboard" replace />;
  }
  
  console.log('PublicRoute: No user, rendering children');
  return <>{children}</>;
};

function AppRoutes() {
  return (
    <Routes>
      <Route 
        path="/login" 
        element={
          <PublicRoute>
            <Login />
          </PublicRoute>
        } 
      />
      <Route 
        path="/register" 
        element={
          <PublicRoute>
            <Register />
          </PublicRoute>
        } 
      />
      <Route 
        path="/dashboard" 
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        } 
      />
      <Route 
        path="/patient-profile" 
        element={
          <ProtectedRoute>
            <PatientProfile />
          </ProtectedRoute>
        } 
      />
      <Route path="/" element={<Navigate to="/dashboard" />} />
    </Routes>
  );
}

function App() {
  return (
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <Router>
            <div className="App">
              <ErrorBoundary>
                <AppRoutes />
              </ErrorBoundary>
            </div>
          </Router>
        </AuthProvider>
      </QueryClientProvider>
    </ThemeProvider>
  );
}

export default App;