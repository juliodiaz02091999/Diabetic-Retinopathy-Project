import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { supabase, User } from '@/lib/supabase';
import { supabaseAuthAPI } from '@/lib/supabaseApi';

interface AuthContextType {
  user: User | null;
  login: (email: string, password: string) => Promise<void>;
  register: (userData: {
    username: string;
    name: string;
    password: string;
    email: string;
  }) => Promise<void>;
  logout: () => Promise<void>;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

const AUTH_LOADING_MAX_MS = 8000;

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;

    const safetyTimeout = setTimeout(() => {
      if (isMounted) setIsLoading(false);
    }, AUTH_LOADING_MAX_MS);

    const getCurrentSession = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();

        if (session?.user && isMounted) {
          const basicUser: User = {
            id: session.user.id,
            username: session.user.user_metadata?.username || session.user.email?.split('@')[0] || 'user',
            name: session.user.user_metadata?.name || session.user.email?.split('@')[0] || 'User',
            email: session.user.email || '',
            created_at: new Date().toISOString()
          };
          setUser(basicUser);
        }
      } catch (error) {
        console.error('Error getting current session:', error);
      } finally {
        if (isMounted) {
          setIsLoading(false);
          clearTimeout(safetyTimeout);
        }
      }
    };

    getCurrentSession();

    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        if (!isMounted) return;

        if (event === 'SIGNED_IN' && session?.user) {
          const basicUser: User = {
            id: session.user.id,
            username: session.user.user_metadata?.username || session.user.email?.split('@')[0] || 'user',
            name: session.user.user_metadata?.name || session.user.email?.split('@')[0] || 'User',
            email: session.user.email || '',
            created_at: new Date().toISOString()
          };
          setUser(basicUser);
        } else if (event === 'SIGNED_OUT') {
          setUser(null);
        }
      }
    );

    return () => {
      isMounted = false;
      clearTimeout(safetyTimeout);
      subscription.unsubscribe();
    };
  }, []);

  const login = async (email: string, password: string) => {
    try {
      await supabaseAuthAPI.login({ email, password });
    } catch (error) {
      throw error;
    }
  };

  const register = async (userData: {
    username: string;
    name: string;
    password: string;
    email: string;
  }) => {
    try {
      await supabaseAuthAPI.register(userData);
    } catch (error) {
      throw error;
    }
  };

  const logout = async () => {
    try {
      await supabaseAuthAPI.logout();
      setUser(null);
    } catch (error) {
      console.error('Error during logout:', error);
      setUser(null);
    }
  };

  const value = {
    user,
    login,
    register,
    logout,
    isLoading,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
