import { Sun, Moon } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';

interface ThemeToggleProps {
  className?: string;
  /** When true, shows a text label in light mode (Lumon style) */
  labeled?: boolean;
}

export const ThemeToggle = ({ className = '', labeled = false }: ThemeToggleProps) => {
  const { theme, toggle } = useTheme();

  if (labeled) {
    return (
      <button
        onClick={toggle}
        aria-label="Toggle theme"
        className={`inline-flex items-center gap-2 px-3 py-1.5 border border-border
          bg-card text-[10px] font-bold uppercase tracking-[0.15em] text-muted-foreground
          hover:text-foreground hover:border-foreground/30 transition-all duration-150
          rounded-[var(--radius)] ${className}`}
      >
        {theme === 'dark'
          ? <><Sun className="h-3 w-3" />Light Mode</>
          : <><Moon className="h-3 w-3" />Dark Mode</>}
      </button>
    );
  }

  return (
    <button
      onClick={toggle}
      aria-label="Toggle theme"
      className={`w-8 h-8 rounded-[calc(var(--radius)-2px)] border border-border
        flex items-center justify-center transition-all duration-150
        bg-card text-muted-foreground hover:text-foreground hover:border-foreground/30
        ${className}`}
    >
      {theme === 'dark' ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
    </button>
  );
};
