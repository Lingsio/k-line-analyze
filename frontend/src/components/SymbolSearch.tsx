import { useState, useRef, useEffect } from 'react';
import { Search, X } from 'lucide-react';

interface SymbolSearchProps {
  value: string;
  market: string;
  onSymbolChange: (symbol: string) => void;
  onMarketChange: (market: string) => void;
}

const MARKETS = [
  { value: 'us', label: 'US', flag: '🇺🇸' },
  { value: 'tw', label: 'TW', flag: '🇹🇼' },
  { value: 'cn', label: 'CN', flag: '🇨🇳' },
  { value: 'hk', label: 'HK', flag: '🇭🇰' },
  { value: 'crypto', label: 'Crypto', flag: '₿' },
];

const POPULAR_SYMBOLS: Record<string, string[]> = {
  us: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD'],
  tw: ['2330', '2317', '2454', '2412', '2308', '2881', '2882', '2891'],
  cn: ['600519', '000858', '601318', '000001', '600036', '000333'],
  hk: ['0700', '9988', '0005', '1299', '0941', '2318'],
  crypto: ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA'],
};

export function SymbolSearch({
  value,
  market,
  onSymbolChange,
  onMarketChange,
}: SymbolSearchProps) {
  const [isFocused, setIsFocused] = useState(false);
  const [inputValue, setInputValue] = useState(value);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setInputValue(value);
  }, [value]);

  const handleSubmit = () => {
    const trimmed = inputValue.trim().toUpperCase();
    if (trimmed) {
      onSymbolChange(trimmed);
    }
    setIsFocused(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSubmit();
    } else if (e.key === 'Escape') {
      setIsFocused(false);
      setInputValue(value);
    }
  };

  const handleSuggestionClick = (symbol: string) => {
    setInputValue(symbol);
    onSymbolChange(symbol);
    setIsFocused(false);
  };

  const popularSymbols = POPULAR_SYMBOLS[market] || POPULAR_SYMBOLS.us;

  return (
    <div className="flex items-center gap-2">
      {/* Market selector */}
      <div className="flex bg-surface border border-border rounded-lg overflow-hidden">
        {MARKETS.map((m) => (
          <button
            key={m.value}
            onClick={() => onMarketChange(m.value)}
            className={`px-3 py-2 text-sm transition-colors ${
              market === m.value
                ? 'bg-accent text-background'
                : 'hover:bg-border text-secondary'
            }`}
          >
            <span className="mr-1">{m.flag}</span>
            {m.label}
          </button>
        ))}
      </div>

      {/* Symbol input */}
      <div className="relative flex-1 max-w-xs">
        <div className="relative">
          <Search
            size={18}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-secondary"
          />
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value.toUpperCase())}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setTimeout(() => setIsFocused(false), 200)}
            onKeyDown={handleKeyDown}
            placeholder="Enter symbol..."
            className="w-full bg-surface border border-border rounded-lg pl-10 pr-10 py-2 text-sm focus:outline-none focus:border-accent"
          />
          {inputValue && (
            <button
              onClick={() => {
                setInputValue('');
                inputRef.current?.focus();
              }}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-secondary hover:text-primary"
            >
              <X size={16} />
            </button>
          )}
        </div>

        {/* Suggestions dropdown */}
        {isFocused && (
          <div className="absolute top-full left-0 right-0 mt-1 bg-surface border border-border rounded-lg shadow-lg z-50 overflow-hidden">
            <div className="p-2 text-xs text-secondary border-b border-border">
              Popular Symbols
            </div>
            <div className="flex flex-wrap gap-1 p-2">
              {popularSymbols.map((symbol) => (
                <button
                  key={symbol}
                  onClick={() => handleSuggestionClick(symbol)}
                  className="px-2 py-1 text-sm bg-background rounded hover:bg-accent hover:text-background transition-colors"
                >
                  {symbol}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Search button */}
      <button
        onClick={handleSubmit}
        className="px-4 py-2 bg-accent text-background rounded-lg hover:bg-accent/90 transition-colors"
      >
        Load
      </button>
    </div>
  );
}

export default SymbolSearch;
