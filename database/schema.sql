-- Mutual Fund Metadata Table
CREATE TABLE IF NOT EXISTS mutual_funds (
    scheme_code TEXT PRIMARY KEY,
    scheme_name TEXT NOT NULL,
    fund_house TEXT,
    category TEXT,
    sub_category TEXT,
    launch_date DATE,
    nav_date DATE,
    current_nav REAL,
    aum REAL,
    expense_ratio REAL,
    min_investment REAL,
    exit_load TEXT,
    returns_1m REAL,
    returns_3m REAL,
    returns_6m REAL,
    returns_1y REAL,
    returns_3y REAL,
    returns_5y REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NAV History Table
CREATE TABLE IF NOT EXISTS nav_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scheme_code TEXT NOT NULL,
    date DATE NOT NULL,
    nav REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (scheme_code) REFERENCES mutual_funds(scheme_code),
    UNIQUE(scheme_code, date)
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_nav_history_scheme_date 
ON nav_history(scheme_code, date);

CREATE INDEX IF NOT EXISTS idx_mutual_funds_category 
ON mutual_funds(category);

CREATE INDEX IF NOT EXISTS idx_mutual_funds_name 
ON mutual_funds(scheme_name);

-- Fund Statistics Table
CREATE TABLE IF NOT EXISTS fund_statistics (
    scheme_code TEXT PRIMARY KEY,
    mean_daily_return REAL,
    std_daily_return REAL,
    annual_volatility REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    avg_30d_volume REAL,
    beta REAL,
    alpha REAL,
    calculated_date DATE,
    FOREIGN KEY (scheme_code) REFERENCES mutual_funds(scheme_code)
);

-- Correlation Matrix Table (for pre-calculated correlations)
CREATE TABLE IF NOT EXISTS fund_correlations (
    scheme_code_1 TEXT NOT NULL,
    scheme_code_2 TEXT NOT NULL,
    correlation REAL NOT NULL,
    period_days INTEGER NOT NULL,
    calculated_date DATE NOT NULL,
    PRIMARY KEY (scheme_code_1, scheme_code_2, period_days),
    FOREIGN KEY (scheme_code_1) REFERENCES mutual_funds(scheme_code),
    FOREIGN KEY (scheme_code_2) REFERENCES mutual_funds(scheme_code)
);
