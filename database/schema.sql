-- database/schema_multi_asset.sql

-- ==========================================
-- ASSET MASTER TABLE (Common attributes)
-- ==========================================
CREATE TABLE IF NOT EXISTS assets (
    isin TEXT PRIMARY KEY,
    asset_type TEXT NOT NULL CHECK(asset_type IN ('mutual_fund', 'stock', 'bond', 'etf')),
    symbol TEXT,
    name TEXT NOT NULL,
    exchange TEXT,
    currency TEXT DEFAULT 'INR',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_assets_type ON assets(asset_type);
CREATE INDEX IF NOT EXISTS idx_assets_symbol ON assets(symbol);
CREATE INDEX IF NOT EXISTS idx_assets_name ON assets(name);

-- ==========================================
-- MUTUAL FUNDS SPECIFIC
-- ==========================================
CREATE TABLE IF NOT EXISTS mutual_funds (
    isin TEXT PRIMARY KEY,
    scheme_code TEXT UNIQUE,
    scheme_name TEXT NOT NULL,
    amc_name TEXT,
    category TEXT,
    sub_category TEXT,
    plan_type TEXT,
    option_type TEXT,
    launch_date DATE,
    aum REAL,
    expense_ratio REAL,
    exit_load TEXT,
    min_investment REAL,
    min_sip_investment REAL,
    risk_level TEXT,
    benchmark TEXT,
    fund_manager TEXT,
    returns_1m REAL,
    returns_3m REAL,
    returns_6m REAL,
    returns_1y REAL,
    returns_3y REAL,
    returns_5y REAL,
    FOREIGN KEY (isin) REFERENCES assets(isin) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_mf_category ON mutual_funds(category);
CREATE INDEX IF NOT EXISTS idx_mf_amc ON mutual_funds(amc_name);

-- ==========================================
-- STOCKS SPECIFIC
-- ==========================================
CREATE TABLE IF NOT EXISTS stocks (
    isin TEXT PRIMARY KEY,
    symbol TEXT NOT NULL UNIQUE,
    company_name TEXT NOT NULL,
    exchange TEXT NOT NULL,
    series TEXT,
    sector TEXT,
    industry TEXT,
    market_cap REAL,
    face_value REAL,
    issued_capital REAL,
    listing_date DATE,
    paid_up_capital REAL,
    pe_ratio REAL,
    pb_ratio REAL,
    div_yield REAL,
    week_52_high REAL,
    week_52_low REAL,
    beta REAL,
    eps REAL,
    book_value REAL,
    FOREIGN KEY (isin) REFERENCES assets(isin) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol);
CREATE INDEX IF NOT EXISTS idx_stocks_sector ON stocks(sector);
CREATE INDEX IF NOT EXISTS idx_stocks_exchange ON stocks(exchange);

-- ==========================================
-- BONDS SPECIFIC
-- ==========================================
CREATE TABLE IF NOT EXISTS bonds (
    isin TEXT PRIMARY KEY,
    bond_name TEXT NOT NULL,
    issuer TEXT NOT NULL,
    bond_type TEXT CHECK(bond_type IN ('government', 'corporate', 'municipal', 'psu')),
    issue_date DATE,
    maturity_date DATE,
    face_value REAL,
    coupon_rate REAL,
    coupon_frequency TEXT,
    credit_rating TEXT,
    rating_agency TEXT,
    yield_to_maturity REAL,
    duration REAL,
    issue_size REAL,
    outstanding_amount REAL,
    security_type TEXT,
    FOREIGN KEY (isin) REFERENCES assets(isin) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_bonds_type ON bonds(bond_type);
CREATE INDEX IF NOT EXISTS idx_bonds_issuer ON bonds(issuer);
CREATE INDEX IF NOT EXISTS idx_bonds_maturity ON bonds(maturity_date);

-- ==========================================
-- PRICE HISTORY (Common for all assets)
-- ==========================================
CREATE TABLE IF NOT EXISTS price_history (
    id SERIAL PRIMARY KEY,
    isin TEXT NOT NULL,
    date DATE NOT NULL,
    open_price REAL,
    high_price REAL,
    low_price REAL,
    close_price REAL,
    volume BIGINT,
    value REAL,
    trades INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (isin) REFERENCES assets(isin) ON DELETE CASCADE,
    UNIQUE(isin, date)
);

CREATE INDEX IF NOT EXISTS idx_price_history_isin_date ON price_history(isin, date);
CREATE INDEX IF NOT EXISTS idx_price_history_date ON price_history(date);

-- ==========================================
-- CORPORATE ACTIONS
-- ==========================================
CREATE TABLE IF NOT EXISTS corporate_actions (
    id SERIAL PRIMARY KEY,
    isin TEXT NOT NULL,
    action_type TEXT NOT NULL CHECK(action_type IN ('dividend', 'bonus', 'split', 'rights', 'buyback')),
    ex_date DATE NOT NULL,
    record_date DATE,
    announcement_date DATE,
    details JSONB,
    FOREIGN KEY (isin) REFERENCES assets(isin) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_corporate_actions_isin ON corporate_actions(isin);

-- ==========================================
-- ASSET STATISTICS
-- ==========================================
CREATE TABLE IF NOT EXISTS asset_statistics (
    isin TEXT PRIMARY KEY,
    mean_daily_return REAL,
    std_daily_return REAL,
    annual_volatility REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    calmar_ratio REAL,
    alpha REAL,
    beta REAL,
    var_95 REAL,
    cvar_95 REAL,
    calculated_date DATE,
    FOREIGN KEY (isin) REFERENCES assets(isin) ON DELETE CASCADE
);

-- ==========================================
-- CORRELATIONS
-- ==========================================
CREATE TABLE IF NOT EXISTS asset_correlations (
    isin_1 TEXT NOT NULL,
    isin_2 TEXT NOT NULL,
    correlation REAL NOT NULL,
    period_days INTEGER NOT NULL,
    calculated_date DATE NOT NULL,
    PRIMARY KEY (isin_1, isin_2, period_days),
    FOREIGN KEY (isin_1) REFERENCES assets(isin) ON DELETE CASCADE,
    FOREIGN KEY (isin_2) REFERENCES assets(isin) ON DELETE CASCADE
);

-- ==========================================
-- DATA QUALITY TRACKING
-- ==========================================
CREATE TABLE IF NOT EXISTS data_quality_log (
    id SERIAL PRIMARY KEY,
    isin TEXT NOT NULL,
    check_date DATE NOT NULL,
    data_points_count INTEGER,
    missing_days INTEGER,
    quality_score REAL,
    issues JSONB,
    FOREIGN KEY (isin) REFERENCES assets(isin) ON DELETE CASCADE
);

-- ==========================================
-- USER EMAILS
-- ==========================================
CREATE TABLE IF NOT EXISTS user_emails (
    id SERIAL PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================
-- INDEX BENCHMARKS
-- ==========================================
CREATE TABLE IF NOT EXISTS indices (
    symbol TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    exchange TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS index_price_history (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    open_price REAL,
    high_price REAL,
    low_price REAL,
    close_price REAL,
    volume BIGINT,
    FOREIGN KEY (symbol) REFERENCES indices(symbol) ON DELETE CASCADE,
    UNIQUE(symbol, date)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_index_price_history_symbol_date ON index_price_history(symbol, date);

-- ==========================================
-- BATCH LOAD CHECKPOINTING
-- ==========================================
CREATE TABLE IF NOT EXISTS batch_checkpoint (
    id SERIAL PRIMARY KEY,
    job_type TEXT NOT NULL,
    identifier TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(job_type, identifier)
);
