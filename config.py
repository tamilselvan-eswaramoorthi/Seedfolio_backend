import os
from dotenv import load_dotenv
load_dotenv()
from passlib.context import CryptContext

class Config:
    DB_HOST = os.getenv('DB_HOST', '').strip()
    DB_PORT = os.getenv('DB_PORT', '1433').strip()
    DB_NAME = os.getenv('DB_NAME', '').strip()
    DB_USER = os.getenv('DB_USER', '').strip()
    DB_PASSWORD = os.getenv('DB_PASSWORD', '').strip()
    DB_SCHEMA = os.getenv('DB_SCHEMA', 'dbo').strip()

    PWD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")

    REDIRECT_URI = os.getenv('REDIRECT_URI', "http://localhost:7071/api/gmail/oauth-callback/").strip()
    
    JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key').strip()
    JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256').strip()
    JWT_EXP_DELTA_SECONDS = os.getenv('JWT_EXP_DELTA_SECONDS', 86400)
    BENCHMARK_TICKERS = '^NSEI,^CRSLDX,^BSESN'
    BENCHMARK_NAMES = {
        '^NSEI': 'Nifty 50',
        '^NSEBANK': 'Nifty Bank',
        '^BSESN': 'BSE Sensex'
    }

    TRADEBRAINS_BASE_URL = "https://portal.tradebrains.in/api/corporateactions/upcomming/"
