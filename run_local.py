"""Local development server — set DATABASE_URL before running."""
import os

# Set this to your Railway Postgres URL for local DB access
# os.environ['DATABASE_URL'] = 'postgresql://user:pass@host/db'

from server import app
if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)
