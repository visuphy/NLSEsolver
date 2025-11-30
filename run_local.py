import os
from werkzeug.serving import run_simple
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from app import app # Import the app from your app.py file

# Set an environment variable to indicate local development mode
# This will be checked in app.py to bypass limits
os.environ['APP_ENV'] = 'development'

# This middleware creates a "virtual" subdirectory for your app.
application = DispatcherMiddleware(lambda e, s: s('404 NOT FOUND', [('Content-Type', 'text/plain')]), {
    '/NLSEsolver': app
})

if __name__ == '__main__':
    print("Starting local development server for NLSEsolver...")
    print("APP_ENV is set to:", os.environ.get('APP_ENV'))
    print("Access at: http://localhost:5000/NLSEsolver/")
    run_simple('localhost', 5000, application, use_reloader=True, use_debugger=True)