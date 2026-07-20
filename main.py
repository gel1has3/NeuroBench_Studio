"""
NeuroBench Studio — Application Launcher
=========================================
Entry point used by:
  - PyInstaller bundled executable  (frozen)
  - Direct Python run               (development)

Architecture:
  1. Flask server starts on a daemon thread (port 5000, localhost-only)
  2. We poll until Flask is ready
  3. pywebview opens a native OS window pointing at http://localhost:5000
  4. When the window closes → process exits cleanly
"""

import os
import sys
import time
import socket
import threading
import logging
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Path resolution — works in both normal Python and PyInstaller frozen mode
# ─────────────────────────────────────────────────────────────────────────────

# When frozen by PyInstaller, sys._MEIPASS points to the temp extraction dir.
# When running normally, it is the project root (parent of this file).
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    BUNDLE_DIR = Path(sys._MEIPASS)          # read-only bundled resources
    PROJECT_ROOT = BUNDLE_DIR
else:
    # Running in a normal Python environment
    PROJECT_ROOT = Path(__file__).parent.resolve()
    BUNDLE_DIR = PROJECT_ROOT

# User-writable data directory (results, configs, downloaded models)
# Stored in the user's home folder so it survives app updates.
USER_DATA_DIR = Path.home() / "NeuroBench_Studio"
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
(USER_DATA_DIR / "results").mkdir(exist_ok=True)
(USER_DATA_DIR / "data").mkdir(exist_ok=True)
(USER_DATA_DIR / "configs").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Disable numba JIT early (same as flask_app.py / pipeline_executor.py)
# ─────────────────────────────────────────────────────────────────────────────

os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(USER_DATA_DIR / '.numba_cache')

# ─────────────────────────────────────────────────────────────────────────────
# Inject project root into sys.path so local imports resolve correctly
# ─────────────────────────────────────────────────────────────────────────────

project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000
APP_TITLE = 'NeuroBench Studio'
APP_VERSION = '2.0.0'
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 900
FLASK_READY_TIMEOUT = 60  # seconds to wait for Flask before showing an error

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

log_dir = USER_DATA_DIR / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'neurobench.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('NeuroBench.Launcher')


# ─────────────────────────────────────────────────────────────────────────────
# Flask server thread
# ─────────────────────────────────────────────────────────────────────────────

flask_ready = threading.Event()
flask_error = None   # Will hold the Exception if startup fails


def _run_flask():
    """Start the Flask application on a daemon background thread."""
    global flask_error
    try:
        # Set environment variables the Flask app reads
        os.environ.setdefault('FLASK_HOST', FLASK_HOST)
        os.environ.setdefault('FLASK_PORT', str(FLASK_PORT))
        os.environ.setdefault('FLASK_DEBUG', 'false')
        os.environ['RESULTS_DIR'] = str(USER_DATA_DIR / 'results')

        from src.dashboard.flask_app import create_app
        app = create_app()

        logger.info(f'Starting Flask on {FLASK_HOST}:{FLASK_PORT}')

        # Use Werkzeug's underlying server directly so we can pass
        # use_reloader=False (required inside a thread/frozen bundle).
        from werkzeug.serving import make_server
        server = make_server(FLASK_HOST, FLASK_PORT, app, threaded=True)
        flask_ready.set()
        logger.info('Flask server is ready')
        server.serve_forever()

    except Exception as exc:
        flask_error = exc
        flask_ready.set()  # unblock the waiter even on error
        logger.exception('Flask server failed to start')


# ─────────────────────────────────────────────────────────────────────────────
# Readiness probe
# ─────────────────────────────────────────────────────────────────────────────

def _wait_for_flask(timeout=FLASK_READY_TIMEOUT):
    """Poll the Flask port until it accepts connections or timeout elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((FLASK_HOST, FLASK_PORT), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info(f'=== {APP_TITLE} v{APP_VERSION} starting ===')
    logger.info(f'Bundle dir  : {BUNDLE_DIR}')
    logger.info(f'User data   : {USER_DATA_DIR}')
    logger.info(f'Python      : {sys.version}')
    logger.info(f'Frozen      : {getattr(sys, "frozen", False)}')

    # ── 1. Start Flask in background ──────────────────────────────────────
    flask_thread = threading.Thread(target=_run_flask, name='FlaskServer', daemon=True)
    flask_thread.start()

    # ── 2. Wait for Flask to be ready ─────────────────────────────────────
    flask_ready.wait(timeout=FLASK_READY_TIMEOUT)

    if flask_error:
        _show_error_and_exit(
            'Flask server failed to start:\n\n{}\n\nCheck {} for details.'.format(
                flask_error, log_dir / 'neurobench.log'
            )
        )
        return

    if not _wait_for_flask():
        _show_error_and_exit(
            'Flask did not become ready within {}s.\n'
            'Check {} for details.'.format(
                FLASK_READY_TIMEOUT, log_dir / 'neurobench.log'
            )
        )
        return

    # ── 3. Open native window with pywebview ──────────────────────────────
    try:
        import webview  # pywebview
    except ImportError:
        # Fallback: open in system browser if pywebview is not available
        logger.warning('pywebview not found — opening system browser as fallback')
        import webbrowser
        webbrowser.open('http://{}:{}'.format(FLASK_HOST, FLASK_PORT))
        logger.info('Press Ctrl+C to stop the server.')
        try:
            flask_thread.join()
        except KeyboardInterrupt:
            logger.info('Shutting down.')
        return

    url = 'http://{}:{}'.format(FLASK_HOST, FLASK_PORT)
    logger.info('Opening window -> {}'.format(url))

    webview.create_window(
        title=APP_TITLE,
        url=url,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=(1024, 600),
        resizable=True,
        text_select=True,
    )

    # pywebview.start() blocks until the window is closed
    webview.start(debug=False)
    logger.info('Window closed — exiting.')


def _show_error_and_exit(message):
    """Show an error dialog (best-effort) and exit with code 1."""
    logger.error(message)
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(APP_TITLE, message)
        root.destroy()
    except Exception:
        print('\n[ERROR] {}\n'.format(message), file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
    main()
