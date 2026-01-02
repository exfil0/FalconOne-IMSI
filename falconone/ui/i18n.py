# FalconOne Internationalization (i18n)
# Python Backend Support
# Version: 3.0.0

"""
Internationalization support for FalconOne.
Supports multiple languages with Flask-Babel.
"""

from flask import Flask, request, session
from flask_babel import Babel, gettext, ngettext
import os
import json

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Español',
    'fr': 'Français',
    'de': 'Deutsch',
    'zh': '中文',
    'ja': '日本語',
    'ar': 'العربية',
    'ru': 'Русский'
}

# Default language
DEFAULT_LANGUAGE = 'en'

# Right-to-left languages
RTL_LANGUAGES = ['ar', 'he']


class I18nManager:
    """Manage internationalization for FalconOne."""
    
    def __init__(self, app: Flask = None):
        """Initialize i18n manager."""
        self.app = app
        self.babel = None
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """Initialize Flask app with i18n support."""
        self.app = app
        
        # Configure Babel
        app.config['BABEL_DEFAULT_LOCALE'] = DEFAULT_LANGUAGE
        app.config['BABEL_SUPPORTED_LOCALES'] = list(SUPPORTED_LANGUAGES.keys())
        app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'
        
        # Initialize Babel
        self.babel = Babel(app)
        self.babel.init_app(app, locale_selector=self.get_locale)
        
        # Register template filters
        app.jinja_env.filters['translate'] = gettext
        app.jinja_env.filters['ntranslate'] = ngettext
        
        # Register context processor
        @app.context_processor
        def inject_i18n():
            return {
                'get_locale': self.get_locale,
                'get_language_name': self.get_language_name,
                'supported_languages': SUPPORTED_LANGUAGES,
                'is_rtl': self.is_rtl,
                'current_language': self.get_locale()
            }
    
    def get_locale(self):
        """Determine the locale for the current request."""
        # 1. Check URL parameter
        locale = request.args.get('lang')
        if locale in SUPPORTED_LANGUAGES:
            session['language'] = locale
            return locale
        
        # 2. Check session
        if 'language' in session:
            locale = session['language']
            if locale in SUPPORTED_LANGUAGES:
                return locale
        
        # 3. Check Accept-Language header
        locale = request.accept_languages.best_match(list(SUPPORTED_LANGUAGES.keys()))
        if locale:
            return locale
        
        # 4. Default to English
        return DEFAULT_LANGUAGE
    
    def get_language_name(self, code: str) -> str:
        """Get language name from code."""
        return SUPPORTED_LANGUAGES.get(code, code)
    
    def is_rtl(self, code: str = None) -> bool:
        """Check if language is right-to-left."""
        if code is None:
            code = self.get_locale()
        return code in RTL_LANGUAGES
    
    def set_language(self, code: str):
        """Set current language."""
        if code in SUPPORTED_LANGUAGES:
            session['language'] = code
            return True
        return False
    
    def load_translations(self, locale: str) -> dict:
        """Load translation file for locale."""
        translations_file = os.path.join(
            self.app.root_path,
            'translations',
            locale,
            'LC_MESSAGES',
            'messages.json'
        )
        
        if os.path.exists(translations_file):
            with open(translations_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {}
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate a key with optional formatting."""
        return gettext(key).format(**kwargs)
    
    def translate_plural(self, singular: str, plural: str, n: int, **kwargs) -> str:
        """Translate with pluralization."""
        return ngettext(singular, plural, n).format(n=n, **kwargs)


# Translation helper functions
def _(message):
    """Shorthand for gettext."""
    return gettext(message)


def _n(singular, plural, n):
    """Shorthand for ngettext."""
    return ngettext(singular, plural, n)


# Flask route for language switching
def setup_language_routes(app: Flask, i18n_manager: I18nManager):
    """Set up language switching routes."""
    
    @app.route('/api/language', methods=['GET'])
    def get_current_language():
        """Get current language."""
        return {
            'success': True,
            'language': i18n_manager.get_locale(),
            'name': i18n_manager.get_language_name(i18n_manager.get_locale()),
            'rtl': i18n_manager.is_rtl()
        }
    
    @app.route('/api/language', methods=['POST'])
    def set_language():
        """Set language."""
        from flask import request, jsonify
        
        data = request.get_json()
        language = data.get('language')
        
        if i18n_manager.set_language(language):
            return jsonify({
                'success': True,
                'language': language,
                'name': i18n_manager.get_language_name(language),
                'rtl': i18n_manager.is_rtl(language)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Unsupported language'
            }), 400
    
    @app.route('/api/languages', methods=['GET'])
    def list_languages():
        """List all supported languages."""
        return {
            'success': True,
            'languages': [
                {
                    'code': code,
                    'name': name,
                    'rtl': code in RTL_LANGUAGES
                }
                for code, name in SUPPORTED_LANGUAGES.items()
            ]
        }


# Example usage in Flask app
def create_app():
    """Create Flask app with i18n support."""
    app = Flask(__name__)
    app.secret_key = 'your-secret-key'
    
    # Initialize i18n
    i18n_manager = I18nManager(app)
    
    # Setup language routes
    setup_language_routes(app, i18n_manager)
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
