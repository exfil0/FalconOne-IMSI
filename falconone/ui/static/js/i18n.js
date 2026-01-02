// FalconOne Internationalization (i18n) - Frontend
// Version: 3.0.0

(function() {
    'use strict';
    
    // Supported languages
    const SUPPORTED_LANGUAGES = {
        'en': { name: 'English', nativeName: 'English', rtl: false },
        'es': { name: 'Spanish', nativeName: 'Español', rtl: false },
        'fr': { name: 'French', nativeName: 'Français', rtl: false },
        'de': { name: 'German', nativeName: 'Deutsch', rtl: false },
        'zh': { name: 'Chinese', nativeName: '中文', rtl: false },
        'ja': { name: 'Japanese', nativeName: '日本語', rtl: false },
        'ar': { name: 'Arabic', nativeName: 'العربية', rtl: true },
        'ru': { name: 'Russian', nativeName: 'Русский', rtl: false }
    };
    
    // Translation storage
    let translations = {};
    let currentLanguage = 'en';
    
    // Initialize i18n
    const init = async () => {
        // Load saved language preference
        const savedLang = localStorage.getItem('falconone-language');
        const browserLang = navigator.language.split('-')[0];
        
        // Use saved language, browser language, or default to English
        const lang = savedLang || (SUPPORTED_LANGUAGES[browserLang] ? browserLang : 'en');
        
        await setLanguage(lang);
        createLanguageSelector();
        setupKeyboardShortcuts();
    };
    
    // Load translations for a language
    const loadTranslations = async (lang) => {
        try {
            const response = await fetch(`/static/i18n/${lang}.json`);
            if (response.ok) {
                translations[lang] = await response.json();
                return true;
            }
        } catch (error) {
            console.error(`Failed to load translations for ${lang}:`, error);
        }
        return false;
    };
    
    // Set current language
    const setLanguage = async (lang) => {
        if (!SUPPORTED_LANGUAGES[lang]) {
            console.error(`Unsupported language: ${lang}`);
            return false;
        }
        
        // Load translations if not already loaded
        if (!translations[lang]) {
            await loadTranslations(lang);
        }
        
        currentLanguage = lang;
        localStorage.setItem('falconone-language', lang);
        
        // Update HTML lang attribute
        document.documentElement.setAttribute('lang', lang);
        
        // Update text direction for RTL languages
        const langInfo = SUPPORTED_LANGUAGES[lang];
        document.documentElement.setAttribute('dir', langInfo.rtl ? 'rtl' : 'ltr');
        
        // Translate all elements
        translatePage();
        
        // Update language selector
        updateLanguageSelector();
        
        // Emit custom event
        window.dispatchEvent(new CustomEvent('languageChanged', { detail: { lang } }));
        
        // Announce language change to screen readers
        announceLanguageChange(lang);
        
        // Update backend
        await updateBackendLanguage(lang);
        
        return true;
    };
    
    // Translate page
    const translatePage = () => {
        // Translate elements with data-i18n attribute
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            const translation = translate(key);
            
            if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
                element.placeholder = translation;
            } else {
                element.textContent = translation;
            }
        });
        
        // Translate elements with data-i18n-html attribute
        document.querySelectorAll('[data-i18n-html]').forEach(element => {
            const key = element.getAttribute('data-i18n-html');
            element.innerHTML = translate(key);
        });
        
        // Translate title attribute
        document.querySelectorAll('[data-i18n-title]').forEach(element => {
            const key = element.getAttribute('data-i18n-title');
            element.title = translate(key);
        });
        
        // Translate aria-label attribute
        document.querySelectorAll('[data-i18n-aria-label]').forEach(element => {
            const key = element.getAttribute('data-i18n-aria-label');
            element.setAttribute('aria-label', translate(key));
        });
        
        // Update page title
        const titleKey = document.querySelector('meta[name="i18n-title"]');
        if (titleKey) {
            document.title = translate(titleKey.content);
        }
    };
    
    // Translate a key
    const translate = (key, params = {}) => {
        const langTranslations = translations[currentLanguage] || {};
        let translation = langTranslations[key] || key;
        
        // Replace parameters
        Object.keys(params).forEach(paramKey => {
            translation = translation.replace(`{${paramKey}}`, params[paramKey]);
        });
        
        return translation;
    };
    
    // Translate with pluralization
    const translatePlural = (key, count) => {
        const langTranslations = translations[currentLanguage] || {};
        const pluralKey = count === 1 ? `${key}_one` : `${key}_other`;
        const translation = langTranslations[pluralKey] || langTranslations[key] || key;
        
        return translation.replace('{count}', count);
    };
    
    // Create language selector dropdown
    const createLanguageSelector = () => {
        if (document.querySelector('.language-selector')) return;
        
        const selector = document.createElement('div');
        selector.className = 'language-selector';
        selector.innerHTML = `
            <button class="language-selector-btn" aria-label="Change language" aria-haspopup="true" aria-expanded="false">
                <i class="fas fa-globe"></i>
                <span class="current-language">${currentLanguage.toUpperCase()}</span>
            </button>
            <ul class="language-selector-menu" role="menu" aria-label="Language options">
                ${Object.entries(SUPPORTED_LANGUAGES).map(([code, info]) => `
                    <li role="none">
                        <button role="menuitem" data-lang="${code}" ${code === currentLanguage ? 'aria-current="true"' : ''}>
                            ${info.nativeName}
                        </button>
                    </li>
                `).join('')}
            </ul>
        `;
        
        // Add event listeners
        const btn = selector.querySelector('.language-selector-btn');
        const menu = selector.querySelector('.language-selector-menu');
        
        btn.addEventListener('click', () => {
            const isOpen = menu.classList.toggle('show');
            btn.setAttribute('aria-expanded', isOpen);
        });
        
        // Language selection
        selector.querySelectorAll('[data-lang]').forEach(langBtn => {
            langBtn.addEventListener('click', async (e) => {
                const lang = e.target.getAttribute('data-lang');
                await setLanguage(lang);
                menu.classList.remove('show');
                btn.setAttribute('aria-expanded', 'false');
            });
        });
        
        // Close on click outside
        document.addEventListener('click', (e) => {
            if (!selector.contains(e.target)) {
                menu.classList.remove('show');
                btn.setAttribute('aria-expanded', 'false');
            }
        });
        
        document.body.appendChild(selector);
    };
    
    // Update language selector
    const updateLanguageSelector = () => {
        const currentLangElement = document.querySelector('.language-selector .current-language');
        if (currentLangElement) {
            currentLangElement.textContent = currentLanguage.toUpperCase();
        }
        
        // Update active state
        document.querySelectorAll('.language-selector [data-lang]').forEach(btn => {
            const lang = btn.getAttribute('data-lang');
            if (lang === currentLanguage) {
                btn.setAttribute('aria-current', 'true');
                btn.classList.add('active');
            } else {
                btn.removeAttribute('aria-current');
                btn.classList.remove('active');
            }
        });
    };
    
    // Update backend language setting
    const updateBackendLanguage = async (lang) => {
        try {
            await fetch('/api/language', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ language: lang })
            });
        } catch (error) {
            console.error('Failed to update backend language:', error);
        }
    };
    
    // Announce language change to screen readers
    const announceLanguageChange = (lang) => {
        const announcement = document.createElement('div');
        announcement.setAttribute('role', 'status');
        announcement.setAttribute('aria-live', 'polite');
        announcement.className = 'sr-only';
        announcement.textContent = `Language changed to ${SUPPORTED_LANGUAGES[lang].name}`;
        document.body.appendChild(announcement);
        
        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    };
    
    // Keyboard shortcuts
    const setupKeyboardShortcuts = () => {
        document.addEventListener('keydown', (e) => {
            // Ctrl+Shift+L to open language selector
            if (e.ctrlKey && e.shiftKey && e.key === 'L') {
                e.preventDefault();
                const btn = document.querySelector('.language-selector-btn');
                if (btn) btn.click();
            }
        });
    };
    
    // Export API
    window.I18n = {
        init: init,
        setLanguage: setLanguage,
        translate: translate,
        translatePlural: translatePlural,
        getCurrentLanguage: () => currentLanguage,
        getSupportedLanguages: () => SUPPORTED_LANGUAGES
    };
    
    // Auto-initialize
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();

// Add CSS for language selector
const style = document.createElement('style');
style.textContent = `
    .language-selector {
        position: fixed;
        top: 80px;
        right: 20px;
        z-index: 999;
    }
    
    .language-selector-btn {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px var(--shadow);
        position: relative;
    }
    
    .language-selector-btn:hover {
        background-color: var(--bg-tertiary);
        transform: scale(1.1);
        box-shadow: 0 4px 12px var(--shadow-strong);
    }
    
    .language-selector-btn i {
        font-size: 20px;
        color: var(--text-primary);
    }
    
    .current-language {
        position: absolute;
        bottom: -4px;
        right: -4px;
        background-color: var(--accent-primary);
        color: #ffffff;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 4px;
        border-radius: 4px;
    }
    
    .language-selector-menu {
        position: absolute;
        top: 100%;
        right: 0;
        margin-top: 8px;
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        min-width: 200px;
        max-height: 0;
        overflow: hidden;
        opacity: 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px var(--shadow);
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .language-selector-menu.show {
        max-height: 400px;
        opacity: 1;
        margin-top: 12px;
        overflow-y: auto;
    }
    
    .language-selector-menu li {
        border-bottom: 1px solid var(--border-color);
    }
    
    .language-selector-menu li:last-child {
        border-bottom: none;
    }
    
    .language-selector-menu button {
        width: 100%;
        padding: 12px 16px;
        text-align: left;
        background: transparent;
        border: none;
        color: var(--text-primary);
        cursor: pointer;
        transition: all 0.2s ease;
        min-height: 44px;
    }
    
    .language-selector-menu button:hover {
        background-color: var(--bg-tertiary);
        color: var(--accent-primary);
    }
    
    .language-selector-menu button[aria-current="true"],
    .language-selector-menu button.active {
        background-color: var(--accent-primary);
        color: #ffffff;
        font-weight: 700;
    }
    
    @media (max-width: 768px) {
        .language-selector {
            top: 70px;
            right: 15px;
        }
        
        .language-selector-btn {
            width: 40px;
            height: 40px;
        }
        
        .language-selector-btn i {
            font-size: 16px;
        }
    }
`;
document.head.appendChild(style);
