// ==================== DOCUMENTATION VIEWER ====================
// Interactive documentation viewer with search, dark mode, and syntax highlighting

// Configuration
const CONFIG = {
    apiEndpoint: '/api/documentation/content',
    tocStorageKey: 'doc-toc',
    themeStorageKey: 'doc-theme',
    scrollOffset: 80,
    searchDebounceMs: 300
};

// State management
const state = {
    documentation: null,
    sections: [],
    currentSection: null,
    searchTerm: '',
    theme: localStorage.getItem(CONFIG.themeStorageKey) || 'light'
};

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
    initializeTheme();
    initializeEventListeners();
    loadDocumentation();
    initializeSyntaxHighlighting();
});

// ==================== THEME MANAGEMENT ====================
function initializeTheme() {
    document.documentElement.setAttribute('data-theme', state.theme);
    updateThemeIcon();
}

function toggleTheme() {
    state.theme = state.theme === 'light' ? 'dark' : 'light';
    localStorage.setItem(CONFIG.themeStorageKey, state.theme);
    document.documentElement.setAttribute('data-theme', state.theme);
    updateThemeIcon();
}

function updateThemeIcon() {
    const icon = document.querySelector('#themeToggle i');
    if (icon) {
        icon.className = state.theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
    }
}

// ==================== EVENT LISTENERS ====================
function initializeEventListeners() {
    // Theme toggle
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }

    // Sidebar toggle (mobile)
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('docSidebar');
    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });
    }

    // Search input
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        let searchTimeout;
        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                handleSearch(e.target.value);
            }, CONFIG.searchDebounceMs);
        });
    }

    // Scroll to top button
    const scrollToTopBtn = document.getElementById('scrollToTop');
    if (scrollToTopBtn) {
        scrollToTopBtn.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });

        // Show/hide scroll to top button
        window.addEventListener('scroll', () => {
            if (window.pageYOffset > 300) {
                scrollToTopBtn.classList.add('visible');
            } else {
                scrollToTopBtn.classList.remove('visible');
            }
        });
    }

    // Handle TOC clicks
    document.addEventListener('click', (e) => {
        if (e.target.matches('.toc-link')) {
            e.preventDefault();
            const targetId = e.target.getAttribute('href').substring(1);
            scrollToSection(targetId);
            
            // Close mobile sidebar
            if (window.innerWidth <= 768) {
                document.getElementById('docSidebar').classList.remove('open');
            }
        }
    });

    // Track scroll position for active TOC item
    window.addEventListener('scroll', debounce(updateActiveTocItem, 100));
}

// ==================== DOCUMENTATION LOADING ====================
async function loadDocumentation() {
    try {
        const response = await fetch(CONFIG.apiEndpoint);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        state.documentation = data.content;
        state.sections = parseSections(data.content);
        
        renderTableOfContents();
        renderDocumentation(state.documentation);
        initializeSyntaxHighlighting();
        
    } catch (error) {
        console.error('Error loading documentation:', error);
        showError('Failed to load documentation. Please try again later.');
    }
}

// ==================== SECTION PARSING ====================
function parseSections(markdown) {
    const sections = [];
    const lines = markdown.split('\n');
    let currentSection = null;
    let currentContent = [];
    
    lines.forEach((line, index) => {
        // Match h1 headers (# Title)
        const h1Match = line.match(/^#\s+(.+)$/);
        if (h1Match) {
            // Save previous section
            if (currentSection) {
                currentSection.content = currentContent.join('\n');
                sections.push(currentSection);
            }
            
            // Start new section
            currentSection = {
                id: slugify(h1Match[1]),
                title: h1Match[1],
                level: 1,
                line: index,
                subsections: []
            };
            currentContent = [line];
            return;
        }
        
        // Match h2 headers (## Subtitle)
        const h2Match = line.match(/^##\s+(.+)$/);
        if (h2Match && currentSection) {
            const subsection = {
                id: slugify(h2Match[1]),
                title: h2Match[1],
                level: 2,
                line: index
            };
            currentSection.subsections.push(subsection);
        }
        
        if (currentSection) {
            currentContent.push(line);
        }
    });
    
    // Save last section
    if (currentSection) {
        currentSection.content = currentContent.join('\n');
        sections.push(currentSection);
    }
    
    return sections;
}

function slugify(text) {
    return text
        .toLowerCase()
        .replace(/[^\w\s-]/g, '')
        .replace(/\s+/g, '-')
        .replace(/-+/g, '-')
        .trim();
}

// ==================== TABLE OF CONTENTS ====================
function renderTableOfContents() {
    const tocContainer = document.getElementById('tableOfContents');
    
    if (state.sections.length === 0) {
        tocContainer.innerHTML = '<div class="no-results">No sections found</div>';
        return;
    }
    
    const tocHTML = state.sections.map(section => {
        const subsectionsHTML = section.subsections.length > 0 
            ? `<ul class="toc-subsection">
                ${section.subsections.map(sub => 
                    `<li class="toc-item">
                        <a href="#${sub.id}" class="toc-link">${escapeHtml(sub.title)}</a>
                    </li>`
                ).join('')}
               </ul>`
            : '';
        
        return `
            <div class="toc-section">
                <div class="toc-item">
                    <a href="#${section.id}" class="toc-link">${escapeHtml(section.title)}</a>
                </div>
                ${subsectionsHTML}
            </div>
        `;
    }).join('');
    
    tocContainer.innerHTML = tocHTML;
}

function updateActiveTocItem() {
    const tocLinks = document.querySelectorAll('.toc-link');
    const scrollPosition = window.scrollY + CONFIG.scrollOffset;
    
    // Remove all active classes
    tocLinks.forEach(link => link.classList.remove('active'));
    
    // Find current section
    const sections = document.querySelectorAll('.documentation-article h1, .documentation-article h2');
    let currentSection = null;
    
    sections.forEach(section => {
        if (section.offsetTop <= scrollPosition) {
            currentSection = section;
        }
    });
    
    // Highlight active TOC item
    if (currentSection) {
        const id = currentSection.id;
        const activeLink = document.querySelector(`.toc-link[href="#${id}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
    }
}

// ==================== DOCUMENTATION RENDERING ====================
function renderDocumentation(markdown) {
    const article = document.getElementById('documentationArticle');
    
    try {
        // Configure marked options
        marked.setOptions({
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (err) {
                        console.error('Highlight error:', err);
                    }
                }
                return hljs.highlightAuto(code).value;
            },
            breaks: true,
            gfm: true,
            headerIds: true,
            mangle: false
        });
        
        // Convert markdown to HTML
        const html = marked.parse(markdown);
        
        // Add IDs to headers for anchoring
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        
        doc.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach(header => {
            if (!header.id) {
                header.id = slugify(header.textContent);
            }
        });
        
        article.innerHTML = doc.body.innerHTML;
        
        // Add copy buttons to code blocks
        addCopyButtonsToCodeBlocks();
        
        // Update breadcrumb
        updateBreadcrumb('Documentation');
        
    } catch (error) {
        console.error('Error rendering documentation:', error);
        showError('Failed to render documentation.');
    }
}

function addCopyButtonsToCodeBlocks() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach((block, index) => {
        const pre = block.parentElement;
        pre.style.position = 'relative';
        
        const button = document.createElement('button');
        button.className = 'copy-code-btn';
        button.innerHTML = '<i class="fas fa-copy"></i>';
        button.title = 'Copy code';
        button.style.cssText = `
            position: absolute;
            top: 8px;
            right: 8px;
            background: var(--accent-color);
            color: white;
            border: none;
            padding: 6px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
            opacity: 0.8;
            transition: opacity 0.2s;
        `;
        
        button.addEventListener('mouseenter', () => {
            button.style.opacity = '1';
        });
        
        button.addEventListener('mouseleave', () => {
            button.style.opacity = '0.8';
        });
        
        button.addEventListener('click', () => {
            const code = block.textContent;
            navigator.clipboard.writeText(code).then(() => {
                button.innerHTML = '<i class="fas fa-check"></i>';
                setTimeout(() => {
                    button.innerHTML = '<i class="fas fa-copy"></i>';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy:', err);
                button.innerHTML = '<i class="fas fa-times"></i>';
                setTimeout(() => {
                    button.innerHTML = '<i class="fas fa-copy"></i>';
                }, 2000);
            });
        });
        
        pre.appendChild(button);
    });
}

// ==================== SEARCH ====================
function handleSearch(query) {
    state.searchTerm = query.toLowerCase().trim();
    
    if (!state.searchTerm) {
        // Show all sections
        renderDocumentation(state.documentation);
        renderTableOfContents();
        return;
    }
    
    // Filter sections by search term
    const matchingSections = state.sections.filter(section => {
        const titleMatch = section.title.toLowerCase().includes(state.searchTerm);
        const contentMatch = section.content.toLowerCase().includes(state.searchTerm);
        return titleMatch || contentMatch;
    });
    
    // Render filtered content
    if (matchingSections.length === 0) {
        showNoResults();
    } else {
        const filteredMarkdown = matchingSections.map(section => section.content).join('\n\n');
        renderDocumentation(filteredMarkdown);
        highlightSearchTerms();
    }
}

function highlightSearchTerms() {
    if (!state.searchTerm) return;
    
    const article = document.getElementById('documentationArticle');
    const walker = document.createTreeWalker(
        article,
        NodeFilter.SHOW_TEXT,
        null,
        false
    );
    
    const nodesToReplace = [];
    let node;
    
    while (node = walker.nextNode()) {
        if (node.parentElement.tagName === 'SCRIPT' || node.parentElement.tagName === 'STYLE') {
            continue;
        }
        
        const text = node.textContent;
        if (text.toLowerCase().includes(state.searchTerm)) {
            nodesToReplace.push(node);
        }
    }
    
    nodesToReplace.forEach(node => {
        const text = node.textContent;
        const regex = new RegExp(`(${escapeRegex(state.searchTerm)})`, 'gi');
        const highlighted = text.replace(regex, '<mark class="search-highlight">$1</mark>');
        
        const span = document.createElement('span');
        span.innerHTML = highlighted;
        node.parentElement.replaceChild(span, node);
    });
}

function showNoResults() {
    const article = document.getElementById('documentationArticle');
    article.innerHTML = `
        <div class="no-results" style="padding: 4rem 2rem; text-align: center;">
            <i class="fas fa-search" style="font-size: 3rem; color: var(--text-secondary); margin-bottom: 1rem;"></i>
            <h2 style="color: var(--text-secondary);">No results found</h2>
            <p style="color: var(--text-secondary);">Try a different search term</p>
        </div>
    `;
}

// ==================== NAVIGATION ====================
function scrollToSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        const offset = element.offsetTop - CONFIG.scrollOffset;
        window.scrollTo({ top: offset, behavior: 'smooth' });
    }
}

function updateBreadcrumb(title) {
    const breadcrumb = document.getElementById('breadcrumb');
    breadcrumb.innerHTML = `
        <a href="/">Dashboard</a>
        <span class="breadcrumb-separator">/</span>
        <span>${escapeHtml(title)}</span>
    `;
}

// ==================== SYNTAX HIGHLIGHTING ====================
function initializeSyntaxHighlighting() {
    // Register additional languages if needed
    if (typeof hljs !== 'undefined') {
        // Highlight all code blocks
        document.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
    }
}

// ==================== ERROR HANDLING ====================
function showError(message) {
    const article = document.getElementById('documentationArticle');
    article.innerHTML = `
        <div class="error-message" style="padding: 4rem 2rem; text-align: center;">
            <i class="fas fa-exclamation-triangle" style="font-size: 3rem; color: var(--error-color); margin-bottom: 1rem;"></i>
            <h2 style="color: var(--error-color);">Error</h2>
            <p style="color: var(--text-secondary);">${escapeHtml(message)}</p>
            <button onclick="location.reload()" style="
                margin-top: 1rem;
                padding: 0.75rem 1.5rem;
                background: var(--accent-color);
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 1rem;
            ">Reload Page</button>
        </div>
    `;
}

// ==================== UTILITY FUNCTIONS ====================
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ==================== EXPORT FOR TESTING ====================
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        slugify,
        parseSections,
        escapeHtml,
        escapeRegex
    };
}
