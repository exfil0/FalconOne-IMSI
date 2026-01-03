/**
 * FalconOne Accessible UI Components (WCAG 2.1 AA Compliant)
 * JavaScript implementation for accessible interactions
 * 
 * Version 1.9.3: Enhanced accessibility and usability
 */

(function(global) {
    'use strict';

    // ==================== Toast Notification System ====================
    
    class ToastManager {
        constructor(options = {}) {
            this.options = {
                position: options.position || 'top-right',
                duration: options.duration || 5000,
                maxToasts: options.maxToasts || 5,
                pauseOnHover: options.pauseOnHover !== false,
                pauseOnFocusLoss: options.pauseOnFocusLoss !== false,
                ...options
            };
            
            this.toasts = new Map();
            this.container = null;
            this.toastId = 0;
            
            this._init();
        }
        
        _init() {
            // Create container with ARIA live region
            this.container = document.createElement('div');
            this.container.className = 'toast-container';
            this.container.setAttribute('role', 'region');
            this.container.setAttribute('aria-label', 'Notifications');
            this.container.setAttribute('aria-live', 'polite');
            this.container.setAttribute('aria-atomic', 'false');
            
            // Position based on options
            const positions = {
                'top-right': { top: '24px', right: '24px' },
                'top-left': { top: '24px', left: '24px' },
                'bottom-right': { bottom: '24px', right: '24px' },
                'bottom-left': { bottom: '24px', left: '24px' },
                'top-center': { top: '24px', left: '50%', transform: 'translateX(-50%)' },
                'bottom-center': { bottom: '24px', left: '50%', transform: 'translateX(-50%)' }
            };
            
            Object.assign(this.container.style, positions[this.options.position] || positions['top-right']);
            
            document.body.appendChild(this.container);
            
            // Handle focus loss
            if (this.options.pauseOnFocusLoss) {
                document.addEventListener('visibilitychange', () => {
                    if (document.hidden) {
                        this._pauseAllTimers();
                    } else {
                        this._resumeAllTimers();
                    }
                });
            }
        }
        
        /**
         * Show a toast notification
         * @param {Object} options Toast options
         * @returns {number} Toast ID
         */
        show(options) {
            const id = ++this.toastId;
            
            const config = {
                id,
                type: options.type || 'info',
                title: options.title || '',
                message: options.message || '',
                duration: options.duration ?? this.options.duration,
                dismissible: options.dismissible !== false,
                actions: options.actions || [],
                onDismiss: options.onDismiss || null,
                ...options
            };
            
            // Enforce max toasts
            while (this.toasts.size >= this.options.maxToasts) {
                const firstId = this.toasts.keys().next().value;
                this.dismiss(firstId);
            }
            
            const toast = this._createToastElement(config);
            this.container.appendChild(toast);
            
            // Store toast data
            this.toasts.set(id, {
                element: toast,
                config,
                timer: null,
                remainingTime: config.duration,
                pausedAt: null
            });
            
            // Start auto-dismiss timer
            if (config.duration > 0) {
                this._startTimer(id);
            }
            
            // Focus management for accessibility
            toast.focus();
            
            return id;
        }
        
        _createToastElement(config) {
            const toast = document.createElement('div');
            toast.className = `toast toast-${config.type}`;
            toast.setAttribute('role', 'alert');
            toast.setAttribute('aria-live', config.type === 'error' ? 'assertive' : 'polite');
            toast.setAttribute('tabindex', '-1');
            toast.dataset.toastId = config.id;
            
            // Icon based on type
            const icons = {
                success: '✓',
                error: '✕',
                warning: '⚠',
                info: 'ℹ'
            };
            
            const iconLabel = {
                success: 'Success',
                error: 'Error',
                warning: 'Warning',
                info: 'Information'
            };
            
            toast.innerHTML = `
                <span class="toast-icon" aria-hidden="true">${icons[config.type] || icons.info}</span>
                <div class="toast-content">
                    ${config.title ? `<div class="toast-title">${this._escapeHtml(config.title)}</div>` : ''}
                    <div class="toast-message">${this._escapeHtml(config.message)}</div>
                    ${config.actions.length > 0 ? this._renderActions(config.actions) : ''}
                </div>
                ${config.dismissible ? `
                    <button class="toast-close" aria-label="Dismiss notification" type="button">
                        <span aria-hidden="true">×</span>
                    </button>
                ` : ''}
                ${config.duration > 0 ? `<div class="toast-progress" style="animation-duration: ${config.duration}ms"></div>` : ''}
            `;
            
            // Add screen reader prefix
            const srPrefix = document.createElement('span');
            srPrefix.className = 'sr-only';
            srPrefix.textContent = iconLabel[config.type] + ': ';
            toast.querySelector('.toast-content').prepend(srPrefix);
            
            // Event listeners
            if (config.dismissible) {
                toast.querySelector('.toast-close').addEventListener('click', () => {
                    this.dismiss(config.id);
                });
            }
            
            // Pause on hover
            if (this.options.pauseOnHover && config.duration > 0) {
                toast.addEventListener('mouseenter', () => this._pauseTimer(config.id));
                toast.addEventListener('mouseleave', () => this._resumeTimer(config.id));
                toast.addEventListener('focusin', () => this._pauseTimer(config.id));
                toast.addEventListener('focusout', () => this._resumeTimer(config.id));
            }
            
            // Keyboard dismiss
            toast.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && config.dismissible) {
                    this.dismiss(config.id);
                }
            });
            
            return toast;
        }
        
        _renderActions(actions) {
            return `
                <div class="toast-actions" role="group" aria-label="Notification actions">
                    ${actions.map((action, i) => `
                        <button class="toast-action" data-action-index="${i}" type="button">
                            ${this._escapeHtml(action.label)}
                        </button>
                    `).join('')}
                </div>
            `;
        }
        
        _startTimer(id) {
            const toast = this.toasts.get(id);
            if (!toast || toast.config.duration <= 0) return;
            
            toast.timer = setTimeout(() => {
                this.dismiss(id);
            }, toast.remainingTime);
        }
        
        _pauseTimer(id) {
            const toast = this.toasts.get(id);
            if (!toast || !toast.timer) return;
            
            clearTimeout(toast.timer);
            toast.timer = null;
            toast.pausedAt = Date.now();
            
            // Pause progress bar animation
            const progress = toast.element.querySelector('.toast-progress');
            if (progress) {
                progress.style.animationPlayState = 'paused';
            }
        }
        
        _resumeTimer(id) {
            const toast = this.toasts.get(id);
            if (!toast || toast.timer) return;
            
            if (toast.pausedAt) {
                const elapsed = Date.now() - toast.pausedAt;
                toast.remainingTime = Math.max(0, toast.remainingTime - elapsed);
                toast.pausedAt = null;
            }
            
            if (toast.remainingTime > 0) {
                this._startTimer(id);
            }
            
            // Resume progress bar animation
            const progress = toast.element.querySelector('.toast-progress');
            if (progress) {
                progress.style.animationPlayState = 'running';
            }
        }
        
        _pauseAllTimers() {
            this.toasts.forEach((_, id) => this._pauseTimer(id));
        }
        
        _resumeAllTimers() {
            this.toasts.forEach((_, id) => this._resumeTimer(id));
        }
        
        /**
         * Dismiss a toast
         * @param {number} id Toast ID
         */
        dismiss(id) {
            const toast = this.toasts.get(id);
            if (!toast) return;
            
            // Clear timer
            if (toast.timer) {
                clearTimeout(toast.timer);
            }
            
            // Animate out
            toast.element.style.animation = 'toast-slide-in 200ms ease-in-out reverse';
            
            setTimeout(() => {
                toast.element.remove();
                this.toasts.delete(id);
                
                // Call dismiss callback
                if (toast.config.onDismiss) {
                    toast.config.onDismiss();
                }
            }, 200);
        }
        
        /**
         * Dismiss all toasts
         */
        dismissAll() {
            this.toasts.forEach((_, id) => this.dismiss(id));
        }
        
        // Convenience methods
        success(message, options = {}) {
            return this.show({ ...options, type: 'success', message });
        }
        
        error(message, options = {}) {
            return this.show({ ...options, type: 'error', message, duration: options.duration ?? 0 });
        }
        
        warning(message, options = {}) {
            return this.show({ ...options, type: 'warning', message });
        }
        
        info(message, options = {}) {
            return this.show({ ...options, type: 'info', message });
        }
        
        _escapeHtml(str) {
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }
    }
    
    // ==================== Drag and Drop System ====================
    
    class DraggableList {
        constructor(container, options = {}) {
            this.container = typeof container === 'string' 
                ? document.querySelector(container) 
                : container;
            
            this.options = {
                itemSelector: options.itemSelector || '.draggable-item',
                handleSelector: options.handleSelector || '.drag-handle',
                onReorder: options.onReorder || null,
                keyboardReorder: options.keyboardReorder !== false,
                ...options
            };
            
            this.draggedItem = null;
            this.draggedIndex = null;
            
            this._init();
        }
        
        _init() {
            if (!this.container) return;
            
            this.container.setAttribute('role', 'listbox');
            this.container.setAttribute('aria-label', this.options.label || 'Reorderable list');
            
            this._initItems();
            this._bindEvents();
        }
        
        _initItems() {
            const items = this.container.querySelectorAll(this.options.itemSelector);
            
            items.forEach((item, index) => {
                item.setAttribute('role', 'option');
                item.setAttribute('tabindex', index === 0 ? '0' : '-1');
                item.setAttribute('draggable', 'true');
                item.setAttribute('aria-grabbed', 'false');
                item.dataset.index = index;
                
                // Add keyboard reorder controls
                if (this.options.keyboardReorder) {
                    this._addReorderControls(item, index, items.length);
                }
            });
        }
        
        _addReorderControls(item, index, total) {
            const controls = document.createElement('div');
            controls.className = 'reorder-controls';
            controls.setAttribute('role', 'group');
            controls.setAttribute('aria-label', 'Reorder controls');
            
            controls.innerHTML = `
                <button class="reorder-btn" data-direction="up" 
                        aria-label="Move up" 
                        ${index === 0 ? 'disabled' : ''}>↑</button>
                <button class="reorder-btn" data-direction="down" 
                        aria-label="Move down"
                        ${index === total - 1 ? 'disabled' : ''}>↓</button>
            `;
            
            item.appendChild(controls);
        }
        
        _bindEvents() {
            // Drag events
            this.container.addEventListener('dragstart', this._onDragStart.bind(this));
            this.container.addEventListener('dragend', this._onDragEnd.bind(this));
            this.container.addEventListener('dragover', this._onDragOver.bind(this));
            this.container.addEventListener('drop', this._onDrop.bind(this));
            this.container.addEventListener('dragleave', this._onDragLeave.bind(this));
            
            // Keyboard events
            this.container.addEventListener('keydown', this._onKeydown.bind(this));
            
            // Reorder button clicks
            this.container.addEventListener('click', this._onReorderClick.bind(this));
        }
        
        _onDragStart(e) {
            const item = e.target.closest(this.options.itemSelector);
            if (!item) return;
            
            this.draggedItem = item;
            this.draggedIndex = parseInt(item.dataset.index);
            
            item.classList.add('dragging');
            item.setAttribute('aria-grabbed', 'true');
            
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', this.draggedIndex);
            
            // Announce to screen readers
            this._announce(`Grabbed item ${this.draggedIndex + 1}. Use arrow keys or drop on another item to reorder.`);
        }
        
        _onDragEnd(e) {
            if (this.draggedItem) {
                this.draggedItem.classList.remove('dragging');
                this.draggedItem.setAttribute('aria-grabbed', 'false');
                this.draggedItem = null;
                this.draggedIndex = null;
            }
            
            // Remove all drag-over classes
            this.container.querySelectorAll('.drag-over').forEach(item => {
                item.classList.remove('drag-over');
            });
        }
        
        _onDragOver(e) {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            
            const item = e.target.closest(this.options.itemSelector);
            if (item && item !== this.draggedItem) {
                // Remove from others
                this.container.querySelectorAll('.drag-over').forEach(i => {
                    if (i !== item) i.classList.remove('drag-over');
                });
                item.classList.add('drag-over');
            }
        }
        
        _onDragLeave(e) {
            const item = e.target.closest(this.options.itemSelector);
            if (item) {
                item.classList.remove('drag-over');
            }
        }
        
        _onDrop(e) {
            e.preventDefault();
            
            const dropTarget = e.target.closest(this.options.itemSelector);
            if (!dropTarget || dropTarget === this.draggedItem) return;
            
            dropTarget.classList.remove('drag-over');
            
            const fromIndex = this.draggedIndex;
            const toIndex = parseInt(dropTarget.dataset.index);
            
            this._reorderItems(fromIndex, toIndex);
        }
        
        _onKeydown(e) {
            const item = e.target.closest(this.options.itemSelector);
            if (!item) return;
            
            const items = Array.from(this.container.querySelectorAll(this.options.itemSelector));
            const currentIndex = items.indexOf(item);
            
            switch (e.key) {
                case 'ArrowUp':
                case 'ArrowLeft':
                    e.preventDefault();
                    if (currentIndex > 0) {
                        items[currentIndex - 1].focus();
                    }
                    break;
                    
                case 'ArrowDown':
                case 'ArrowRight':
                    e.preventDefault();
                    if (currentIndex < items.length - 1) {
                        items[currentIndex + 1].focus();
                    }
                    break;
                    
                case ' ':
                case 'Enter':
                    if (e.key === ' ') e.preventDefault();
                    this._toggleGrabbed(item, items);
                    break;
                    
                case 'Escape':
                    if (item.getAttribute('aria-grabbed') === 'true') {
                        item.setAttribute('aria-grabbed', 'false');
                        this.draggedItem = null;
                        this._announce('Reorder cancelled');
                    }
                    break;
                    
                case 'Home':
                    e.preventDefault();
                    items[0].focus();
                    break;
                    
                case 'End':
                    e.preventDefault();
                    items[items.length - 1].focus();
                    break;
            }
        }
        
        _toggleGrabbed(item, items) {
            const isGrabbed = item.getAttribute('aria-grabbed') === 'true';
            
            if (isGrabbed) {
                // Drop at current position
                if (this.draggedItem && this.draggedItem !== item) {
                    const fromIndex = parseInt(this.draggedItem.dataset.index);
                    const toIndex = parseInt(item.dataset.index);
                    this._reorderItems(fromIndex, toIndex);
                }
                this.draggedItem.setAttribute('aria-grabbed', 'false');
                this.draggedItem = null;
            } else {
                // Pick up
                if (this.draggedItem) {
                    this.draggedItem.setAttribute('aria-grabbed', 'false');
                }
                this.draggedItem = item;
                this.draggedIndex = parseInt(item.dataset.index);
                item.setAttribute('aria-grabbed', 'true');
                this._announce(`Grabbed item ${this.draggedIndex + 1}. Navigate to target position and press Enter or Space.`);
            }
        }
        
        _onReorderClick(e) {
            const btn = e.target.closest('.reorder-btn');
            if (!btn) return;
            
            const item = btn.closest(this.options.itemSelector);
            const currentIndex = parseInt(item.dataset.index);
            const direction = btn.dataset.direction;
            
            const newIndex = direction === 'up' ? currentIndex - 1 : currentIndex + 1;
            
            if (newIndex >= 0 && newIndex < this.container.querySelectorAll(this.options.itemSelector).length) {
                this._reorderItems(currentIndex, newIndex);
                
                // Keep focus on the moved item
                requestAnimationFrame(() => {
                    const items = this.container.querySelectorAll(this.options.itemSelector);
                    items[newIndex].focus();
                });
            }
        }
        
        _reorderItems(fromIndex, toIndex) {
            const items = Array.from(this.container.querySelectorAll(this.options.itemSelector));
            const item = items[fromIndex];
            
            if (fromIndex < toIndex) {
                items[toIndex].after(item);
            } else {
                items[toIndex].before(item);
            }
            
            // Update indices
            this._updateIndices();
            
            // Announce to screen readers
            this._announce(`Moved item from position ${fromIndex + 1} to position ${toIndex + 1}`);
            
            // Callback
            if (this.options.onReorder) {
                this.options.onReorder(fromIndex, toIndex, this._getOrder());
            }
        }
        
        _updateIndices() {
            const items = this.container.querySelectorAll(this.options.itemSelector);
            items.forEach((item, index) => {
                item.dataset.index = index;
                
                // Update reorder buttons
                const upBtn = item.querySelector('[data-direction="up"]');
                const downBtn = item.querySelector('[data-direction="down"]');
                
                if (upBtn) upBtn.disabled = index === 0;
                if (downBtn) downBtn.disabled = index === items.length - 1;
            });
        }
        
        _getOrder() {
            return Array.from(this.container.querySelectorAll(this.options.itemSelector))
                .map(item => item.dataset.id || item.dataset.index);
        }
        
        _announce(message) {
            const announcement = document.createElement('div');
            announcement.setAttribute('role', 'status');
            announcement.setAttribute('aria-live', 'polite');
            announcement.className = 'sr-only';
            announcement.textContent = message;
            
            document.body.appendChild(announcement);
            
            setTimeout(() => announcement.remove(), 1000);
        }
    }
    
    // ==================== Pagination Component ====================
    
    class Pagination {
        constructor(container, options = {}) {
            this.container = typeof container === 'string'
                ? document.querySelector(container)
                : container;
            
            this.options = {
                totalItems: options.totalItems || 0,
                itemsPerPage: options.itemsPerPage || 10,
                currentPage: options.currentPage || 1,
                maxVisiblePages: options.maxVisiblePages || 7,
                onChange: options.onChange || null,
                ...options
            };
            
            this.totalPages = Math.ceil(this.options.totalItems / this.options.itemsPerPage);
            this.currentPage = Math.min(this.options.currentPage, this.totalPages) || 1;
            
            this._render();
        }
        
        _render() {
            if (!this.container) return;
            
            this.container.innerHTML = '';
            this.container.className = 'pagination';
            this.container.setAttribute('role', 'navigation');
            this.container.setAttribute('aria-label', 'Pagination');
            
            if (this.totalPages <= 1) return;
            
            // Previous button
            this._addButton('Previous', this.currentPage - 1, this.currentPage === 1, '←');
            
            // Page buttons
            const pages = this._getVisiblePages();
            pages.forEach(page => {
                if (page === '...') {
                    const ellipsis = document.createElement('span');
                    ellipsis.className = 'pagination-ellipsis';
                    ellipsis.textContent = '...';
                    ellipsis.setAttribute('aria-hidden', 'true');
                    this.container.appendChild(ellipsis);
                } else {
                    this._addButton(
                        `Page ${page}`,
                        page,
                        false,
                        page.toString(),
                        page === this.currentPage
                    );
                }
            });
            
            // Next button
            this._addButton('Next', this.currentPage + 1, this.currentPage === this.totalPages, '→');
            
            // Page info for screen readers
            const info = document.createElement('span');
            info.className = 'sr-only';
            info.setAttribute('aria-live', 'polite');
            info.textContent = `Page ${this.currentPage} of ${this.totalPages}`;
            this.container.appendChild(info);
        }
        
        _addButton(label, page, disabled, text, isActive = false) {
            const btn = document.createElement('button');
            btn.className = 'pagination-btn' + (isActive ? ' active' : '');
            btn.textContent = text;
            btn.disabled = disabled;
            btn.type = 'button';
            
            if (isActive) {
                btn.setAttribute('aria-current', 'page');
            }
            
            btn.setAttribute('aria-label', isActive ? `Current page, ${label}` : label);
            
            if (!disabled) {
                btn.addEventListener('click', () => this.goToPage(page));
            }
            
            this.container.appendChild(btn);
        }
        
        _getVisiblePages() {
            const total = this.totalPages;
            const current = this.currentPage;
            const max = this.options.maxVisiblePages;
            
            if (total <= max) {
                return Array.from({ length: total }, (_, i) => i + 1);
            }
            
            const pages = [];
            const half = Math.floor(max / 2);
            
            let start = Math.max(1, current - half);
            let end = Math.min(total, start + max - 1);
            
            if (end - start < max - 1) {
                start = Math.max(1, end - max + 1);
            }
            
            if (start > 1) {
                pages.push(1);
                if (start > 2) pages.push('...');
            }
            
            for (let i = start; i <= end; i++) {
                pages.push(i);
            }
            
            if (end < total) {
                if (end < total - 1) pages.push('...');
                pages.push(total);
            }
            
            return pages;
        }
        
        goToPage(page) {
            if (page < 1 || page > this.totalPages || page === this.currentPage) return;
            
            this.currentPage = page;
            this._render();
            
            if (this.options.onChange) {
                this.options.onChange(page);
            }
        }
        
        setTotalItems(total) {
            this.options.totalItems = total;
            this.totalPages = Math.ceil(total / this.options.itemsPerPage);
            this.currentPage = Math.min(this.currentPage, this.totalPages) || 1;
            this._render();
        }
    }
    
    // ==================== Virtual Scroll ====================
    
    class VirtualScroll {
        constructor(container, options = {}) {
            this.container = typeof container === 'string'
                ? document.querySelector(container)
                : container;
            
            this.options = {
                itemHeight: options.itemHeight || 40,
                buffer: options.buffer || 5,
                renderItem: options.renderItem || ((item, index) => `<div>${item}</div>`),
                ...options
            };
            
            this.items = [];
            this.scrollTop = 0;
            this.visibleStart = 0;
            this.visibleEnd = 0;
            
            this._init();
        }
        
        _init() {
            if (!this.container) return;
            
            this.container.className = 'virtual-scroll-container';
            this.container.setAttribute('role', 'list');
            this.container.setAttribute('tabindex', '0');
            
            this.content = document.createElement('div');
            this.content.className = 'virtual-scroll-content';
            this.container.appendChild(this.content);
            
            this.container.addEventListener('scroll', this._onScroll.bind(this));
            this.container.addEventListener('keydown', this._onKeydown.bind(this));
            
            this._render();
        }
        
        setItems(items) {
            this.items = items;
            this._render();
        }
        
        _onScroll() {
            this.scrollTop = this.container.scrollTop;
            this._render();
        }
        
        _onKeydown(e) {
            const scrollAmount = this.options.itemHeight * 3;
            
            switch (e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    this.container.scrollTop += scrollAmount;
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    this.container.scrollTop -= scrollAmount;
                    break;
                case 'PageDown':
                    e.preventDefault();
                    this.container.scrollTop += this.container.clientHeight;
                    break;
                case 'PageUp':
                    e.preventDefault();
                    this.container.scrollTop -= this.container.clientHeight;
                    break;
                case 'Home':
                    e.preventDefault();
                    this.container.scrollTop = 0;
                    break;
                case 'End':
                    e.preventDefault();
                    this.container.scrollTop = this.container.scrollHeight;
                    break;
            }
        }
        
        _render() {
            const containerHeight = this.container.clientHeight;
            const itemHeight = this.options.itemHeight;
            const buffer = this.options.buffer;
            
            const totalHeight = this.items.length * itemHeight;
            this.content.style.height = `${totalHeight}px`;
            
            const visibleCount = Math.ceil(containerHeight / itemHeight);
            this.visibleStart = Math.max(0, Math.floor(this.scrollTop / itemHeight) - buffer);
            this.visibleEnd = Math.min(this.items.length, this.visibleStart + visibleCount + buffer * 2);
            
            // Clear and re-render visible items
            this.content.innerHTML = '';
            
            for (let i = this.visibleStart; i < this.visibleEnd; i++) {
                const item = this.items[i];
                const row = document.createElement('div');
                row.className = 'virtual-scroll-row';
                row.style.top = `${i * itemHeight}px`;
                row.style.height = `${itemHeight}px`;
                row.setAttribute('role', 'listitem');
                row.innerHTML = this.options.renderItem(item, i);
                this.content.appendChild(row);
            }
        }
    }
    
    // ==================== Export to global scope ====================
    
    global.FalconOneUI = {
        ToastManager,
        DraggableList,
        Pagination,
        VirtualScroll
    };
    
    // Auto-initialize toast manager
    global.toast = new ToastManager();
    
})(window);
