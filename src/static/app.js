document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const papersList = document.getElementById('papersList');
    const btnRefresh = document.getElementById('btnRefresh');

    const activePaperTitle = document.getElementById('activePaperTitle');
    const activePaperBadge = document.getElementById('activePaperBadge');
    const activePaperStats = document.getElementById('activePaperStats');
    const statPages = document.getElementById('statPages');
    const statChunks = document.getElementById('statChunks');
    const statFigures = document.getElementById('statFigures');

    const processingCard = document.getElementById('processingCard');
    const processingPercent = document.getElementById('processingPercent');
    const progressBarFill = document.getElementById('progressBarFill');
    const processingMessage = document.getElementById('processingMessage');

    const messagesArea = document.getElementById('messagesArea');
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const btnSend = document.getElementById('btnSend');

    // State
    let activePaperId = null;
    let papersMap = {};
    let pollInterval = null;

    // --- Drag & Drop Handlers ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) handleFileUpload(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    btnRefresh.addEventListener('click', loadPapersList);

    // --- API Calls ---

    async function loadPapersList() {
        try {
            const res = await fetch('/api/v1/papers');
            if (!res.ok) return;
            const papers = await res.json();
            
            papersMap = {};
            papers.forEach(p => papersMap[p.paper_id] = p);
            
            renderPapersList(papers);

            // Auto-select latest paper if none active
            if (!activePaperId && papers.length > 0) {
                selectPaper(papers[papers.length - 1].paper_id);
            }
        } catch (err) {
            console.error('Error fetching papers list:', err);
        }
    }

    function renderPapersList(papers) {
        if (papers.length === 0) {
            papersList.innerHTML = '<div class="empty-state">No papers uploaded yet</div>';
            return;
        }

        papersList.innerHTML = '';
        papers.forEach(p => {
            const item = document.createElement('div');
            item.className = `paper-item ${p.paper_id === activePaperId ? 'active' : ''}`;
            item.onclick = (e) => {
                if (e.target.classList.contains('btn-delete-paper')) return;
                selectPaper(p.paper_id);
            };

            item.innerHTML = `
                <div class="paper-item-top">
                    <span class="paper-name">${escapeHtml(p.filename)}</span>
                    <button class="btn-delete-paper" title="Delete paper" data-id="${p.paper_id}">✕</button>
                </div>
                <div class="paper-item-bottom">
                    <span class="badge ${p.status}">${p.status}</span>
                    <span>${p.pages || 0} pages</span>
                </div>
            `;

            const delBtn = item.querySelector('.btn-delete-paper');
            delBtn.onclick = (e) => {
                e.stopPropagation();
                deletePaper(p.paper_id);
            };

            papersList.appendChild(item);
        });
    }

    async function handleFileUpload(file) {
        if (!file.name.toLowerCase().endswith('.pdf')) {
            alert('Please select a PDF document.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            showProcessingBanner('Uploading PDF...', 5);
            
            const res = await fetch('/api/v1/papers/upload', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || 'Upload failed');
            }

            const data = await res.json();
            activePaperId = data.paper_id;

            await loadPapersList();
            startPollingStatus(data.paper_id);

        } catch (err) {
            alert(`Upload Error: ${err.message}`);
            hideProcessingBanner();
        }
    }

    function selectPaper(paperId) {
        activePaperId = paperId;
        const paper = papersMap[paperId];
        
        loadPapersList();

        if (!paper) return;

        activePaperTitle.textContent = paper.filename;
        activePaperBadge.style.display = 'inline-block';
        activePaperBadge.className = `badge ${paper.status}`;
        activePaperBadge.textContent = paper.status;

        if (paper.status === 'ready') {
            activePaperStats.style.display = 'flex';
            statPages.textContent = paper.pages || 0;
            statChunks.textContent = paper.chunks || 0;
            statFigures.textContent = paper.images || 0;
            
            enableChat();
            hideProcessingBanner();
        } else if (paper.status === 'processing') {
            activePaperStats.style.display = 'none';
            disableChat();
            startPollingStatus(paperId);
        } else {
            activePaperStats.style.display = 'none';
            disableChat();
            hideProcessingBanner();
        }
    }

    function startPollingStatus(paperId) {
        if (pollInterval) clearInterval(pollInterval);

        const checkStatus = async () => {
            try {
                const res = await fetch(`/api/v1/papers/${paperId}/status`);
                if (!res.ok) return;
                const statusData = await res.json();

                // Update processing card
                showProcessingBanner(statusData.message || 'Processing paper...', statusData.progress || 10);

                // Update paper object in state
                papersMap[paperId] = statusData;

                if (statusData.status === 'ready') {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    await loadPapersList();
                    selectPaper(paperId);
                } else if (statusData.status === 'failed') {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    alert(`Processing failed: ${statusData.message || statusData.error}`);
                    await loadPapersList();
                    selectPaper(paperId);
                }
            } catch (err) {
                console.error('Polling error:', err);
            }
        };

        checkStatus();
        pollInterval = setInterval(checkStatus, 2000);
    }

    async function deletePaper(paperId) {
        if (!confirm('Are you sure you want to delete this paper and all its vector index records?')) return;

        try {
            const res = await fetch(`/api/v1/papers/${paperId}`, { method: 'DELETE' });
            if (!res.ok) throw new Error('Deletion failed');

            if (activePaperId === paperId) {
                activePaperId = null;
                resetWorkspace();
            }
            await loadPapersList();
        } catch (err) {
            alert(`Error: ${err.message}`);
        }
    }

    // --- Chat Logic ---

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = userInput.value.trim();
        if (!query || !activePaperId) return;

        userInput.value = '';
        appendMessage('user', query);

        // Append loading assistant message
        const loadingDiv = appendMessage('assistant', 'Searching paper context and generating answer...', true);

        try {
            const res = await fetch(`/api/v1/papers/${activePaperId}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, n_results: 5 })
            });

            const data = await res.json();

            if (!res.ok) {
                loadingDiv.innerHTML = `<div class="message-bubble error">Error: ${escapeHtml(data.detail || 'Query failed')}</div>`;
                return;
            }

            renderAssistantResponse(loadingDiv, data.answer, data.sources);
        } catch (err) {
            loadingDiv.innerHTML = `<div class="message-bubble error">Error: ${escapeHtml(err.message)}</div>`;
        }
    });

    function appendMessage(role, text, isLoading = false) {
        // Hide welcome card if present
        const welcome = messagesArea.querySelector('.welcome-card');
        if (welcome) welcome.style.display = 'none';

        const row = document.createElement('div');
        row.className = `message-row ${role}`;

        const bubble = document.createElement('div');
        bubble.className = `message-bubble ${isLoading ? 'loading' : ''}`;
        bubble.textContent = text;

        row.appendChild(bubble);
        messagesArea.appendChild(row);
        messagesArea.scrollTop = messagesArea.scrollHeight;

        return row;
    }

    function renderAssistantResponse(rowElement, answerMarkdown, sources) {
        const bubble = rowElement.querySelector('.message-bubble');
        bubble.classList.remove('loading');

        // Render Markdown
        let parsedHtml = marked.parse(answerMarkdown || '');
        bubble.innerHTML = parsedHtml;

        // Render LaTeX if Katex available
        if (window.renderMathInElement) {
            renderMathInElement(bubble, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '\\[', right: '\\]', display: true}
                ]
            });
        }

        // Render source badges if any
        if (sources && sources.length > 0) {
            const sourcesContainer = document.createElement('div');
            sourcesContainer.className = 'sources-container';

            const uniquePages = [...new Set(sources.map(s => s.page))].sort((a, b) => a - b);
            uniquePages.forEach(p => {
                const badge = document.createElement('span');
                badge.className = 'source-badge';
                badge.textContent = `Page ${p}`;
                badge.title = `Source context from Page ${p}`;
                sourcesContainer.appendChild(badge);
            });

            rowElement.appendChild(sourcesContainer);
        }

        messagesArea.scrollTop = messagesArea.scrollHeight;
    }

    function showProcessingBanner(msg, pct) {
        processingCard.style.display = 'block';
        processingMessage.textContent = msg;
        processingPercent.textContent = `${pct}%`;
        progressBarFill.style.width = `${pct}%`;
    }

    function hideProcessingBanner() {
        processingCard.style.display = 'none';
    }

    function enableChat() {
        userInput.disabled = false;
        btnSend.disabled = false;
        userInput.placeholder = `Ask a question about ${papersMap[activePaperId]?.filename || 'this paper'}...`;
    }

    function disableChat() {
        userInput.disabled = true;
        btnSend.disabled = true;
        userInput.placeholder = 'Please wait for paper to finish processing...';
    }

    function resetWorkspace() {
        activePaperTitle.textContent = 'Select or Upload a Research Paper';
        activePaperBadge.style.display = 'none';
        activePaperStats.style.display = 'none';
        disableChat();
        hideProcessingBanner();
        messagesArea.innerHTML = `
            <div class="welcome-card">
                <div class="welcome-icon">📄🔍</div>
                <h2>Multimodal Research Paper QA</h2>
                <p>Upload a PDF research paper on the left to extract text, tables, figures, and start asking questions grounded directly in the paper.</p>
            </div>
        `;
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    // String endswith polyfill check
    if (!String.prototype.endswith) {
        String.prototype.endswith = function(suffix) {
            return this.indexOf(suffix, this.length - suffix.length) !== -1;
        };
    }

    // Initial Load
    loadPapersList();
});
