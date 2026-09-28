document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const textInput = document.getElementById('text-input');
    const languageSelect = document.getElementById('language');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = document.getElementById('loading-spinner');
    
    const resultSection = document.getElementById('result-section');
    const verdictContainer = document.getElementById('verdict-container');
    const detectedLanguage = document.getElementById('detected-language');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceValue = document.getElementById('confidence-value');
    const processingTime = document.getElementById('processing-time');
    const requestId = document.getElementById('request-id');
    
    const toast = document.getElementById('error-toast');
    const errorMessage = document.getElementById('error-message');
    
    let isSubmitting = false;

    // --- Icons ---
    const alertIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`;
    const shieldIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="M9 12l2 2 4-4"></path></svg>`;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (isSubmitting) return;
        
        const text = textInput.value.trim();
        if (!text) return;
        
        const language = languageSelect.value;
        
        setLoadingState(true);
        hideError();
        
        try {
            const response = await fetch('/api/v1/predict/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    language: language
                })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || data.detail || 'Failed to analyze text');
            }
            
            renderResult(data, language);
            
        } catch (error) {
            console.error('Prediction error:', error);
            showError(error.message || 'Unable to connect to the server.');
            resultSection.classList.add('hidden');
        } finally {
            setLoadingState(false);
        }
    });
    
    function setLoadingState(isLoading) {
        isSubmitting = isLoading;
        submitBtn.disabled = isLoading;
        
        if (isLoading) {
            btnText.classList.add('hidden');
            spinner.classList.remove('hidden');
            // Hide result section during loading for better UX
            resultSection.classList.add('hidden');
        } else {
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
        }
    }
    
    function renderResult(data, selectedLang = 'auto') {
        const result = data.result;
        
        // Remove old classes
        confidenceFill.classList.remove('hate-gradient', 'safe-gradient');
        
        const isHate = result.label.toUpperCase() === 'HATE';
        const confidencePct = (result.confidence * 100).toFixed(1);
        
        if (isHate) {
            verdictContainer.innerHTML = `
                <div class="verdict is-hate">
                    <div class="verdict-icon">${alertIcon}</div>
                    <div class="verdict-text">
                        <div class="verdict-title">Hate Speech Detected</div>
                        <div class="verdict-desc">This content has been flagged as potentially offensive or hateful.</div>
                    </div>
                </div>
            `;
            confidenceFill.classList.add('hate-gradient');
        } else {
            verdictContainer.innerHTML = `
                <div class="verdict is-safe">
                    <div class="verdict-icon">${shieldIcon}</div>
                    <div class="verdict-text">
                        <div class="verdict-title">No Hate Speech Detected</div>
                        <div class="verdict-desc">This content appears to be safe and complies with community standards.</div>
                    </div>
                </div>
            `;
            confidenceFill.classList.add('safe-gradient');
        }
        
        // Set metrics
        // Add a small timeout to allow CSS animation to trigger
        setTimeout(() => {
            confidenceFill.style.width = `${confidencePct}%`;
        }, 50);
        
        confidenceValue.textContent = `${confidencePct}%`;
        
        const langCode = result.language_detected || (selectedLang && selectedLang !== 'auto' ? selectedLang : 'Auto');
        detectedLanguage.textContent = `Lang: ${langCode.toUpperCase()}`;
        
        processingTime.textContent = `${result.processing_time_ms.toFixed(1)} ms`;
        requestId.textContent = data.request_id.split('-')[0] + '...';
        
        // Render SHAP Explainability if present
        const shapContainer = document.getElementById('shap-container');
        const shapBase = document.getElementById('shap-base');
        const shapSum = document.getElementById('shap-sum');
        const shapPosList = document.getElementById('shap-pos-list');
        const shapNegList = document.getElementById('shap-neg-list');

        if (result.explanation && shapContainer) {
            const exp = result.explanation;
            shapBase.textContent = `Base: ${exp.base_value.toFixed(4)}`;
            shapSum.textContent = `${exp.shap_sum > 0 ? '+' : ''}${exp.shap_sum.toFixed(4)}`;

            shapPosList.innerHTML = (exp.top_positive_features || []).map(f => 
                `<li class="shap-item pos"><span>Dim #${f.dimension}</span><span>+${f.impact.toFixed(4)}</span></li>`
            ).join('');

            shapNegList.innerHTML = (exp.top_negative_features || []).map(f => 
                `<li class="shap-item neg"><span>Dim #${f.dimension}</span><span>${f.impact.toFixed(4)}</span></li>`
            ).join('');

            shapContainer.classList.remove('hidden');
        } else if (shapContainer) {
            shapContainer.classList.add('hidden');
        }

        // Show section
        resultSection.classList.remove('hidden');
        
        // Scroll to result on mobile
        if (window.innerWidth < 640) {
            setTimeout(() => {
                resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }, 100);
        }
    }
    
    let toastTimeout;
    function showError(message) {
        errorMessage.textContent = message;
        toast.classList.remove('hidden');
        
        clearTimeout(toastTimeout);
        toastTimeout = setTimeout(() => {
            hideError();
        }, 5000);
    }
    
    function hideError() {
        toast.classList.add('hidden');
    }
    
    // Auto-resize textarea
    textInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if(this.value.trim() === '') {
            this.style.height = '150px';
        }
    });
});
