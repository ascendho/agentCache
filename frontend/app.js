document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const accessCodeInput = document.getElementById('access-code');
    const saveKeyBtn = document.getElementById('save-key-btn');
    const quickActions = document.getElementById('quick-actions');
    const quickPills = document.querySelectorAll('.quick-pill');

    // API Backend url - when deploying, switch to your remote url e.g. https://my-agent.zeabur.app/chat
    const API_URL = 'http://127.0.0.1:8000/chat';

    // Access Key Persistence
    let finalAccessKey = localStorage.getItem('agent_access_key') || '';
    if (finalAccessKey) {
        accessCodeInput.value = finalAccessKey;
        accessCodeInput.disabled = true;
        saveKeyBtn.innerHTML = '已保存 <span class="pl-1 opacity-80 cursor-pointer">🔒</span>';
        saveKeyBtn.classList.add('bg-black', 'text-white');
    }

    saveKeyBtn.addEventListener('click', () => {
        const val = accessCodeInput.value.trim();
        if (accessCodeInput.disabled) {
            // Unlock mode
            accessCodeInput.disabled = false;
            accessCodeInput.focus();
            saveKeyBtn.innerHTML = '保存 <span class="pl-1">✅</span>';
        } else if (val) {
            // Save mode
            finalAccessKey = val;
            localStorage.setItem('agent_access_key', val);
            accessCodeInput.disabled = true;
            saveKeyBtn.innerHTML = '已保存 <span class="pl-1 opacity-80 cursor-pointer">🔒</span>';
        } else {
            alert("请先输入有效的邀请码");
        }
    });

    // Handle quick pills (Persistent)
    quickPills.forEach(pill => {
        pill.addEventListener('click', (e) => {
            const presetTxt = e.target.textContent.trim();
            handleSend(presetTxt);
        });
    });

    // Helper: Scroll to bottom
    const scrollToBottom = () => {
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    // Helper: Escape HTML
    const escapeHTML = (str) => {
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag] || tag)
        );
    };

    // Helper: Format Server Response
    const formatText = (text) => {
        return escapeHTML(text).replace(/\n/g, '<br/>');
    };

    // Helper: Create User Message Element
    const appendUserMessage = (text) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'flex w-full mt-4 space-x-4 max-w-2xl ml-auto justify-end';
        msgDiv.innerHTML = `
            <div class="bg-gray-100 text-gray-900 border border-gray-200/60 p-4 rounded-2xl rounded-tr-sm shadow-sm text-[15px] break-words whitespace-pre-wrap leading-relaxed">
                ${escapeHTML(text)}
            </div>
            <div class="flex-shrink-0 mt-1 h-9 w-9 rounded-full bg-gray-200 flex items-center justify-center shadow-sm">
                <span class="text-gray-600 font-medium text-sm">U</span>
            </div>
        `;
        chatBox.appendChild(msgDiv);
        scrollToBottom();
    };

    // Helper: Create Bot Message Element
    const appendBotMessage = (text, latency = null, cacheHit = false, intercepted = false) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'flex w-full mt-4 space-x-4 max-w-2xl';

        let badgeHtml = '';
        if (latency) {
            let badgeStyle = '';
            if (cacheHit) {
                badgeStyle = '<span class="px-2.5 py-1 bg-green-50 text-green-700 border border-green-200 rounded-full font-medium tracking-wide">⚡ Cache Hit</span>';
            } else if (intercepted) {
                badgeStyle = '<span class="px-2.5 py-1 bg-yellow-50 text-yellow-700 border border-yellow-200 rounded-full font-medium tracking-wide shadow-sm">🛡️ Zero-Layer Intercept</span>';
            } else {
                badgeStyle = '<span class="px-2.5 py-1 bg-blue-50 text-blue-700 border border-blue-200 rounded-full font-medium tracking-wide">🧠 LLM Analysis</span>';
            }

            badgeHtml = `
                <div class="mt-2 flex items-center space-x-2 text-[11px] uppercase">
                    ${badgeStyle}
                    <span class="text-gray-400 font-medium tracking-wider">⏱ ${latency}ms</span>
                </div>
            `;
        }

        msgDiv.innerHTML = `
            <div class="flex-shrink-0 mt-1 h-9 w-9 rounded-full bg-gradient-to-tr from-gray-700 to-gray-900 flex items-center justify-center shadow-md">
                <span class="text-white font-medium text-sm">A</span>
            </div>
            <div class="flex flex-col flex-1 mr-8">
                <div class="bg-white p-4 rounded-2xl rounded-tl-sm shadow-sm border ${intercepted ? 'border-yellow-200/60' : 'border-gray-100'} text-gray-800 text-[15px] whitespace-pre-wrap leading-relaxed model-response font-medium">
                    ${formatText(text)}
                </div>
                ${badgeHtml}
            </div>
        `;
        chatBox.appendChild(msgDiv);
        scrollToBottom();
    };

    // Helper: Create Error Message Element with Retry
    const appendErrorMessage = (message, failedQuery) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'flex w-full mt-4 space-x-4 max-w-2xl';
        msgDiv.innerHTML = `
            <div class="flex-shrink-0 mt-1 h-9 w-9 rounded-full bg-red-50 border border-red-100 flex items-center justify-center shadow-sm">
                <span class="text-red-500 font-bold text-sm">!</span>
            </div>
            <div class="flex flex-col flex-1 mr-8">
                <div class="bg-red-50 p-4 rounded-2xl rounded-tl-sm border border-red-100/60 text-red-800 text-[15px] leading-relaxed shadow-sm">
                    <p class="font-medium mb-2">${escapeHTML(message)}</p>
                    <button onclick="window.retryLastRequest()" class="inline-flex items-center space-x-1 text-sm bg-white hover:bg-gray-50 border border-red-200 text-red-600 px-3 py-1.5 rounded-lg transition-colors shadow-sm focus:outline-none">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-4 h-4"><path stroke-linecap="round" stroke-linejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" /></svg>
                        <span>重试 (Retry)</span>
                    </button>
                </div>
            </div>
        `;
        chatBox.appendChild(msgDiv);
        scrollToBottom();
    };

    // Helper: Create Loading Indicator
    const createLoadingIndicator = () => {
        const id = 'loader-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'flex w-full mt-4 space-x-4 max-w-2xl';
        msgDiv.id = id;
        msgDiv.innerHTML = `
            <div class="flex-shrink-0 mt-1 h-9 w-9 rounded-full bg-gradient-to-tr from-gray-700 to-gray-900 flex items-center justify-center shadow-md">
                <span class="text-white font-medium text-sm">A</span>
            </div>
            <div class="flex flex-col flex-1 mr-8">
                <div class="bg-white p-4 py-5 rounded-2xl rounded-tl-sm shadow-sm border border-gray-100 flex space-x-1.5 items-center">
                    <div class="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
                    <div class="w-2 h-2 bg-gray-400 rounded-full animate-pulse delay-75"></div>
                    <div class="w-2 h-2 bg-gray-400 rounded-full animate-pulse delay-150"></div>
                </div>
            </div>
        `;
        chatBox.appendChild(msgDiv);
        scrollToBottom();
        return id;
    };

    const removeLoadingIndicator = (id) => {
        const el = document.getElementById(id);
        if (el) el.remove();
    };

    // Retry global binding
    window.lastAttemptedQuery = "";
    window.retryLastRequest = () => {
        if(window.lastAttemptedQuery) {
            handleSend(window.lastAttemptedQuery, true);
        }
    };

    // Action: Handle Send
    const handleSend = async (overrideText = null, isRetry = false) => {
        const query = typeof overrideText === 'string' ? overrideText : userInput.value.trim();
        const accessCode = finalAccessKey || accessCodeInput.value.trim();

        if (!query) return;

        if (!accessCode) {
            alert('🔒 请先在右上角输入专属【邀请码】并点击【保存】！\n💡 您可以在我的简历中找到它。');
            accessCodeInput.focus();
            return;
        }

        // Add user msg if not retrying
        if(!isRetry) {
             appendUserMessage(query);
        }
        
        window.lastAttemptedQuery = query; // Save for retry
        
        if (typeof overrideText !== 'string') {
            userInput.value = '';
        }
        
        // Show loader
        const loaderId = createLoadingIndicator();
        sendBtn.disabled = true;

        try {
            // Request backend
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    access_code: accessCode
                })
            });

            if (!response.ok) {
                if (response.status === 401) {
                    throw new Error('❌ 邀请码错误！请求被拦截，未消耗 Tokens。');
                } else if (response.status === 429) {
                    throw new Error('⚠️ API 访问过于频繁。防刷限制已触发，请稍后再试。');
                }
                const errData = await response.json();
                throw new Error(errData.detail || 'The API request failed.');
            }

            const data = await response.json();
            removeLoadingIndicator(loaderId);
            
            // Build response
            appendBotMessage(data.answer, data.latency_ms, data.cache_hit, data.intercepted);
            
            // Clear last attempt on success
            window.lastAttemptedQuery = "";

        } catch (error) {
            removeLoadingIndicator(loaderId);
            // Render specific retry UI
            if(error.message === 'Failed to fetch') {
                 appendErrorMessage("网络请求失败，后端服务可能未启动或网络异常。", query);
            } else {
                 appendErrorMessage(error.message, query);
            }
        } finally {
            sendBtn.disabled = false;
            userInput.focus();
        }
    };

    // Listeners
    sendBtn.addEventListener('click', () => handleSend());
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSend();
    });
});
