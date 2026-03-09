// Language Switcher for Enigmata Website

// Default language is English
let currentLanguage = localStorage.getItem('language') || 'en';

// Initialize language on page load
document.addEventListener('DOMContentLoaded', function() {
    // Set initial language based on localStorage or default to English
    setLanguage(currentLanguage);
    
    // Set the language toggle button state
    const languageToggle = document.getElementById('language-toggle');
    if (languageToggle) {
        languageToggle.textContent = currentLanguage === 'en' ? '中文' : 'English';
    }
    
    // Add event listener to language toggle button
    if (languageToggle) {
        languageToggle.addEventListener('click', function() {
            // Toggle language
            currentLanguage = currentLanguage === 'en' ? 'zh' : 'en';
            
            // Save to localStorage
            localStorage.setItem('language', currentLanguage);
            
            // Update button text
            languageToggle.textContent = currentLanguage === 'en' ? '中文' : 'English';
            
            // Apply language change
            setLanguage(currentLanguage);
        });
    }
});

// Function to set the language
function setLanguage(lang) {
    // Hide all language elements except the html element
    document.querySelectorAll('[lang]').forEach(elem => {
        // 排除 html 元素
        if (elem.tagName.toLowerCase() !== 'html') {
            elem.style.display = 'none';
        }
    });

    // Show elements for the current language
    document.querySelectorAll(`[lang="${lang}"]`).forEach(elem => {
        if (elem.tagName.toLowerCase() !== 'html') {
            elem.style.display = '';
        }
    });

    // Update html lang attribute
    document.documentElement.lang = lang;
}