import './styles/main.css';
import { disposeCrashLogger, initCrashLogger } from './app/CrashLogger.js';
import { ViewerApp } from './app/ViewerApp.js';

initCrashLogger();

const app = new ViewerApp();
app.initialize().catch((err) => {
    console.error('Failed to initialize viewer:', err);
    const errorDiv = document.getElementById('webgpu-error');
    const container = document.querySelector('.container') as HTMLElement | null;
    if (errorDiv) errorDiv.style.display = 'flex';
    if (container) container.style.display = 'none';
});

// Vite HMR: dispose old instance so event listeners don't leak
if (import.meta.hot) {
    import.meta.hot.dispose(() => {
        disposeCrashLogger();
        app.dispose();
    });
}
