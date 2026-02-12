import './styles/main.css';
import { ViewerApp } from './app/ViewerApp.js';

const app = new ViewerApp();
app.initialize().catch((err) => {
    console.error('Failed to initialize viewer:', err);
    const errorDiv = document.getElementById('webgpu-error');
    const container = document.querySelector('.container') as HTMLElement | null;
    if (errorDiv) errorDiv.style.display = 'flex';
    if (container) container.style.display = 'none';
});
