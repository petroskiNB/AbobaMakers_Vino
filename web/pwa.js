(() => {
  const button = document.getElementById('install');
  const help = document.getElementById('install-help');
  let prompt;
  const standalone = () => matchMedia('(display-mode: standalone)').matches || navigator.standalone;
  const installed = () => {
    button.hidden = true;
    help.textContent = 'Сканер открыт как приложение. Для распознавания нужен доступ к серверу.';
  };
  if (standalone()) installed();
  else if (!window.isSecureContext) {
    help.textContent = 'Для установки на телефон откройте сайт по HTTPS. По этому адресу можно пользоваться веб-версией.';
  } else if (/iPad|iPhone|iPod/.test(navigator.userAgent)) {
    help.textContent = 'На iPhone: откройте меню «Поделиться» → «На экран Домой». Если пункта нет, откройте сайт в Safari.';
  }
  window.addEventListener('beforeinstallprompt', event => {
    event.preventDefault();
    prompt = event;
    if (!standalone()) button.hidden = false;
  });
  button.addEventListener('click', async () => {
    if (!prompt) return;
    button.disabled = true;
    try {
      await prompt.prompt();
      await prompt.userChoice;
    } catch {
      help.textContent = 'Откройте меню браузера и выберите установку приложения.';
    } finally {
      prompt = null;
      button.hidden = true;
      button.disabled = false;
    }
  });
  window.addEventListener('appinstalled', installed);
  if ('serviceWorker' in navigator && window.isSecureContext) {
    navigator.serviceWorker.register('/sw.js').catch(() => {
      help.textContent = 'Не удалось подготовить работу без сети. Обновите страницу при доступном сервере.';
    });
  }
})();
