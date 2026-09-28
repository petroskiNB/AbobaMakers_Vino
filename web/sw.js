const CACHE_PREFIX = 'wine-scanner-shell-';
const CACHE_NAME = `${CACHE_PREFIX}v5`;
const SHELL = ['/', '/static/app.css', '/static/app.js', '/static/pwa.js',
  '/manifest.webmanifest', '/static/icons/icon-192.png', '/static/icons/icon-512.png',
  '/static/icons/maskable-512.png', '/static/icons/apple-touch-icon.png'];

self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE_NAME).then(cache =>
    cache.addAll(SHELL.map(url => new Request(url, {cache: 'reload'})))));
});
self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    for (const name of await caches.keys()) {
      if (name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME) await caches.delete(name);
    }
    await self.clients.claim();
  })());
});
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  // Never cache health checks, uploads, or recognition results.
  if (event.request.method !== 'GET' || url.origin !== self.location.origin ||
      !SHELL.includes(url.pathname)) return;
  event.respondWith((async () => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 4000);
    try {
      const response = await fetch(event.request, {signal: controller.signal, cache: 'no-cache'});
      if (!response.ok) throw new Error('Server unavailable');
      return response;
    } catch {
      const cached = await caches.match(url.pathname, {cacheName: CACHE_NAME});
      return cached || Response.error();
    } finally {
      clearTimeout(timeout);
    }
  })());
});
