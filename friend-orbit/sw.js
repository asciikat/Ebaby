// Friend Orbit offline support. Bump VERSION whenever any app file changes
// so installed copies pick up the new version on their next launch.
const VERSION = 'friend-orbit-v3';
const FONTS = 'friend-orbit-fonts';
const SHELL = [
  './',
  './index.html',
  './manifest.webmanifest',
  './icons/icon.svg',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './icons/maskable-512.png',
  './icons/apple-touch-icon.png',
];

self.addEventListener('install', event => {
  event.waitUntil(caches.open(VERSION).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k.startsWith('friend-orbit-') && k !== VERSION && k !== FONTS).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin === 'https://fonts.googleapis.com' || url.origin === 'https://fonts.gstatic.com'){
    event.respondWith(cacheFirst(req));
    return;
  }
  if (url.origin !== self.location.origin) return;
  event.respondWith(staleWhileRevalidate(event));
});

// Serve the saved copy instantly (works offline), refresh it in the background.
async function staleWhileRevalidate(event){
  const req = event.request;
  const cache = await caches.open(VERSION);
  const key = req.mode === 'navigate' ? './index.html' : req;
  const cached = await cache.match(key, {ignoreSearch: true});
  const network = fetch(req)
    .then(res => { if (res && res.ok && res.type === 'basic') cache.put(key, res.clone()); return res; })
    .catch(() => null);
  if (cached){ event.waitUntil(network); return cached; }
  return (await network) || new Response('Friend Orbit is offline and has not been saved on this device yet.', {status: 503, headers: {'Content-Type': 'text/plain; charset=utf-8'}});
}

async function cacheFirst(req){
  const cache = await caches.open(FONTS);
  const hit = await cache.match(req);
  if (hit) return hit;
  try {
    const res = await fetch(req);
    if (res && (res.ok || res.type === 'opaque')) cache.put(req, res.clone());
    return res;
  } catch (e) {
    return Response.error();
  }
}
