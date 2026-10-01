# Friend Orbit

A calm, shame-free friends list for ADHD and CPTSD brains. You're the sun,
your people orbit you, drift outward slowly, and never fall out of your sky.

## Use it as an app

The app lives at **https://asciikat.github.io/Ebaby/friend-orbit/**.

- **Windows / Mac (Chrome or Edge):** open the link, then click **Install app**
  at the bottom of the page (or the install icon in the address bar). It gets
  its own window and a Start menu / Dock icon.
- **Android (Chrome):** open the link, tap **Install app**, or use the menu
  and pick **Add to Home screen**.
- **iPhone / iPad (Safari):** open the link, tap **Share**, then
  **Add to Home Screen**.

After the first visit it works offline. Long-press (or right-click) the icon
for the **Hard day mode** and **Add someone** shortcuts.

You can also just double-click `index.html` to open it in a browser. It works
the same, minus offline install.

## Your data

Everything is saved on the device you use it on. Nothing is uploaded. Use
**Back up** now and then, and **Restore a backup** to move your people to
another device (or over from the claude.ai version).

## Your own ambient sound

The ambient button (the wave in the top bar) plays a soft synth drone. To use
your own track instead, add MP3 files here:

- `audio/ambient.mp3` plays normally.
- `audio/hard-day.mp3` (optional) plays in hard day mode. Without it, the main
  track just gets quieter.

Tracks loop with a crossfade (up to 6 seconds, never more than a quarter of the track),
so they don't need to loop perfectly. Longer tracks repeat less noticeably. Keep
each one under about 8 MB. Bump `VERSION` in `sw.js` after adding or
changing a track.

## Hosting

GitHub Pages publishes the repo straight from a branch
(**Settings → Pages → Build and deployment → Deploy from a branch**, folder
`/ (root)`), so this folder is served at `/Ebaby/friend-orbit/`. Point that
branch setting at `rebuild-v3` once this folder is merged there.

## Changing the app

Edit `index.html`, then bump `VERSION` in `sw.js` so installed copies pick
up the change on their next launch.
