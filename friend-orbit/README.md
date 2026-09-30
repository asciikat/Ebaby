# Friend Orbit

A calm, shame-free friends list for ADHD and CPTSD brains. You're the sun,
your people orbit you, drift outward slowly, and never fall out of your sky.

## Use it as an app

Once GitHub Pages is on, the app lives at **https://asciikat.github.io/Ebaby/**.

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

## Turning on GitHub Pages (one time)

Repo **Settings → Pages → Build and deployment → Source: GitHub Actions**.
The `Deploy Friend Orbit` workflow then publishes this folder whenever it
changes on `rebuild-v3`. You can also run it by hand from the **Actions** tab.

## Changing the app

Edit `index.html`, then bump `VERSION` in `sw.js` so installed copies pick
up the change on their next launch.
