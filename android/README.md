# Simon Android (WIP)

This is the Android replacement for the legacy web UI.

## Transport

- REST: `https://<host>:8000/...`
- WebSocket: `wss://<host>:8000/ws`

The app uses TOFU (trust-on-first-use) certificate pinning:
- First connection: show certificate + SHA-256 fingerprint and ask to trust
- On certificate change: block and ask to trust again

## Backend Notes

Voice streaming from Android uses raw PCM:
- Send `AUDIO:PCM16LE:16000` after WS open
- Stream binary frames containing PCM16LE mono @ 16kHz
- Send `CMD:COMMIT_AUDIO` to finalize

