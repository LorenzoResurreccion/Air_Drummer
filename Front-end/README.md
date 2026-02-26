# AIR DRUMMER Frontend

React.js frontend application for the AIR DRUMMER virtual drumming system.

## Features

- Real-time WebSocket connection to backend
- Low-latency audio playback using Web Audio API
- Bass drum sound triggering from right foot taps
- Connection status monitoring
- Trigger count display

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Add Bass Drum Audio File

Place a bass drum audio file in `public/sounds/bass-drum.wav`

See `public/sounds/README.md` for details on where to find free bass drum sounds.

### 3. Start the Development Server

```bash
npm start
```

The app will open at [http://localhost:3000](http://localhost:3000)

## Usage

1. Ensure the backend is running:
   ```bash
   python Back-end/Body-Tracker.py
   ```

2. Open the frontend in your browser (should auto-open at http://localhost:3000)

3. The app will automatically connect to the WebSocket server at `http://localhost:5000`

4. Position yourself in front of your webcam and tap your right foot to trigger bass drum sounds

## Components

### BassDrumPlayer

The main component that handles:
- WebSocket connection to backend
- Audio loading and playback
- Event handling for bass drum triggers
- Connection status display

**Props:**
- `serverUrl` (optional): WebSocket server URL (default: `http://localhost:5000`)

## Configuration

### Change WebSocket Server URL

Edit `src/App.js` and modify the `serverUrl` prop:

```javascript
<BassDrumPlayer serverUrl="http://your-server:5000" />
```

### Change Audio File

If using a different audio format or filename, edit `src/components/BassDrumPlayer.js` line 38:

```javascript
const response = await fetch('/sounds/your-audio-file.mp3');
```

## Troubleshooting

### Audio Not Playing

1. Check browser console for errors
2. Verify bass drum audio file exists at `public/sounds/bass-drum.wav`
3. Ensure backend WebSocket server is running
4. Check browser audio permissions
5. Try clicking on the page to resume audio context (browser autoplay policy)

### WebSocket Connection Failed

1. Verify backend is running: `python Back-end/Body-Tracker.py`
2. Check backend is using port 5000 (or update `serverUrl` in App.js)
3. Check firewall settings
4. Look for connection errors in browser console

### No Triggers Received

1. Verify tap detection is enabled in backend (not using `--disable-tap-detection`)
2. Check backend console for tap detection logs
3. Ensure your foot is visible to the webcam
4. Try adjusting tap detection sensitivity in backend configuration

## Browser Compatibility

- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support (may require user interaction to start audio)

## Development

### Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm eject` - Eject from Create React App (irreversible)

## Requirements Implemented

This component implements:
- **Requirement 4.2**: Audio player receives trigger events and plays bass drum sound
- **Requirement 4.3**: Sound restart if already playing for responsive feedback
