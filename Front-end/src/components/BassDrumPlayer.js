import { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';

/**
 * BassDrumPlayer Component
 * 
 * Connects to the backend WebSocket server and plays bass drum sounds
 * when tap events are received. Uses Web Audio API for low-latency
 * audio playback with the ability to restart sounds if already playing.
 * 
 * Requirements: 4.2, 4.3
 */
const BassDrumPlayer = ({ serverUrl = 'http://localhost:5001' }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [triggerCount, setTriggerCount] = useState(0);
  const [lastTapType, setLastTapType] = useState(null);
  const [error, setError] = useState(null);
  
  // Refs for audio context and buffer
  const audioContextRef = useRef(null);
  const audioBufferRef = useRef(null);
  const currentSourceRef = useRef(null);
  const socketRef = useRef(null);

  /**
   * Initialize Web Audio API context and load bass drum sound
   */
  useEffect(() => {
    // Create audio context (suspended until user interaction due to browser policies)
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    audioContextRef.current = new AudioContext();

    // Load bass drum audio file
    const loadAudio = async () => {
      try {
        // Fetch audio file from public directory
        const response = await fetch('/sounds/bass-drum.wav');
        
        if (!response.ok) {
          throw new Error(`Failed to load audio file: ${response.status} ${response.statusText}`);
        }
        
        const arrayBuffer = await response.arrayBuffer();
        
        // Decode audio data
        const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
        audioBufferRef.current = audioBuffer;
        
        console.log('Bass drum audio loaded successfully');
      } catch (err) {
        console.error('Error loading bass drum audio:', err);
        setError(`Failed to load audio: ${err.message}`);
      }
    };

    loadAudio();

    // Cleanup audio context on unmount
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  /**
   * Set up WebSocket connection to backend
   */
  useEffect(() => {
    // Create socket connection
    const socket = io(serverUrl, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5
    });

    socketRef.current = socket;

    // Connection event handlers
    socket.on('connect', () => {
      console.log('Connected to WebSocket server');
      setIsConnected(true);
      setError(null);
      
      // Resume audio context on first connection (handles browser autoplay policy)
      if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
        audioContextRef.current.resume();
      }
    });

    socket.on('disconnect', () => {
      console.log('Disconnected from WebSocket server');
      setIsConnected(false);
    });

    socket.on('connect_error', (err) => {
      console.error('WebSocket connection error:', err);
      setError(`Connection error: ${err.message}`);
    });

    // Listen for bass drum trigger events
    socket.on('bass_drum_trigger', (data) => {
      console.log('Bass drum trigger received:', data);
      playBassDrum(data);
    });

    // Cleanup on unmount
    return () => {
      socket.disconnect();
    };
  }, [serverUrl]);

  /**
   * Play bass drum sound using Web Audio API
   * 
   * If a sound is already playing, it will be stopped and restarted
   * from the beginning to ensure responsive feedback for rapid taps.
   * 
   * @param {Object} tapData - Tap event data from backend
   */
  const playBassDrum = (tapData) => {
    // Ensure audio context and buffer are ready
    if (!audioContextRef.current || !audioBufferRef.current) {
      console.warn('Audio not ready yet');
      return;
    }

    // Resume audio context if suspended (browser autoplay policy)
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }

    try {
      // Stop currently playing sound if any (allows restart)
      if (currentSourceRef.current) {
        try {
          currentSourceRef.current.stop();
        } catch (e) {
          // Ignore errors if source already stopped
        }
      }

      // Create new audio source
      const source = audioContextRef.current.createBufferSource();
      source.buffer = audioBufferRef.current;
      source.connect(audioContextRef.current.destination);

      // Optional: Apply velocity-based volume control in future
      // const gainNode = audioContextRef.current.createGain();
      // gainNode.gain.value = Math.min(1.0, tapData.velocity || 0.5);
      // source.connect(gainNode);
      // gainNode.connect(audioContextRef.current.destination);

      // Play sound
      source.start(0);
      currentSourceRef.current = source;

      // Update UI state
      setTriggerCount(prev => prev + 1);
      setLastTapType(tapData.tap_type);

      // Clear source reference when sound finishes
      source.onended = () => {
        if (currentSourceRef.current === source) {
          currentSourceRef.current = null;
        }
      };

    } catch (err) {
      console.error('Error playing bass drum:', err);
      setError(`Playback error: ${err.message}`);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.statusPanel}>
        <h2 style={styles.title}>Bass Drum Player</h2>
        
        <div style={styles.statusItem}>
          <span style={styles.label}>Connection:</span>
          <span style={{
            ...styles.value,
            color: isConnected ? '#4CAF50' : '#f44336'
          }}>
            {isConnected ? '● Connected' : '○ Disconnected'}
          </span>
        </div>

        <div style={styles.statusItem}>
          <span style={styles.label}>Triggers:</span>
          <span style={styles.value}>{triggerCount}</span>
        </div>

        {lastTapType && (
          <div style={styles.statusItem}>
            <span style={styles.label}>Last Tap:</span>
            <span style={styles.value}>{lastTapType}</span>
          </div>
        )}

        {error && (
          <div style={styles.error}>
            <strong>Error:</strong> {error}
          </div>
        )}

        {!isConnected && !error && (
          <div style={styles.info}>
            Waiting for backend connection...
          </div>
        )}
      </div>
    </div>
  );
};

// Inline styles for the component
const styles = {
  container: {
    padding: '20px',
    fontFamily: 'Arial, sans-serif'
  },
  statusPanel: {
    backgroundColor: '#f5f5f5',
    border: '1px solid #ddd',
    borderRadius: '8px',
    padding: '20px',
    maxWidth: '400px'
  },
  title: {
    margin: '0 0 20px 0',
    fontSize: '24px',
    color: '#333'
  },
  statusItem: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '12px',
    fontSize: '16px'
  },
  label: {
    fontWeight: 'bold',
    color: '#666'
  },
  value: {
    color: '#333'
  },
  error: {
    marginTop: '15px',
    padding: '10px',
    backgroundColor: '#ffebee',
    border: '1px solid #f44336',
    borderRadius: '4px',
    color: '#c62828',
    fontSize: '14px'
  },
  info: {
    marginTop: '15px',
    padding: '10px',
    backgroundColor: '#e3f2fd',
    border: '1px solid #2196F3',
    borderRadius: '4px',
    color: '#1565c0',
    fontSize: '14px'
  }
};

export default BassDrumPlayer;
