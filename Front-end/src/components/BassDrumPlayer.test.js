import { render, screen, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import BassDrumPlayer from './BassDrumPlayer';
import io from 'socket.io-client';

// Mock socket.io-client
jest.mock('socket.io-client');

// Mock Web Audio API
let mockAudioContext;

const mockAudioBuffer = {
  duration: 1.0,
  length: 44100,
  numberOfChannels: 2,
  sampleRate: 44100
};

const mockSource = {
  buffer: null,
  connect: jest.fn(),
  start: jest.fn(),
  stop: jest.fn(),
  onended: null
};

// Mock fetch for audio file loading
global.fetch = jest.fn();

// Mock AudioContext - will be reset in beforeEach
global.AudioContext = jest.fn();
global.webkitAudioContext = jest.fn();

describe('BassDrumPlayer Integration Tests', () => {
  let mockSocket;
  let socketEventHandlers;

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    
    // Create fresh mock audio context for each test
    mockAudioContext = {
      state: 'running',
      destination: {},
      decodeAudioData: jest.fn(),
      createBufferSource: jest.fn(),
      resume: jest.fn(),
      close: jest.fn()
    };
    
    // Setup AudioContext constructor
    global.AudioContext = jest.fn(() => mockAudioContext);
    global.webkitAudioContext = jest.fn(() => mockAudioContext);
    
    // Setup socket event handlers storage
    socketEventHandlers = {};
    
    // Create mock socket instance
    mockSocket = {
      on: jest.fn((event, handler) => {
        socketEventHandlers[event] = handler;
      }),
      disconnect: jest.fn(),
      connected: false
    };
    
    // Mock io() to return our mock socket
    io.mockReturnValue(mockSocket);
    
    // Mock fetch to return audio file
    global.fetch.mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      arrayBuffer: jest.fn().mockResolvedValue(new ArrayBuffer(8))
    });
    
    // Mock decodeAudioData to return audio buffer
    mockAudioContext.decodeAudioData.mockResolvedValue(mockAudioBuffer);
    
    // Mock createBufferSource to return source
    mockAudioContext.createBufferSource.mockReturnValue({
      ...mockSource,
      connect: jest.fn().mockReturnThis(),
      start: jest.fn(),
      stop: jest.fn()
    });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  /**
   * Test: WebSocket connection establishment
   * Requirements: 4.2, 4.3
   */
  describe('WebSocket Connection', () => {
    test('should establish WebSocket connection on mount', () => {
      render(<BassDrumPlayer serverUrl="http://localhost:5000" />);
      
      // Verify io was called with correct parameters
      expect(io).toHaveBeenCalledWith('http://localhost:5000', {
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionAttempts: 5
      });
      
      // Verify event listeners were registered
      expect(mockSocket.on).toHaveBeenCalledWith('connect', expect.any(Function));
      expect(mockSocket.on).toHaveBeenCalledWith('disconnect', expect.any(Function));
      expect(mockSocket.on).toHaveBeenCalledWith('connect_error', expect.any(Function));
      expect(mockSocket.on).toHaveBeenCalledWith('bass_drum_trigger', expect.any(Function));
    });

    test('should display connected status when connection established', async () => {
      render(<BassDrumPlayer />);
      
      // Initially should show disconnected
      expect(screen.getByText(/Disconnected/i)).toBeInTheDocument();
      
      // Simulate connection
      act(() => {
        socketEventHandlers['connect']();
      });
      
      // Should now show connected
      await waitFor(() => {
        expect(screen.getByText(/Connected/i)).toBeInTheDocument();
      });
    });

    test('should display disconnected status when connection lost', async () => {
      render(<BassDrumPlayer />);
      
      // Establish connection first
      act(() => {
        socketEventHandlers['connect']();
      });
      
      await waitFor(() => {
        expect(screen.getByText(/Connected/i)).toBeInTheDocument();
      });
      
      // Simulate disconnection
      act(() => {
        socketEventHandlers['disconnect']();
      });
      
      // Should show disconnected
      await waitFor(() => {
        expect(screen.getByText(/Disconnected/i)).toBeInTheDocument();
      });
    });

    test('should display error message on connection error', async () => {
      render(<BassDrumPlayer />);
      
      // Simulate connection error
      act(() => {
        socketEventHandlers['connect_error'](new Error('Connection refused'));
      });
      
      // Should display error
      await waitFor(() => {
        expect(screen.getByText(/Connection error/i)).toBeInTheDocument();
        expect(screen.getByText(/Connection refused/i)).toBeInTheDocument();
      });
    });

    test('should disconnect socket on unmount', () => {
      const { unmount } = render(<BassDrumPlayer />);
      
      // Unmount component
      unmount();
      
      // Verify disconnect was called
      expect(mockSocket.disconnect).toHaveBeenCalled();
    });

    test('should use custom server URL when provided', () => {
      render(<BassDrumPlayer serverUrl="http://custom-server:8080" />);
      
      expect(io).toHaveBeenCalledWith(
        'http://custom-server:8080',
        expect.any(Object)
      );
    });
  });

  /**
   * Test: Event reception
   * Requirements: 4.2, 4.3
   */
  describe('Event Reception', () => {
    test('should receive and process bass_drum_trigger events', async () => {
      render(<BassDrumPlayer />);
      
      // Wait for audio to load
      await waitFor(() => {
        expect(mockAudioContext.decodeAudioData).toHaveBeenCalled();
      });
      
      // Simulate receiving trigger event
      const tapData = {
        tap_type: 'HEEL',
        timestamp: 1234567890.123,
        velocity: 0.35
      };
      
      act(() => {
        socketEventHandlers['bass_drum_trigger'](tapData);
      });
      
      // Verify trigger count increased
      await waitFor(() => {
        expect(screen.getByText('1')).toBeInTheDocument();
      });
      
      // Verify last tap type is displayed
      expect(screen.getByText('HEEL')).toBeInTheDocument();
    });

    test('should handle multiple trigger events', async () => {
      render(<BassDrumPlayer />);
      
      await waitFor(() => {
        expect(mockAudioContext.decodeAudioData).toHaveBeenCalled();
      });
      
      // Send multiple trigger events
      act(() => {
        socketEventHandlers['bass_drum_trigger']({ tap_type: 'HEEL', timestamp: 1.0 });
      });
      
      await waitFor(() => {
        expect(screen.getByText('1')).toBeInTheDocument();
      });
      
      act(() => {
        socketEventHandlers['bass_drum_trigger']({ tap_type: 'TOE', timestamp: 2.0 });
      });
      
      await waitFor(() => {
        expect(screen.getByText('2')).toBeInTheDocument();
      });
      
      act(() => {
        socketEventHandlers['bass_drum_trigger']({ tap_type: 'FULL_FOOT', timestamp: 3.0 });
      });
      
      await waitFor(() => {
        expect(screen.getByText('3')).toBeInTheDocument();
      });
      
      // Last tap type should be FULL_FOOT
      expect(screen.getByText('FULL_FOOT')).toBeInTheDocument();
    });

    test('should update last tap type for each event', async () => {
      render(<BassDrumPlayer />);
      
      await waitFor(() => {
        expect(mockAudioContext.decodeAudioData).toHaveBeenCalled();
      });
      
      // Send HEEL tap
      act(() => {
        socketEventHandlers['bass_drum_trigger']({ tap_type: 'HEEL', timestamp: 1.0 });
      });
      
      await waitFor(() => {
        expect(screen.getByText('HEEL')).toBeInTheDocument();
      });
      
      // Send TOE tap
      act(() => {
        socketEventHandlers['bass_drum_trigger']({ tap_type: 'TOE', timestamp: 2.0 });
      });
      
      await waitFor(() => {
        expect(screen.getByText('TOE')).toBeInTheDocument();
      });
      
      // HEEL should no longer be the last tap type
      const toeElements = screen.getAllByText('TOE');
      expect(toeElements.length).toBeGreaterThan(0);
    });
  });

  /**
   * Test: Audio playback
   * Requirements: 4.2, 4.3
   */
  describe('Audio Playback', () => {
    test('should load audio file on mount', async () => {
      render(<BassDrumPlayer />);
      
      // Verify fetch was called for audio file
      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/sounds/bass-drum.wav');
      });
      
      // Verify audio was decoded
      expect(mockAudioContext.decodeAudioData).toHaveBeenCalled();
    });

    test('should create audio context on mount', () => {
      render(<BassDrumPlayer />);
      
      expect(global.AudioContext).toHaveBeenCalled();
    });

    test('should play audio when trigger event received', async () => {
      const mockSourceInstance = {
        buffer: null,
        connect: jest.fn().mockReturnThis(),
        start: jest.fn(),
        stop: jest.fn(),
        onended: null
      };
      
      mockAudioContext.createBufferSource.mockReturnValue(mockSourceInstance);
      
      render(<BassDrumPlayer />);
      
      // Wait for audio to load
      await waitFor(() => {
        expect(mockAudioContext.decodeAudioData).toHaveBeenCalled();
      });
      
      // Trigger audio playback
      act(() => {
        socketEventHandlers['bass_drum_trigger']({ tap_type: 'HEEL', timestamp: 1.0 });
      });
      
      // Verify audio source was created and started
      await waitFor(() => {
        expect(mockAudioContext.createBufferSource).toHaveBeenCalled();
        expect(mockSourceInstance.connect).toHaveBeenCalledWith(mockAudioContext.destination);
        expect(mockSourceInstance.start).toHaveBeenCalledWith(0);
      });
    });

    test('should stop previous audio when new trigger received', async () => {
      const mockSourceInstance1 = {
        buffer: null,
        connect: jest.fn().mockReturnThis(),
        start: jest.fn(),
        stop: jest.fn(),
        onended: null
      };
      
      const mockSourceInstance2 = {
        buffer: null,
        connect: jest.fn().mockReturnThis(),
        start: jest.fn(),
        stop: jest.fn(),
        onended: null
      };
      
      mockAudioContext.createBufferSource
        .mockReturnValueOnce(mockSourceInstance1)
        .mockReturnValueOnce(mockSourceInstance2);
      
      render(<BassDrumPlayer />);
      
      await waitFor(() => {
        expect(mockAudioContext.decodeAudioData).toHaveBeenCalled();
      });
      
      // First trigger
      act(() => {
        socketEventHandlers['bass_drum_trigger']({ tap_type: 'HEEL', timestamp: 1.0 });
      });
      
      await waitFor(() => {
        expect(mockSourceInstance1.start).toHaveBeenCalled();
      });
      
      // Second trigger (should stop first)
      act(() => {
        socketEventHandlers['bass_drum_trigger']({ tap_type: 'TOE', timestamp: 2.0 });
      });
      
      await waitFor(() => {
        expect(mockSourceInstance1.stop).toHaveBeenCalled();
        expect(mockSourceInstance2.start).toHaveBeenCalled();
      });
    });

    test('should handle audio loading errors gracefully', async () => {
      // Mock fetch to fail
      global.fetch.mockRejectedValue(new Error('Network error'));
      
      render(<BassDrumPlayer />);
      
      // Should display error message
      await waitFor(() => {
        expect(screen.getByText(/Failed to load audio/i)).toBeInTheDocument();
        expect(screen.getByText(/Network error/i)).toBeInTheDocument();
      });
    });

    test('should handle audio decoding errors gracefully', async () => {
      // Mock decodeAudioData to fail
      mockAudioContext.decodeAudioData.mockRejectedValue(new Error('Decode error'));
      
      render(<BassDrumPlayer />);
      
      // Should display error message
      await waitFor(() => {
        expect(screen.getByText(/Failed to load audio/i)).toBeInTheDocument();
        expect(screen.getByText(/Decode error/i)).toBeInTheDocument();
      });
    });

    test('should resume audio context when suspended', async () => {
      // Set audio context to suspended state
      mockAudioContext.state = 'suspended';
      
      const mockSourceInstance = {
        buffer: null,
        connect: jest.fn().mockReturnThis(),
        start: jest.fn(),
        stop: jest.fn(),
        onended: null
      };
      
      mockAudioContext.createBufferSource.mockReturnValue(mockSourceInstance);
      
      render(<BassDrumPlayer />);
      
      await waitFor(() => {
        expect(mockAudioContext.decodeAudioData).toHaveBeenCalled();
      });
      
      // Trigger audio playback
      act(() => {
        socketEventHandlers['bass_drum_trigger']({ tap_type: 'HEEL', timestamp: 1.0 });
      });
      
      // Verify audio context was resumed
      await waitFor(() => {
        expect(mockAudioContext.resume).toHaveBeenCalled();
      });
    });

    test('should close audio context on unmount', () => {
      const { unmount } = render(<BassDrumPlayer />);
      
      unmount();
      
      expect(mockAudioContext.close).toHaveBeenCalled();
    });
  });

  /**
   * Test: Integration scenarios
   * Requirements: 4.2, 4.3
   */
  describe('Integration Scenarios', () => {
    test('should handle complete workflow: connect, load audio, receive event, play sound', async () => {
      const mockSourceInstance = {
        buffer: null,
        connect: jest.fn().mockReturnThis(),
        start: jest.fn(),
        stop: jest.fn(),
        onended: null
      };
      
      mockAudioContext.createBufferSource.mockReturnValue(mockSourceInstance);
      
      render(<BassDrumPlayer />);
      
      // Step 1: Verify initial state
      expect(screen.getByText(/Disconnected/i)).toBeInTheDocument();
      expect(screen.getByText('0')).toBeInTheDocument();
      
      // Step 2: Connect to WebSocket
      act(() => {
        socketEventHandlers['connect']();
      });
      
      await waitFor(() => {
        expect(screen.getByText(/Connected/i)).toBeInTheDocument();
      });
      
      // Step 3: Wait for audio to load
      await waitFor(() => {
        expect(mockAudioContext.decodeAudioData).toHaveBeenCalled();
      });
      
      // Step 4: Receive trigger event
      act(() => {
        socketEventHandlers['bass_drum_trigger']({
          tap_type: 'HEEL',
          timestamp: 1234567890.123,
          velocity: 0.35
        });
      });
      
      // Step 5: Verify audio played and UI updated
      await waitFor(() => {
        expect(mockSourceInstance.start).toHaveBeenCalledWith(0);
        expect(screen.getByText('1')).toBeInTheDocument();
        expect(screen.getByText('HEEL')).toBeInTheDocument();
      });
    });

    test('should handle reconnection scenario', async () => {
      render(<BassDrumPlayer />);
      
      // Connect
      act(() => {
        socketEventHandlers['connect']();
      });
      
      await waitFor(() => {
        expect(screen.getByText(/Connected/i)).toBeInTheDocument();
      });
      
      // Disconnect
      act(() => {
        socketEventHandlers['disconnect']();
      });
      
      await waitFor(() => {
        expect(screen.getByText(/Disconnected/i)).toBeInTheDocument();
      });
      
      // Reconnect
      act(() => {
        socketEventHandlers['connect']();
      });
      
      await waitFor(() => {
        expect(screen.getByText(/Connected/i)).toBeInTheDocument();
      });
    });

    test('should maintain trigger count across connection changes', async () => {
      const mockSourceInstance = {
        buffer: null,
        connect: jest.fn().mockReturnThis(),
        start: jest.fn(),
        stop: jest.fn(),
        onended: null
      };
      
      mockAudioContext.createBufferSource.mockReturnValue(mockSourceInstance);
      
      render(<BassDrumPlayer />);
      
      await waitFor(() => {
        expect(mockAudioContext.decodeAudioData).toHaveBeenCalled();
      });
      
      // Connect and trigger
      act(() => {
        socketEventHandlers['connect']();
      });
      
      act(() => {
        socketEventHandlers['bass_drum_trigger']({ tap_type: 'HEEL', timestamp: 1.0 });
      });
      
      await waitFor(() => {
        expect(screen.getByText('1')).toBeInTheDocument();
      });
      
      // Disconnect
      act(() => {
        socketEventHandlers['disconnect']();
      });
      
      // Reconnect
      act(() => {
        socketEventHandlers['connect']();
      });
      
      // Trigger again
      act(() => {
        socketEventHandlers['bass_drum_trigger']({ tap_type: 'TOE', timestamp: 2.0 });
      });
      
      // Count should be 2
      await waitFor(() => {
        expect(screen.getByText('2')).toBeInTheDocument();
      });
    });
  });
});
