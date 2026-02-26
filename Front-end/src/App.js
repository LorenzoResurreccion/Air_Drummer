import React from 'react';
import BassDrumPlayer from './components/BassDrumPlayer';

/**
 * Main App Component for AIR DRUMMER
 * 
 * Integrates the BassDrumPlayer component to provide
 * audio feedback for right foot bass drum taps.
 */
function App() {
  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <h1 style={styles.appTitle}>AIR DRUMMER</h1>
        <p style={styles.subtitle}>Virtual drumming through body movements</p>
      </header>
      
      <main style={styles.main}>
        <BassDrumPlayer serverUrl="http://localhost:5001" />
        
        <div style={styles.instructions}>
          <h3 style={styles.instructionsTitle}>How to Use:</h3>
          <ol style={styles.instructionsList}>
            <li>Ensure the backend is running: <code>python Back-end/Body-Tracker.py</code></li>
            <li>Position yourself in front of your webcam</li>
            <li>Tap your right foot (heel, toe, or full foot) to trigger bass drum sounds</li>
            <li>Watch the trigger count increase as you play!</li>
          </ol>
        </div>
      </main>
    </div>
  );
}

const styles = {
  app: {
    minHeight: '100vh',
    backgroundColor: '#282c34',
    color: 'white'
  },
  header: {
    padding: '30px 20px',
    backgroundColor: '#1a1d24',
    textAlign: 'center',
    borderBottom: '3px solid #4CAF50'
  },
  appTitle: {
    margin: '0 0 10px 0',
    fontSize: '48px',
    fontWeight: 'bold',
    color: '#4CAF50'
  },
  subtitle: {
    margin: 0,
    fontSize: '18px',
    color: '#aaa'
  },
  main: {
    padding: '40px 20px',
    maxWidth: '1200px',
    margin: '0 auto'
  },
  instructions: {
    marginTop: '40px',
    padding: '20px',
    backgroundColor: '#1a1d24',
    borderRadius: '8px',
    maxWidth: '600px'
  },
  instructionsTitle: {
    marginTop: 0,
    color: '#4CAF50'
  },
  instructionsList: {
    lineHeight: '1.8',
    fontSize: '16px'
  }
};

export default App;
