import React from 'react';
import logo from './logo.svg';
import './App.css';

import Device from './components/device';

function App() {

  //React.Component.setState({val0: 200});
  var msg = "init app";

  return (
    <div className="App">
      <Device />
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
        <p>Mike {msg}</p>
      </header>
    </div>
  );
}

export default App;
