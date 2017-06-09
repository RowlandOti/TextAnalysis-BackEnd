import React from 'react';
import ReactDOM from 'react-dom';
import TextField from 'material-ui/TextField';
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider';


import Mines from './Mine.js';

const App = () => (
    <MuiThemeProvider>
        
        <Mines/>
    </MuiThemeProvider>
    
);

ReactDOM.render(
    <App />, document.getElementById('app')
);
