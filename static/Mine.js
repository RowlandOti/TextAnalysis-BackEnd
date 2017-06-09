import 'isomorphic-fetch';
import React from 'react';
import {render} from 'react-dom';
import RaisedButton from 'material-ui/RaisedButton';
import AppBar from 'material-ui/AppBar';
import TextField from 'material-ui/TextField';
import FlatButton from 'material-ui/FlatButton';
import { GridList, GridTile } from 'material-ui/GridList';
import { Card, CardHeader, CardText, CardTitle, CartTitle, CardActions} from 'material-ui/Card';
import axios from 'axios';
import RefreshIndicator from 'material-ui/RefreshIndicator';
import { withStyles, createStyleSheet } from 'material-ui/styles';
import Divider from 'material-ui/Divider';

import Bar from './Search.js'; 


const styles = {

    container: {
        position: 'fixed',
    },

    cardA: {
        display: 'inline-block',
        position: 'fixed',
        marginLeft: '4.8%',
        width: 500,
        height: 200,
        marginTop: 100,
    },
    cardQ: {
        display: 'inline-block',
        position: 'relative',
        marginLeft: 90,
        width: 300,
        height: 200,
        marginTop: 100,
    },
    margins: {
        width: 500,
        marginLeft: '35%',
        marginTop: 70,
        textAlign: 'center',
    },
    button: {
        marginLeft: '50%'
    },
    gridlist: {
        width: 500,
        height: 450,
        overflowY: 'auto',
    },
    refresh: {
        display: 'inline-block',
        position: 'relative',
    }
};


class Mines extends React.Component{
    constructor(props){
        super(props);
        this.handleChange = this.handleChange.bind(this);
        this.handleOn = this.handleOn.bind(this);
        this.handleOff = this.handleOff.bind(this);

        this.state = {
            value: null,
            val: ""
        }

        this.fetch={
            fetching: false,
        }
    }

    

    handleChange(e) {
        this.setState({fetching: true})

        var par = String(e.target.value);

        //Mains
        axios.get('http://127.0.0.1:8000/api/v2.0/'+par)
        .then(response => {
                if(response){
                    this.setState({fetching: false})
                    this.setState({val: response.data.analyze.sentiment})
                }else{
                    this.setState({val: "Bad"})
                }
                }    
            
        )
        .catch((error) => {
            if(error){
                this.setState({val: ""})
            }
            else{
               this.setState({fetching: false}) 
            }
        });

        this.setState({value: par})

        if(par.length == 0){
            this.setState({fetching: false}),
            this.setState({val: "Good"})
        }

        
    }

    handleOff(e) {
        this.setState({fetching: false})
    }

    handleOn(e) {
        this.setState({fetching: true})
    }

    render(){
        
        const fetching = this.state.fetching;
        let load = null;
        if (fetching){
            
            load = <RefreshIndicator
                        size={35}
                        top={50}
                        left={10}
                        loadingColor="#FF9800"
                        status="loading"
                        style={styles.refresh} />
                    //<LinearProgress mode='indeterminate'/>
        }

        return(
            <div>
                <Bar />
                <TextField style={styles.margins}  hintText = "Type text you need analyzed." 
                 onChange={this.handleChange} />
                 
                 <br />
                
                
                <div className="container" style={styles.container}>

                    <Card style={styles.cardQ}>
                        <CardHeader 
                            title="Your Text"/>
                        <Divider />


                        <CardText>{this.state.value}</CardText>

                        { load }
                        
                    </Card>
                    
                    <Card style={styles.cardA}>
                        <CardHeader 
                            title="Analyzed"/>
                        <Divider />
                        
                        <CardText>{this.state.val}</CardText>

                        
                    </Card>

                </div>
                <br />
            </div>
            
        );
    }

}

export default Mines;
