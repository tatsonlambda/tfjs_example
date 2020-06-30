import React, { useReducer, useState, Component } from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';

class App extends Component {
	constructor(props) {
		super(props);
		this.state = {
			file : null,
			blue : 0,
			yellow: 0,
			prediction: ""
		}
		this.model = null;
	}

	async componentDidMount(){
		this.model = await tf.loadLayersModel('../assets/model/model.json');
	}
	
	fileSelectedHandler = event => {
		this.setState({
			file: event.target.files[0]
		})
	}

	fileUploadHandler() {
		let img = new Image()
        img.src = window.URL.createObjectURL(this.state.file)
        img.onload = async () => {
			const a = tf.browser.fromPixels(img)
			let offset = tf.scalar(127.5);
			let a2 = a.sub(offset)
				.div(offset)
				.expandDims();
			const aaa = tf.image.resizeBilinear(a2, [224,224])
			console.log(aaa.arraySync())
			//const aa = tf.stack([aaa])
			const aa = aaa
			const prediction = (await this.model.predict(aa));
			console.log(prediction.dataSync())
			const classNames = ["blue", "yellow"];
			this.setState({
				blue: prediction.dataSync()[0],
				yellow: prediction.dataSync()[1],
				prediction: classNames[tf.argMax(prediction, 1).dataSync()]
			})
			return true;
		}
	}

	render(){
        return(
            <div className="App">
                <input type="file" onChange={this.fileSelectedHandler}/>
				<button onClick={this.fileUploadHandler.bind(this)}>Upload</button>
				<p>You are a {this.state.prediction}</p><br/>
				<p>{this.state.blue}% blue and {this.state.yellow}% yellow</p>
            </div> 
        )
    }
}


export default App;
