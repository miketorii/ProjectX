import React from 'react';
import ReactDOM from 'react-dom';

class Clock extends React.Component {
  constructor(props) {
      super(props);
      this.state = {
	  date: new Date(),
	  month: 'Oct',
	  number: 10
	  
      };
      this.name = 'mike';
  }

  componentDidMount(){
      this.timerID = setInterval(
	  () => this.tick(),
	  1000
      );
  }

  componentWillUnmount(){
      clearInterval(this.timerID);
  }

  tick(){
      this.setState({date: new Date()});
  }
    
  render() {
    return (
      <div>
          <h1>Hello Mike</h1>
	  <h2>{this.name}{this.state.month}{this.state.number}</h2>
	  <h2>{this.state.date.toLocaleTimeString()}</h2>
      </div>
    );
  }
}

ReactDOM.render(
  <Clock />,
  document.getElementById('root')
);


