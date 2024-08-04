import React from 'react';
import axios from 'axios'

class Device extends React.Component {

  constructor(props) {
    super(props);

    this.state = {
      val777: 123,
      str777: 'Mike'
    }; 
  }

  changevalue2() {
    this.setState({val777: 567});

    var msg = 'Tom';
    this.setState({str777: msg});
  }

  async changevalue()
  {
    const CALL_ENDPOINT = 'http://localhost:3000/hello';

    try {
      const result = await axios.get(CALL_ENDPOINT);
      console.log(result)
     
      const items = result.data;
      var idx = 'message';
      var obj = items[idx];

      //var msg = result.status;

      this.setState({str777: obj});
      this.setState({val777: 111});
    } catch(err){
      console.log('ERROR!! occurred in Backend.')
      this.setState({val777: 5678});

    }
  }

  render() {
    return (
      <div>
          <h1>val777 = {this.state.val777}</h1>
          <h1>Name = {this.state.str777}</h1>
          <button onClick={this.changevalue.bind(this)}>Change</button>
      </div>
    );
  }

}

export default Device;
