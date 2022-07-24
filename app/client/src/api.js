import axios from 'axios'

const API_URI = process.env.REACT_APP_SERVER_URL || "http://localhost:8080"

const api = {
    inference: {
        generate: (data) => axios.post(API_URI + '/inference', data)
    }
}

export default api