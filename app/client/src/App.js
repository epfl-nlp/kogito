import React, { useState } from 'react'
import { Grid, Form, TextArea, Dropdown, Button, Container, Message, Label, Input, Icon } from 'semantic-ui-react'
import api from './api'
import _ from 'lodash'

import './App.css'

function App() {
  const modelOptions = [
    {
      key: 'comet-gpt2',
      text: 'COMET-GPT2',
      value: 'comet-gpt2'
    },
    {
      key: 'comet-bart',
      text: 'COMET-BART',
      value: 'comet-bart'
    }
  ]

  const [text, setText] = useState('')
  const [model, setModel] = useState('comet-bart')
  const [results, setResults] = useState('')
  const [heads, setHeads] = useState([])

  const generate = () => {
    console.log(text)
    console.log(model)
    api.inference
    .generate({text: text, model: model})
    .then(response => {
      console.log(response.data)
      setResults(JSON.stringify(response.data))
    })
    .catch(error => {

    })
  }

  const handleHeadChange = (index, event) => {
    let newHeads = [...heads]
    newHeads[index] = event.target.value
    setHeads(newHeads)
  }

  const removeHead = (index) => {
    let newHeads = [...heads]
    _.pullAt(newHeads, index)
    setHeads(newHeads)
  }

  const addHead = () => {
    setHeads([...heads, ''])
  }

  const getHeadsJSX = () => {
    return _.isEmpty(heads) ? null :
      <Container>
        {_.map(heads, (head, index) => {
          return (
            <Container key={index} className='cntr-head'>
              <Input
                fluid
                placeholder='Head text'
                onChange={e => handleHeadChange(index, e)}
                value={head}
                label={<Button icon onClick={e => removeHead(index)}><Icon name='minus'></Icon></Button>}
                labelPosition='right'
              />
            </Container>
          )
        })}
      </Container>
  }

  return (
    <Grid container>
      <Grid.Row>
        <Grid.Column>
          <p className='logo'><span className='logo-k'>K</span>ogito</p>
        </Grid.Column>
      </Grid.Row>
      <Grid.Row>
        <Grid.Column>
          <Message color='grey'>
            <Message.Header>Knowledge Inference Tool</Message.Header>
            <p className='description'>Infer knowledge from text</p>
          </Message>
        </Grid.Column>
      </Grid.Row>
      <Grid.Row>
        <Grid columns={2}>
          <Grid.Column>
            <Container className='cntr'>
              <Form>
                <Container className='cntr-label'>
                  <Label color='teal'>Text</Label>
                </Container>
                <TextArea 
                  placeholder='PersonX becomes a great basketball player'
                  onChange={e => setText(e.target.value)}
                  value={text}
                  label='Text'
                />
              </Form>
            </Container>
            <Container className='cntr'>
              <Form>
                <Container className='cntr-label'>
                  <Label color='teal'>Heads</Label>
                </Container>
                <Button icon basic labelPosition='left' onClick={addHead}>
                  <Icon name='plus' />
                  Add Head
                </Button>
                {getHeadsJSX()}
              </Form>
            </Container>
            <Container className='cntr'>
              <Button onClick={generate} fluid className='kbtn'>Generate</Button>
            </Container>
            <Container className='cntr'>
              <Form>
                <Container className='cntr-label'>
                  <Label color='black'>Results</Label>
                </Container>
                <TextArea
                  placeholder='Results'
                  value={results}
                  disabled/>
              </Form>
            </Container>
          </Grid.Column>
          <Grid.Column>
            <Container className='cntr'>
              <Container className='cntr-label'>
                <Label color='teal'>Model</Label>
              </Container>
              <Dropdown
                placeholder='Model'
                fluid
                selection
                options={modelOptions}
                value={model}
                onChange={e => setModel(e.target.value)}
              />
            </Container>
          </Grid.Column>
        </Grid>
      </Grid.Row>
    </Grid>
  )
}
export default App
