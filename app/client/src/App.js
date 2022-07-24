import React, { useState } from 'react'
import { Grid, Form, TextArea, Dropdown, Button, Container, Message, Label, Input, Icon, Radio, Tab, Table, Accordion, Popup, Segment } from 'semantic-ui-react'
import api from './api'
import RELATIONS from './relations'
import _ from 'lodash'
import {CopyToClipboard} from 'react-copy-to-clipboard'
import {saveAs} from 'file-saver'

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
    },
    {
      key: 'gpt2',
      text: 'GPT-2',
      value: 'gpt2'
    }
  ]

  const relationOptions = _.map(RELATIONS, rel => {
    return {key: rel, text: rel, value: rel}
  })

  const headProcOptions = [
    {
      key: 'sentence_extractor',
      text: 'Sentence Extractor',
      value: 'sentence_extractor'
    },
    {
      key: 'noun_phrase_extractor',
      text: 'Noun Phrase Extractor',
      value: 'noun_phrase_extractor'
    },
    {
      key: 'verb_phrase_extractor',
      text: 'Verb Phrase Extractor',
      value: 'verb_phrase_extractor'
    }
  ]

  const relProcOptions = [
    {
      key: 'simple_relation_matcher',
      text: 'Heuristic Matcher',
      value: 'simple_relation_matcher'
    },
    {
      key: 'swem_relation_matcher',
      text: 'GloVe-based Matcher',
      value: 'swem_relation_matcher'
    },
    {
      key: 'distilbert_relation_matcher',
      text: 'DistilBERT-based Matcher',
      value: 'distilbert_relation_matcher'
    },
    {
      key: 'bert_relation_matcher',
      text: 'BERT-based Matcher',
      value: 'bert_relation_matcher'
    },
  ]

  const [text, setText] = useState('')
  const [model, setModel] = useState('comet-bart')
  const [results, setResults] = useState('')
  const [heads, setHeads] = useState([])
  const [relations, setRelations] = useState([])
  const [extractHeads, setExtractHeads] = useState(true)
  const [matchRelations, setMatchRelations] = useState(true)
  const [dryRun, setDryRun] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [showError, setShowError] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)
  const [headProcs, setHeadProcs] = useState(['sentence_extractor', 'noun_phrase_extractor', 'verb_phrase_extractor'])
  const [relProcs, setRelProcs] = useState(['simple_relation_matcher'])
  const [activeHead, setActiveHead] = useState(null)
  const [copiedResults, setCopiedResults] = useState(false)

  let resultMap = {}

  for (let res of results) {
    if (!_.has(resultMap, res['head'])) {
      resultMap[res['head']] = []
    }
    resultMap[res['head']].push({relation: res['relation'], tails: res['tails']})
  }

  const generate = () => {
    setGenerating(true)
    api.inference
    .generate({text: text,
               model: model,
               heads: heads,
               relations: relations,
               extractHeads: extractHeads,
               matchRelations: matchRelations,
               dryRun: dryRun,
               headProcs: headProcs,
               relProcs: relProcs})
    .then(response => {
      setResults(response.data)
      setGenerating(false)
    })
    .catch(error => {
      setGenerating(false)
      setErrorMsg(error.response.data)
      setShowError(true)
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

  const clearResults = () => {
    setResults('')
  }

  const saveResults = () => {
    if (!_.isEmpty(results)) {
      const resultsFile = new Blob([JSON.stringify(results, null, 4)], {type: "text/json;charset=utf-8"})
      saveAs(
        resultsFile,
        "results.json"
      )
    }
  }

  const copyResults = () => {
    if (!_.isEmpty(results)) {
      setCopiedResults(true)
      setTimeout(() => {
        setCopiedResults(false)
      }, 3000)
    }
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

  const resultJSONPane = (
    <div>
      <Segment attached basic className='home-results-json-segment'>
        <Form>
          <TextArea
            placeholder='Results'
            value={_.isEmpty(results) ? '' : JSON.stringify(results, null, 4)}
            rows={30}
            disabled/>
        </Form>
      </Segment>
      <Button.Group basic size='small' attached="bottom">
        <Button icon='trash' onClick={clearResults}/>
        <CopyToClipboard text={_.isEmpty(results) ? '' : JSON.stringify(results, null, 4)}
          onCopy={copyResults}>
          <Button icon='copy'/>
        </CopyToClipboard>
        <Button icon='download' onClick={saveResults}/>
      </Button.Group>
      <Message attached='bottom' success hidden={!copiedResults}>Copied!</Message>
    </div>
  )

  const handleActiveHeadChange = (e, data) => {
    if (activeHead === data.index) {
      return setActiveHead(null)
    }
    return setActiveHead(data.index)
  }

  const resultTablePane = () => {
    return _.map(resultMap, (headResults, head) => {
      return (
        <Grid key={head}>
          <Grid.Row>
            <Grid.Column>
              <Accordion styled fluid>
                <Accordion.Title active={activeHead === head} index={head} onClick={(e, data) => handleActiveHeadChange(e, data)}>
                  <Icon name='dropdown' />
                  {head}
                </Accordion.Title>
                <Accordion.Content active={activeHead === head}>
                  <Table celled structured>
                    <Table.Header>
                      <Table.Row>
                        <Table.HeaderCell>Relation</Table.HeaderCell>
                        <Table.HeaderCell>Tails</Table.HeaderCell>
                      </Table.Row>
                    </Table.Header>
                    <Table.Body>
                      {_.map(headResults, (headResult, hrIndex) => {
                        return (
                          <React.Fragment>
                            <Table.Row key={hrIndex}>
                              <Table.Cell rowSpan={headResult['tails'].length > 0 ? headResult['tails'].length : 1} width={3}>{headResult['relation']}</Table.Cell>
                              {headResult['tails'].length > 0 ? <Table.Cell>{_.head(headResult['tails'])}</Table.Cell> : null}
                            </Table.Row>
                            {_.map(_.slice(headResult['tails'], 1), (tail, tIndex) => {
                                return (
                                  <Table.Row key={tIndex}>
                                    <Table.Cell key={tIndex}>{tail}</Table.Cell>
                                  </Table.Row>
                                )
                            })}
                          </React.Fragment>
                        )
                      })}
                    </Table.Body>
                  </Table>
                </Accordion.Content>
              </Accordion>
            </Grid.Column>
          </Grid.Row>
        </Grid>
      )
    })
  }

  const resultPanes = [
    {menuItem: 'Raw JSON', render: () => resultJSONPane},
    {menuItem: 'Table', render: () => resultTablePane()},
  ]

  return (
    <Grid celled="internally" columns={2}>

      <Grid.Column computer={8} mobile={16}>
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
                <p className='description'>This is an interactive playground for <b>kogito</b>, the Python library that provides an intuitive interface to generate knowledge from text. 
                This app is meant to be used for demo purposes and does not support all available features of the library.
                Please, refer to <a href='https://kogito.readthedocs.io/'>kogito docs</a> for more information on how to use the library. Code for the tool and the library can be found <a href='https://github.com/epfl-nlp/kogito'>here</a>
                </p>
              </Message>
            </Grid.Column>
          </Grid.Row>

          <Grid.Row>
            <Grid.Column>
              <div className='cntr'>
                <Form>
                  <div className='cntr-label'>
                    <Popup content='Main text input to extract heads from if enabled, otherwise used as is for knowledge generation' trigger={<Label color='teal'>Text</Label>}/>
                  </div>
                  <TextArea 
                    placeholder='PersonX becomes a great basketball player'
                    onChange={e => setText(e.target.value)}
                    value={text}
                    label='Text'
                    rows={2}
                  />
                </Form>
              </div>
              <div className='cntr'>
                <Grid columns={4}>
                  <Grid.Column computer={3} mobile={16}>
                    <div className='cntr-label'>
                      <Popup content='If enabled, knowledge heads will be extracted from given text using the head processors defined below' trigger={<Label color='teal'>Extract Heads</Label>}/>
                    </div>
                    <Radio toggle checked={extractHeads} onChange={(e, data) => setExtractHeads(data.checked)}/>
                  </Grid.Column>
                  <Grid.Column computer={3} mobile={16}>
                    <div className='cntr-label'>
                      <Popup content='If enabled, (subset of) relations will be matched with extracted heads using the relation processors defined below, otherwise, heads will be matched to all relations given below' trigger={<Label color='teal'>Match Relations</Label>}/>
                    </div>
                    <Radio toggle checked={matchRelations} onChange={(e, data) => setMatchRelations(data.checked)}/>
                  </Grid.Column>
                  <Grid.Column computer={3} mobile={16}>
                    <div className='cntr-label'>
                      <Popup content="If enabled, actual knowledge generation through a model won't be run and final input graph to the model will be returned as a result" trigger={<Label color='teal'>Dry Run</Label>}/>
                    </div>
                    <Radio toggle checked={dryRun} onChange={(e, data) => setDryRun(data.checked)}/>
                  </Grid.Column>
                  <Grid.Column computer={7} mobile={16}>
                    <div className='cntr-label'>
                      <Popup content='Model to use for knowledge generation' trigger={<Label color='teal'>Model</Label>}/>
                    </div>
                    <Dropdown
                      placeholder='Select Model'
                      selection
                      options={modelOptions}
                      value={model}
                      onChange={(e, data) => setModel(data.value)}
                    />
                  </Grid.Column>
                </Grid>
              </div>
              <div className='cntr'>
                <div className='cntr-label'>
                  <Popup content='Strategies to use for extracting heads from given text if any' trigger={<Label color='teal'>Head Processors</Label>}/>
                </div>
                <Dropdown
                  placeholder='Add Head Processor'
                  selection
                  search
                  multiple
                  options={headProcOptions}
                  value={headProcs || []}
                  onChange={(e, data) => setHeadProcs(data.value)}
                />
              </div>
              <div className='cntr'>
                <div className='cntr-label'>
                  <Popup content='Strategy to use for matching relations with extracted heads if any' trigger={<Label color='teal'>Relation Processors</Label>}/>
                </div>
                <Dropdown
                  placeholder='Select Relation Processor'
                  selection
                  search
                  multiple
                  options={relProcOptions}
                  value={relProcs || []}
                  onChange={(e, data) => setRelProcs(data.value)}
                />
              </div>
              <div className='cntr'>
                <Form>
                  <div className='cntr-label'>
                    <Popup content='Custom head inputs that will be processed as is' trigger={<Label color='teal'>Heads</Label>}/>
                  </div>
                  <Button icon basic labelPosition='left' onClick={addHead}>
                    <Icon name='plus' />
                    Add Head
                  </Button>
                  {getHeadsJSX()}
                </Form>
              </div>
              <div className='cntr'>
                <div className='cntr-label'>
                  <Popup content='Subset of relations to match from. By default, all relations are eligible to be matched' trigger={<Label color='teal'>Relations</Label>}/>
                </div>
                <Dropdown
                  placeholder='All'
                  selection
                  search
                  multiple
                  options={relationOptions}
                  value={relations || []}
                  onChange={(e, data) => setRelations(data.value)}
                />
              </div>
            </Grid.Column>
          </Grid.Row>
        </Grid>
      </Grid.Column>

      <Grid.Column computer={8} mobile={16}>
        <Container className='cntr-label'>
          <Label color='black'>Results</Label>
        </Container>
        <Container className='cntr'>
          <Tab menu={{ secondary: true }} panes={resultPanes}/>
        </Container>
        {showError ? 
              <Container>
                <Message
                  negative
                  header='Error'
                  content={errorMsg}
                  onDismiss={() => setShowError(false)}
                />
              </Container> : null
              }
              <Container className='cntr'>
                <Button
                  onClick={generate}
                  className='kbtn'
                  loading={generating}
                  disabled={generating || (_.isEmpty(text) && (_.isEmpty(heads) || _.every(heads, _.isEmpty)))}
                >
                  Generate
                </Button>
              </Container>
      </Grid.Column>

    </Grid>
  )
}
export default App
