const get = (id) => document.getElementById(id)
const canvas = get('image')
const ctx = canvas.getContext('2d')
const slider = get('frame')
const colors = { left: '#81e3d9', right: '#ffc582' }
const captions = { slam_left: 'SLAM left', slam_right: 'SLAM right', upper_left: 'Upper left', upper_right: 'Upper right' }
let workspace
let frame = 0
let camera = 'slam_left'
let hand = 'left'
let image
let drag
let pending = false
let loadVersion = 0
let lastEdit
let source = 'sam'
let job = { state: 'idle', serial: 0, log: [] }
let jobRequest = false
let loadedJob = 0
let pollTimer
let pollRevision = 0
let coverageKey

function status(message, error = false) {
  get('status').textContent = message
  get('status').dataset.error = error
}

function key(sequence, view, side) {
  return [sequence, view, side].join('/')
}

function group() {
  return workspace.groups[frame]
}

function entries(side) {
  const id = key(group().sequence, camera, side)
  return [workspace.labels[id], workspace.proposals[id], workspace.terra?.[id]]
}

function proposalFor(side) {
  const [reviewed, sam, terra] = entries(side)
  return reviewed ? null : source === 'sam' ? sam : terra
}

function actionable(proposal) {
  return proposal?.box_xyxy || proposal?.status === 'absent'
}

function drawBox(box, side, proposal = false) {
  const [x, y, x2, y2] = box
  const pixel = canvas.width / Math.max(1, canvas.clientWidth)
  ctx.strokeStyle = colors[side]
  ctx.lineWidth = Math.max(.65, pixel * 2)
  ctx.setLineDash(proposal ? [pixel * 7, pixel * 5] : [])
  ctx.strokeRect(x, y, x2 - x, y2 - y)
  ctx.setLineDash([])
}

function paint() {
  ctx.fillStyle = '#080a0e'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  if (!image) return
  ctx.drawImage(image, 0, 0)
  for (const side of workspace.hands) {
    const [reviewed] = entries(side)
    const proposal = proposalFor(side)
    if (reviewed?.box_xyxy) drawBox(reviewed.box_xyxy, side)
    else if (proposal?.box_xyxy) drawBox(proposal.box_xyxy, side, true)
  }
  if (drag) drawBox(bounds(drag.start, drag.end), hand)
}

function summarize() {
  const labels = get('labels')
  labels.replaceChildren()
  for (const side of workspace.hands) {
    const [reviewed, sam, terra] = entries(side)
    const title = document.createElement('dt')
    title.textContent = side === 'left' ? 'Left' : 'Right'
    title.className = side
    const detail = document.createElement('dd')
    if (reviewed) {
      if (reviewed.status !== 'box') detail.textContent = reviewed.status === 'absent' ? 'Confirmed absent' : 'Unknown'
      else if (reviewed.source === 'manual') detail.textContent = 'Manual seed'
      else if (reviewed.box_mode === 'amodal') detail.textContent = 'HOT3D amodal box'
      else detail.textContent = 'Reviewed box'
    } else {
      const available = []
      if (sam) available.push('SAM box')
      if (terra) available.push('Terra ' + terra.status)
      detail.textContent = available.join(' · ') || 'Unlabeled'
    }
    labels.append(title, detail)
  }
  const [reviewed] = entries(hand)
  const total = workspace.groups.length * workspace.order.length * workspace.hands.length
  get('progress').textContent = `${labeledCount} / ${total} slots labeled`
  get('unreviewed').disabled = pending || !findTarget(false)
  get('next-proposal').disabled = pending || !findTarget(true)
  get('undo').disabled = pending || !lastEdit || !image
  get('accept').disabled = !image || pending || !actionable(proposalFor(hand))
  get('accept').textContent = proposalFor(hand)?.status === 'absent' ? 'Confirm absent + next' : 'Accept + next'
  get('absent').disabled = !image || pending || reviewed?.status === 'absent'
  get('unknown').disabled = !image || pending || reviewed?.status === 'unknown'
  for (const button of get('hands').children) button.setAttribute('aria-pressed', String(button.value === hand))
  for (const button of get('cameras').children) button.setAttribute('aria-pressed', String(button.value === camera))
  updateModel()
  drawCoverage()
}

function show() {
  const version = ++loadVersion
  get('retry').hidden = true
  get('image-status').textContent = 'Loading image…'
  status('')
  const groupNow = group()
  slider.value = frame
  get('position').textContent = `${frame + 1} / ${workspace.groups.length} · ${(groupNow.time_us / 1e6).toFixed(3)} s`
  const { width, height } = workspace.cameras[camera]
  canvas.width = width
  canvas.height = height
  fitImage()
  drag = null
  image = null
  paint()
  summarize()
  const loading = new Image()
  const view = camera
  loading.onload = () => {
    if (version !== loadVersion) return
    get('image-status').textContent = ''
    image = loading
    summarize()
    paint()
  }
  loading.onerror = () => {
    if (version !== loadVersion) return
    get('image-status').textContent = 'Could not load the tracking image.'
    get('retry').hidden = false
  }
  loading.src = `/frames/${view}/${String(groupNow.sequence).padStart(12, '0')}.png`
}

function fitImage() {
  const maxWidth = canvas.closest('.stage').clientWidth - 8
  const heightLimit = innerHeight * (innerWidth <= 590 ? .38 : .62)
  canvas.style.width = Math.min(maxWidth, heightLimit * canvas.width / canvas.height) + 'px'
}

async function save(action, box) {
  const advancing = action === 'accept'
  if (!workspace || !image || pending) return
  if (action === 'accept') {
    const proposal = proposalFor(hand)
    if (!actionable(proposal)) return
    box = proposal.box_xyxy
    if (source === 'terra') action = 'accept_terra'
  }
  const undoing = action === 'undo'
  if (undoing && !lastEdit) return
  const label = undoing ? lastEdit : { sequence: group().sequence, camera, hand, action, box }
  const id = key(label.sequence, label.camera, label.hand)
  pending = true
  summarize()
  try {
    const response = await fetch(undoing ? '/api/undo' : '/api/label', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(label),
    })
    const result = await response.json()
    if (!response.ok) throw new Error(result.error)
    if (result.label) workspace.labels[id] = result.label
    else delete workspace.labels[id]
    coverageKey = undefined
    labelRevision++
    updateLabelCounts()
    lastEdit = undoing ? null : { sequence: label.sequence, camera: label.camera, hand: label.hand, token: result.token }
    if (advancing) goToTarget(true)
    status(undoing ? `Undid edit to ${captions[label.camera]}, ${label.hand} hand, group ${label.sequence}.` : 'Saved.')
  } catch (error) {
    status(error.message, true)
  } finally {
    pending = false
    summarize()
    paint()
  }
}

function move(next) {
  if (!workspace) return
  const target = Math.min(Math.max(0, next), workspace.groups.length - 1)
  if (target !== frame) {
    frame = target
    show()
  }
}

function selectHand(side) {
  hand = side
  drag = null
  summarize()
  paint()
}

function chooseCamera(view) {
  if (view && view !== camera) {
    camera = view
    show()
  }
}

function button(value, caption, handler) {
  const result = document.createElement('button')
  result.type = 'button'
  result.value = value
  result.textContent = caption
  result.onclick = () => handler(value)
  return result
}

async function reload() {
  try {
    const response = await fetch('/api/workspace')
    if (!response.ok) throw new Error('Could not read the labeling workspace.')
    const opening = !workspace
    workspace = await response.json()
    updateLabelCounts()
    coverageKey = undefined
    if (opening) setRange(1, workspace.groups.length, false)
    get('first').max = get('last').max = workspace.groups.length
    get('take').textContent = workspace.take
    slider.max = workspace.groups.length - 1
    get('hands').replaceChildren(...workspace.hands.map((side) => button(side, side, selectHand)))
    camera = workspace.order.includes(camera) ? camera : workspace.order[0]
    get('cameras').replaceChildren(...workspace.order.map((view) => button(view, captions[view], chooseCamera)))
    show()
    status('')
  } catch (error) {
    status(error.message, true)
  }
}

function point(event) {
  const rect = canvas.getBoundingClientRect()
  return [
    Math.round(Math.max(0, Math.min(canvas.width, (event.clientX - rect.left) * canvas.width / rect.width))),
    Math.round(Math.max(0, Math.min(canvas.height, (event.clientY - rect.top) * canvas.height / rect.height))),
  ]
}

function bounds(a, b) {
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])]
}

canvas.onpointerdown = (event) => {
  if (!image || pending || event.button !== 0) return
  canvas.setPointerCapture(event.pointerId)
  drag = { start: point(event), end: point(event) }
}
canvas.onpointermove = (event) => {
  if (!drag) return
  drag.end = point(event)
  paint()
}
canvas.onpointercancel = () => {
  drag = null
  paint()
}
canvas.onpointerup = (event) => {
  if (!drag) return
  const box = bounds(drag.start, point(event))
  drag = null
  paint()
  if (box[2] > box[0] && box[3] > box[1]) save('box', box)
}
get('retry').onclick = show
get('undo').onclick = () => save('undo')
get('unreviewed').onclick = () => goToTarget(false)
get('next-proposal').onclick = () => goToTarget(true)
slider.oninput = () => move(Number(slider.value))
get('previous').onclick = () => move(frame - 1)
get('next').onclick = () => move(frame + 1)
get('reload').onclick = () => { if (!pending) reload() }
for (const action of ['accept', 'absent', 'unknown']) get(action).onclick = () => save(action)
window.addEventListener('keydown', (event) => {
  if (!workspace || event.altKey || event.ctrlKey || event.metaKey || event.target.closest('input,select,textarea')) return
  const keyName = event.key.toLowerCase()
  if (keyName === 'arrowleft') move(frame - 1)
  else if (keyName === 'arrowright') move(frame + 1)
  else if (/^[1-4]$/.test(keyName)) chooseCamera(workspace.order[Number(keyName) - 1])
  else if (keyName === 'l' || keyName === 'r') selectHand(keyName === 'l' ? 'left' : 'right')
  else if (keyName === 'enter' && event.target.tagName !== 'BUTTON' && !get('accept').disabled) save('accept')
  else return
  event.preventDefault()
})

function range() {
  const first = Number(get('first').value)
  const last = Number(get('last').value)
  if (Number.isInteger(first) && Number.isInteger(last) && first >= 1 && first <= last && last <= workspace.groups.length) {
    return { first, last }
  }
  return null
}

function setRange(first, last, update = true) {
  get('first').value = first
  get('last').value = last
  if (update) summarize()
}

let targets
function updateTargets() {
  const part = range()
  const keyNow = `${source}/${part?.first}/${part?.last}`
  if (targets?.key === keyNow && targets.labels === workspace.labels && targets.revision === labelRevision) return targets
  const unlabeled = [], proposals = []
  if (part) {
    for (let f = part.first - 1; f < part.last; f++) {
      const sequence = workspace.groups[f].sequence
      for (const view of workspace.order) {
        for (const side of workspace.hands) {
          const id = key(sequence, view, side)
          if (workspace.labels[id]) continue
          const target = { frame: f, camera: view, hand: side, index: slot(f, view, side) }
          unlabeled.push(target)
          const proposal = (source === 'sam' ? workspace.proposals : workspace.terra)[id]
          if (actionable(proposal)) proposals.push(target)
        }
      }
    }
  }
  targets = { key: keyNow, labels: workspace.labels, revision: labelRevision, unlabeled, proposals }
  return targets
}

function slot(position, view, side) {
  return (position * workspace.order.length + workspace.order.indexOf(view)) * workspace.hands.length
    + workspace.hands.indexOf(side)
}

function findTarget(onlyProposals) {
  const { proposals, unlabeled } = updateTargets()
  const list = onlyProposals ? proposals : unlabeled
  const current = slot(frame, camera, hand)
  let first = 0, last = list.length
  while (first < last) {
    const middle = Math.floor((first+last)/2)
    if (list[middle].index > current) last = middle
    else first = middle + 1
  }
  return list[first] || (list[0]?.index !== current ? list[0] : undefined)
}

function goToTarget(onlyProposals) {
  const target = findTarget(onlyProposals)
  if (!target) return
  frame = target.frame
  camera = target.camera
  hand = target.hand
  show()
}

let labelRevision = 0
const coverage = get('coverage')
const coverageCtx = coverage.getContext('2d')
function drawCoverage() {
  const part = range()
  const width = coverage.clientWidth
  const keyNow = [width, camera, hand, source, part?.first, part?.last, labelRevision].join('/')
  if (!width || coverageKey === keyNow) return
  coverageKey = keyNow
  coverage.width = width * devicePixelRatio
  coverage.height = coverage.clientHeight * devicePixelRatio
  const count = workspace.groups.length
  const scale = coverage.width / count
  const height = coverage.height
  if (part) {
    coverageCtx.fillStyle = '#273d48'
    coverageCtx.fillRect((part.first-1)*scale, 0, (part.last-part.first+1)*scale, height)
  }
  const priority = new Uint8Array(coverage.width)
  for (const [i, group] of workspace.groups.entries()) {
    const id = key(group.sequence, camera, hand)
    const reviewed = workspace.labels[id]
    const proposal = (source === 'sam' ? workspace.proposals : workspace.terra)[id]
    const marker = reviewed?.source === 'manual' ? 3 : reviewed ? 1 : actionable(proposal) ? 2 : 0
    if (!marker) continue
    const end = Math.min(priority.length, Math.ceil((i+1)*scale))
    for (let p = Math.floor(i*scale); p < end; p++) priority[p] = Math.max(priority[p], marker)
  }
  const palette = ['', '#728397', '#c3abfa', '#81e3d9']
  for (let p = 0; p < priority.length; p++) {
    if (!priority[p]) continue
    coverageCtx.fillStyle = palette[priority[p]]
    coverageCtx.fillRect(p, height*.22, 1, height*.56)
  }
}

function modelBusy() {
  return jobRequest || job.state === 'running' || job.state === 'stopping'
}

function updateModel() {
  const sam = source === 'sam'
  const part = range()
  const capable = workspace.model_capabilities[source]
  const cameras = get('sam-cameras').value === 'all' ? workspace.order : [camera]
  const hasSeeds = cameras.some(view => seeds.has(view))
  const hz = Number(get('hz').value)
  const summary = get('run-summary')
  if (!capable) {
    summary.textContent = sam ? 'SAM is available in the model-enabled frame-hand-labeler package.' : 'Install Codex CLI and sign in to use Terra.'
  } else if (!part) {
    summary.textContent = 'Choose a valid group range.'
  } else if (sam) {
    summary.textContent = hasSeeds
      ? `${part.last-part.first+1} target groups · ${cameras.length === 1 ? captions[cameras[0]] : 'All cameras'}`
      : 'Draw a seed box in one of the selected cameras to start tracking.'
  } else {
    const groups = workspace.groups.slice(part.first-1, part.last)
    const begin = groups[0].time_us
    const count = hz > 0 ? new Set(groups.map(group => Math.floor((group.time_us-begin)*hz/1e6+.5))).size : 0
    summary.textContent = `${count} sampled groups from available frames. Completed groups are reused.`
  }
  get('run-model').textContent = sam ? 'Run SAM 3.1' : 'Run Terra'
  get('run-model').disabled = modelBusy() || !part || !capable || (sam && !hasSeeds)
  get('stop-model').hidden = !modelBusy() || jobRequest
  get('stop-model').disabled = job.state === 'stopping'
}

function selectSource(value) {
  source = value
  for (const name of ['sam', 'terra']) {
    const selected = name === value
    get('source-' + name).setAttribute('aria-pressed', String(selected))
    get(name + '-settings').hidden = !selected
    get(name + '-settings').disabled = !selected
  }
  if (workspace) {
    summarize()
    paint()
  }
}

let seeds = new Set()
let labeledCount = 0
function updateLabelCounts() {
  const labels = Object.entries(workspace.labels)
  labeledCount = labels.length
  seeds = new Set(labels.filter(([, label]) => label.source === 'manual').map(([id]) => id.split('/')[1]))
}

function showJob() {
  const detail = get('job-state')
  const name = job.kind === 'sam' ? 'SAM 3.1' : 'Terra'
  detail.textContent = ({
    idle: '', running: `${name} is running…`, stopping: `Stopping ${name}…`,
    complete: `${name} finished.`, cancelled: `${name} stopped.`, failed: `${name} failed.`,
  })[job.state]
  detail.dataset.error = job.state === 'failed'
  const log = get('job-log')
  log.textContent = job.log.join('\n')
  log.hidden = !job.log.length
  log.scrollTop = log.scrollHeight
  updateModel()
}

async function readJSON(path, data) {
  const options = data === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data),
  }
  const response = await fetch(path, options)
  const result = await response.json()
  if (!response.ok) throw new Error(result.error || 'Workspace request failed.')
  return result
}

async function poll() {
  clearTimeout(pollTimer)
  const revision = ++pollRevision
  try {
    const result = await readJSON('/api/model')
    if (revision !== pollRevision) return
    job = result
    showJob()
    if (modelBusy()) {
      pollTimer = setTimeout(poll, 500)
    } else if (job.serial !== loadedJob) {
      if (pending) {
        pollTimer = setTimeout(poll, 300)
        return
      }
      loadedJob = job.serial
      await reload()
    }
  } catch (error) {
    status(error.message, true)
  }
}

async function runJob(path, data) {
  if (jobRequest) return
  clearTimeout(pollTimer)
  pollRevision++
  jobRequest = true
  updateModel()
  try {
    job = await readJSON(path, data)
  } catch (error) {
    status(error.message, true)
  } finally {
    jobRequest = false
    showJob()
    await poll()
  }
}

get('model-form').onsubmit = (event) => {
  event.preventDefault()
  const part = range()
  if (!part || get('run-model').disabled) return
  const data = { kind: source, ...part }
  if (source === 'sam') {
    data.cameras = get('sam-cameras').value === 'all' ? workspace.order : [camera]
    data.radius = Number(get('radius').value)
    data.device = get('device').value
  } else {
    data.hz = Number(get('hz').value)
  }
  runJob('/api/model', data)
}
get('stop-model').onclick = () => runJob('/api/model/stop', {})
get('source-sam').onclick = () => selectSource('sam')
get('source-terra').onclick = () => selectSource('terra')
get('set-first').onclick = () => workspace && setRange(frame+1, Math.max(frame+1, Number(get('last').value)))
get('set-last').onclick = () => workspace && setRange(Math.min(frame+1, Number(get('first').value)||1), frame+1)
get('whole').onclick = () => workspace && setRange(1, workspace.groups.length)
get('model-form').oninput = () => { if (workspace) summarize() }
new ResizeObserver(() => {
  if (!workspace) return
  fitImage()
  drawCoverage()
  paint()
}).observe(coverage)
window.addEventListener('resize', () => {
  if (!workspace) return
  fitImage()
  paint()
})
await reload()
if (workspace) await poll()
