'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('anyhand', {
  viewport: (job, index, mode) => ipcRenderer.invoke("studio:viewport", {job, index, mode}),
  detectorResult: version => ipcRenderer.invoke('studio:detector-result',version),
  garments: path => ipcRenderer.invoke('studio:garments',path),
  batchFields: () => ipcRenderer.invoke("studio:batch-fields"),
  pickPath: (kind, multiple) => ipcRenderer.invoke("studio:batch-path", {kind, multiple}),
  openJob: () => ipcRenderer.invoke("studio:batch-open"),
  saveJob: job => ipcRenderer.invoke("studio:batch-save", job),
  renderBatch: job => ipcRenderer.invoke("studio:batch-run", job),
  previewBatch: (index, mode) => ipcRenderer.invoke("studio:batch-preview", {index, mode}),
});
