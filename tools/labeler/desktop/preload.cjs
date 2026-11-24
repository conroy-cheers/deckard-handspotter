'use strict';

const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('labeler', {
  choose: (kind) => ipcRenderer.invoke('labeler:choose', kind),
  open: () => ipcRenderer.invoke('labeler:open'),
  importWorkspace: (settings) => ipcRenderer.invoke('labeler:import', settings),
  resume: () => ipcRenderer.invoke('labeler:resume'),
  hasWorkspace: () => ipcRenderer.invoke('labeler:hasWorkspace'),
});
