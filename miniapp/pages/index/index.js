// pages/index/index.js
const api = require('../../utils/api')
const app = getApp()

Page({
  data: {
    loading: false,
    statusText: 'ready'
  },

  onGoMonitor() {
    wx.navigateTo({ url: '/pages/monitor/monitor' })
  },

  onTakePhoto() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['camera'],
      success: (res) => this._handleImage(res.tempFiles[0].tempFilePath)
    })
  },

  onChooseFromAlbum() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album'],
      success: (res) => this._handleImage(res.tempFiles[0].tempFilePath)
    })
  },

  _makeDisplayName() {
    const now = new Date()
    const pad = n => String(n).padStart(2, '0')
    return 'photo_' + now.getFullYear() + '-' + pad(now.getMonth()+1) + '-' + pad(now.getDate()) +
           '_' + pad(now.getHours()) + pad(now.getMinutes()) + pad(now.getSeconds()) + '.jpg'
  },

  _handleImage(filePath) {
    this.setData({ loading: true, statusText: 'compressing' })
    const displayName = this._makeDisplayName()

    wx.compressImage({
      src: filePath,
      quality: 85,
      success: (res) => {
        this.setData({ statusText: 'uploading' })
        api.uploadImage(res.tempFilePath, displayName, 0.25)
          .then((result) => {
            if (result.success) {
              this._saveResults(result)
            } else {
              wx.showToast({ title: '检测失败', icon: 'error' })
            }
          })
          .catch((err) => {
            wx.showToast({ title: '网络错误: ' + (err.message || ''), icon: 'error' })
          })
          .finally(() => {
            this.setData({ loading: false, statusText: 'ready' })
          })
      },
      fail: () => {
        this.setData({ statusText: 'uploading' })
        api.uploadImage(filePath, displayName, 0.25)
          .then((result) => {
            if (result.success) {
              this._saveResults(result)
            } else {
              wx.showToast({ title: '检测失败', icon: 'error' })
            }
          })
          .catch((err) => {
            wx.showToast({ title: '网络错误', icon: 'error' })
          })
          .finally(() => {
            this.setData({ loading: false, statusText: 'ready' })
          })
      }
    })
  },

  _saveResults(result) {
    const ts = Date.now()
    const fs = wx.getFileSystemManager()
    let cleanDone = !result.clean_image_b64
    let annDone = false

    const navigate = () => {
      if (cleanDone && annDone) {
        wx.navigateTo({ url: '/pages/result/result' })
      }
    }

    // save clean image
    if (result.clean_image_b64) {
      const buf = wx.base64ToArrayBuffer(result.clean_image_b64)
      const path = `${wx.env.USER_DATA_PATH}/clean_${ts}.jpg`
      fs.writeFile({
        filePath: path, data: buf,
        success: () => { app.globalData.cleanImagePath = path; cleanDone = true; navigate(); },
        fail: () => { app.globalData.cleanImagePath = ''; cleanDone = true; navigate(); }
      })
    }

    // save annotated image
    const buf = wx.base64ToArrayBuffer(result.annotated_image_b64)
    const path = `${wx.env.USER_DATA_PATH}/annotated_${ts}.jpg`
    fs.writeFile({
      filePath: path, data: buf,
      success: () => { app.globalData.annotatedImagePath = path; annDone = true; navigate(); },
      fail: () => { app.globalData.annotatedImagePath = ''; annDone = true; navigate(); }
    })

    app.globalData.result = result
    navigate()
  }
})
