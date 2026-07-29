// pages/monitor/monitor.js
const app = getApp()

Page({
  data: {
    connected: false,
    hasResult: false,
    showAnnotated: true,
    imagePath: '',
    cleanImagePath: '',
    annotatedImagePath: '',
    filename: '',
    defectCount: 0,
    inferenceTimeMs: 0,
    detectTime: '',
    detections: [],
    lastCheck: ''
  },

  onLoad() {
    this.startPolling()
  },

  onUnload() {
    clearInterval(this._timer)
  },

  onToggle() {
    const show = !this.data.showAnnotated
    this.setData({
      showAnnotated: show,
      imagePath: show ? this.data.annotatedImagePath : this.data.cleanImagePath
    })
  },

  startPolling() {
    const baseUrl = app.globalData.apiBaseUrl
    const doPoll = () => {
      wx.request({
        url: baseUrl + '/api/monitor/latest',
        method: 'GET',
        header: { 'ngrok-skip-browser-warning': 'true' },
        success: (res) => {
          const d = res.data
          if (!d.has_result || d.filename === this.data.filename) {
            this.setData({ connected: true, lastCheck: this._timeNow() })
            return
          }
          // new result
          const detections = (d.detections || []).map(det => {
            const pct = (det.confidence * 100).toFixed(1)
            let barColor = '#2e9d57'
            if (det.confidence <= 0.2) barColor = '#c44536'
            else if (det.confidence <= 0.5) barColor = '#d68c00'
            return { ...det, barWidth: pct, barColor: barColor }
          })

          const fs = wx.getFileSystemManager()
          const ts = Date.now()
          const show = this.data.showAnnotated

          // save clean image
          if (d.clean_b64) {
            const buf = wx.base64ToArrayBuffer(d.clean_b64)
            const cleanPath = `${wx.env.USER_DATA_PATH}/auto_clean_${ts}.jpg`
            fs.writeFile({ filePath: cleanPath, data: buf,
              success: () => this.setData({ cleanImagePath: cleanPath, imagePath: show ? this.data.annotatedImagePath : cleanPath })
            })
          }

          // save annotated image
          if (d.annotated_b64) {
            const buf = wx.base64ToArrayBuffer(d.annotated_b64)
            const annPath = `${wx.env.USER_DATA_PATH}/auto_ann_${ts}.jpg`
            fs.writeFile({ filePath: annPath, data: buf,
              success: () => this.setData({ annotatedImagePath: annPath, imagePath: show ? annPath : (this.data.cleanImagePath || annPath) })
            })
          }

          this.setData({
            connected: true,
            hasResult: true,
            filename: d.filename,
            defectCount: d.defect_count,
            inferenceTimeMs: d.inference_time_ms,
            detectTime: d.time,
            detections: detections,
            lastCheck: this._timeNow()
          })
        },
        fail: () => {
          this.setData({ connected: false, lastCheck: this._timeNow() })
        }
      })
    }
    doPoll()
    this._timer = setInterval(doPoll, 1000)
  },

  _timeNow() {
    const now = new Date()
    const pad = n => String(n).padStart(2, '0')
    return pad(now.getHours()) + ':' + pad(now.getMinutes()) + ':' + pad(now.getSeconds())
  }
})
