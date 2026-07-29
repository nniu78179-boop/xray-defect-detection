// pages/result/result.js
const app = getApp()

Page({
  data: {
    showAnnotated: true,
    imagePath: '',
    filename: '',
    defectCount: 0,
    inferenceTimeMs: 0,
    detections: []
  },

  onLoad() {
    const result = app.globalData.result
    const annotatedPath = app.globalData.annotatedImagePath
    const cleanPath = app.globalData.cleanImagePath

    if (!result) {
      wx.showToast({ title: '无检测结果', icon: 'error' })
      setTimeout(() => wx.navigateBack(), 1500)
      return
    }

    const detections = (result.detections || []).map(det => {
      const pct = (det.confidence * 100).toFixed(1)
      let barColor = '#2e9d57'
      if (det.confidence <= 0.2) barColor = '#c44536'
      else if (det.confidence <= 0.5) barColor = '#d68c00'
      return { ...det, barWidth: pct, barColor: barColor }
    })

    this.setData({
      cleanImagePath: cleanPath || '',
      annotatedImagePath: annotatedPath || '',
      imagePath: annotatedPath || '',
      filename: result.filename,
      defectCount: result.defect_count,
      inferenceTimeMs: result.inference_time_ms,
      detections: detections
    })
  },

  onToggle() {
    const show = !this.data.showAnnotated
    this.setData({
      showAnnotated: show,
      imagePath: show ? this.data.annotatedImagePath : this.data.cleanImagePath
    })
  },

  onBack() {
    wx.navigateBack()
  }
})
