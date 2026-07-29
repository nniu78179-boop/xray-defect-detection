const app = getApp()

function getBaseUrl() {
  return app.globalData.apiBaseUrl
}

function checkHealth() {
  return new Promise((resolve, reject) => {
    wx.request({
      url: getBaseUrl() + '/api/health',
      method: 'GET',
      header: {
        'ngrok-skip-browser-warning': 'true'
      },
      success(res) {
        resolve(res.data)
      },
      fail(err) {
        reject(err)
      }
    })
  })
}

function uploadImage(filePath, displayName, confidence = 0.25) {
  return new Promise((resolve, reject) => {
    wx.uploadFile({
      url: getBaseUrl() + '/api/infer',
      filePath: filePath,
      name: 'file',
      header: {
        'ngrok-skip-browser-warning': 'true'
      },
      formData: {
        confidence: String(confidence),
        display_name: displayName
      },
      success(res) {
        try {
          const data = JSON.parse(res.data)
          resolve(data)
        } catch (e) {
          reject(new Error('Invalid response from server'))
        }
      },
      fail(err) {
        reject(err)
      }
    })
  })
}

module.exports = {
  checkHealth,
  uploadImage
}
