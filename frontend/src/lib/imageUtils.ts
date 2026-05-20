export async function resizeImage(file: File, size: number): Promise<Blob> {
  return new Promise((resolve) => {
    const img = new Image()
    const url = URL.createObjectURL(file)
    img.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = size
      canvas.height = size
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(img, 0, 0, size, size)
      canvas.toBlob((blob) => {
        URL.revokeObjectURL(url)
        resolve(blob!)
      }, 'image/jpeg', 0.92)
    }
    img.src = url
  })
}
