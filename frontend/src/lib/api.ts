import { ERROR_MESSAGES } from './errors'
import { resizeImage } from './imageUtils'

const BACKEND_URL = import.meta.env.VITE_API_URL
if (!BACKEND_URL) console.warn('VITE_API_URL is not set — upload requests will hit relative /upload')

class APIError extends Error {
  status: number
  constructor(status: number) { super(`API error ${status}`); this.status = status }
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))

export async function fetchWithRetry(
  url: string,
  options: RequestInit = {},
  retries = 3,
  delay = 3000,
): Promise<Response> {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const res = await fetch(url, options)
      if (res.ok) return res
      if (res.status >= 400 && res.status < 500) throw new APIError(res.status)
      if (attempt < retries - 1) await sleep(delay)
    } catch (e) {
      if (e instanceof APIError) throw e
      if (attempt === retries - 1) throw e
      await sleep(delay)
    }
  }
  throw new Error('Max retries exceeded')
}

export function getUserMessage(error: unknown): string {
  if (error instanceof APIError) {
    return ERROR_MESSAGES[error.status] ?? ERROR_MESSAGES.default
  }
  return ERROR_MESSAGES.default
}

export async function uploadCreative(file: File, vertical: string) {
  const uploadId = crypto.randomUUID()
  const resized = await resizeImage(file, 224)
  const formData = new FormData()
  formData.append('image', resized, 'image.jpg')
  formData.append('vertical', vertical)
  formData.append('upload_id', uploadId)
  const res = await fetchWithRetry(`${BACKEND_URL}/upload`, {
    method: 'POST',
    body: formData,
  })
  const result = await res.json()
  return { uploadId, ...result }
}

export async function sendChat(imageId: string, message: string, vertical = 'gaming') {
  const res = await fetchWithRetry(`${BACKEND_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: imageId, message, vertical }),
  })
  return res.json()
}
