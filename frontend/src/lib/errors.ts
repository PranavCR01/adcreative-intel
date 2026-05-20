export const ERROR_MESSAGES: Record<string | number, string> = {
  401: "Authentication failed. Please refresh the page.",
  404: "Creative not found. Please upload again.",
  422: "Invalid request. Please check your input.",
  500: "Something went wrong on our end. Please try again.",
  503: "The model is warming up. Please wait 30 seconds and try again.",
  default: "Something went wrong. Please try again.",
  model_warming: "Model is warming up (~30 seconds). Hang tight.",
  upload_failed: "Upload failed. Please try a JPG or PNG under 5MB.",
}
