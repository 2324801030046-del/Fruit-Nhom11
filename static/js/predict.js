const form = document.querySelector("#recognize-form");
const photo = document.querySelector("#photo");
const dropZone = document.querySelector("#drop-zone");
const uploadMode = document.querySelector("#upload-mode");
const cameraMode = document.querySelector("#camera-mode");
const picker = document.querySelector("#choose-label");
const cameraControls = document.querySelector("#camera-controls");
const camera = document.querySelector("#camera");
const capture = document.querySelector("#capture");
const retake = document.querySelector("#retake");
const video = document.querySelector("#video");
const cameraPanel = document.querySelector("#camera-panel");
const cameraPrediction = document.querySelector("#camera-prediction");
const preview = document.querySelector("#preview");
const previewPanel = document.querySelector("#preview-panel");
const filename = document.querySelector("#filename");
const remove = document.querySelector("#remove");
const submit = document.querySelector("#submit");
const status = document.querySelector("#status");
const result = document.querySelector("#result");
const resultPhoto = document.querySelector("#result-photo");
const heading = document.querySelector("#result-heading");
const fruitDetails = document.querySelector("#fruit-details");
const confidence = document.querySelector("#confidence");
const ranking = document.querySelector("#ranking");
const percent = new Intl.NumberFormat("vi-VN", {style: "percent", maximumFractionDigits: 1});
const livePredictionInterval = 700;
const noFruitMessage = "Không phát hiện trái cây";
let stream;
let selection;
let previewUrl;
let busy = false;
let cameraRequest = 0;
let livePredictionController;
let livePredictionTimer;

function announce(message = "", error = false) {
  status.textContent = message;
  status.dataset.error = error;
}

function clearResult() {
  result.hidden = true;
  resultPhoto.removeAttribute("src");
}

function setBusy(value) {
  busy = value;
  photo.disabled = retake.disabled = camera.disabled = uploadMode.disabled = cameraMode.disabled = value;
  picker.classList.toggle("disabled", value);
  capture.disabled = value || !stream || !video.videoWidth;
  submit.disabled = remove.disabled = value || !selection;
}

function stopCamera() {
  cameraRequest++;
  clearTimeout(livePredictionTimer);
  livePredictionController?.abort();
  livePredictionController = null;
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  cameraPanel.hidden = capture.hidden = true;
  cameraPrediction.textContent = "";
  capture.disabled = true;
  camera.disabled = retake.disabled = busy;
  camera.textContent = "Mở camera";
}

function selectFile(file, fromCamera = false) {
  clearResult();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  preview.removeAttribute("src");
  selection = null;
  previewUrl = null;
  previewPanel.hidden = retake.hidden = true;
  submit.disabled = remove.disabled = true;
  if (!file) return announce();
  if (file.size > Number(photo.dataset.maxBytes)) return announce("Ảnh vượt quá 5 MB.", true);
  if (!/\.(jpe?g|png|webp)$/i.test(file.name)) return announce("Hãy chọn ảnh JPG, PNG hoặc WebP.", true);

  selection = file;
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  filename.textContent = file.name;
  previewPanel.hidden = false;
  retake.hidden = !fromCamera;
  submit.disabled = remove.disabled = busy;
  announce();
}

function setCameraMode(enabled) {
  if (cameraControls.hidden === !enabled) return;
  uploadMode.setAttribute("aria-pressed", !enabled);
  cameraMode.setAttribute("aria-pressed", enabled);
  dropZone.hidden = enabled;
  cameraControls.hidden = !enabled;
  if (!enabled) stopCamera();
  announce();
}

uploadMode.addEventListener("click", () => setCameraMode(false));
cameraMode.addEventListener("click", () => setCameraMode(true));

function chooseFile(file) {
  if (busy || !file) return;
  stopCamera();
  setBusy(false);
  selectFile(file);
}

photo.addEventListener("change", () => {
  chooseFile(photo.files[0]);
  photo.value = "";
});
dropZone.addEventListener("dragover", event => {
  event.preventDefault();
  if (!busy) dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", event => {
  if (!dropZone.contains(event.relatedTarget)) dropZone.classList.remove("dragging");
});
dropZone.addEventListener("drop", event => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  chooseFile(event.dataTransfer.files[0]);
});
window.addEventListener("dragover", event => {
  if (event.dataTransfer.types.includes("Files")) event.preventDefault();
});
window.addEventListener("drop", event => {
  if (event.dataTransfer.types.includes("Files")) event.preventDefault();
});

async function openCamera() {
  if (!window.isSecureContext) {
    return announce("Camera cần HTTPS khi mở từ điện thoại qua mạng LAN. Bạn vẫn có thể tải ảnh lên.", true);
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    return announce("Trình duyệt này không hỗ trợ camera. Bạn có thể tải ảnh lên.", true);
  }
  const request = ++cameraRequest;
  camera.disabled = retake.disabled = true;
  announce("Đang yêu cầu quyền dùng camera…");
  try {
    const nextStream = await navigator.mediaDevices.getUserMedia({video: {facingMode: {ideal: "environment"}}, audio: false});
    if (request !== cameraRequest) {
      nextStream.getTracks().forEach(track => track.stop());
      return;
    }
    stream = nextStream;
    stream.getVideoTracks()[0]?.addEventListener("ended", () => {
      if (request !== cameraRequest) return;
      stopCamera();
      announce("Camera đã ngắt kết nối. Bạn có thể mở lại hoặc tải ảnh lên.", true);
    });
    selectFile(null);
    video.muted = true;
    video.srcObject = stream;
    cameraPanel.hidden = capture.hidden = false;
    cameraPrediction.textContent = "Đang nhận hình ảnh…";
    camera.textContent = "Tắt camera";
    camera.disabled = busy;
    await video.play();
    if (request !== cameraRequest) return;
    capture.disabled = busy || !video.videoWidth;
    cameraPrediction.textContent = "Đang dự đoán…";
    livePredictionController = new AbortController();
    predictLiveFrame(document.createElement("canvas"), livePredictionController.signal);
  } catch (error) {
    if (request !== cameraRequest) return;
    const messages = {
      NotAllowedError: "Quyền camera bị từ chối. Hãy cấp quyền hoặc tải ảnh lên.",
      NotFoundError: "Không tìm thấy camera. Bạn có thể tải ảnh lên.",
      NotReadableError: "Không đọc được camera. Hãy đóng ứng dụng khác đang dùng camera rồi thử lại.",
      AbortError: "Không mở được camera. Hãy kiểm tra thiết bị rồi thử lại.",
    };
    if (stream) stopCamera();
    announce(messages[error.name] || "Không mở được camera. Bạn có thể tải ảnh lên.", true);
  } finally {
    if (request === cameraRequest) camera.disabled = retake.disabled = busy;
  }
}

camera.addEventListener("click", () => stream ? stopCamera() : openCamera());
retake.addEventListener("click", openCamera);

async function captureFrame(canvas, maxSize, quality) {
  const scale = Math.min(1, maxSize / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", quality));
}

async function requestPrediction(image, filename, signal) {
  const body = new FormData();
  body.append("photo", image, filename);
  const response = await fetch(form.action, {method: "POST", body, signal});
  if (response.status === 401) window.location.reload();
  const data = await response.json().catch(() => null);
  if (!response.ok || (!data?.non_fruit && !data?.top3?.length)) {
    throw new Error(data?.error || "Máy chủ chưa trả về kết quả. Vui lòng thử lại.");
  }
  return data;
}

async function predictLiveFrame(canvas, signal) {
  try {
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    const blob = await captureFrame(canvas, 640, .85);
    if (signal.aborted) return;
    if (!blob) throw new Error("Không tạo được ảnh từ camera.");
    const data = await requestPrediction(blob, "anh-camera.jpg", signal);
    if (signal.aborted) return;
    if (data.non_fruit) {
      cameraPrediction.textContent = noFruitMessage;
      return;
    }
    const best = data.top3[0];
    const prefix = data.certain ? "Dự đoán: " : "Có thể là: ";
    cameraPrediction.textContent = `${prefix}${best.label.toLocaleUpperCase("vi-VN")} (${percent.format(best.probability)})`;
  } catch {
    if (!signal.aborted) cameraPrediction.textContent = "Chưa thể dự đoán. Đang thử lại…";
  } finally {
    if (!signal.aborted) {
      livePredictionTimer = setTimeout(() => predictLiveFrame(canvas, signal), livePredictionInterval);
    }
  }
}

capture.addEventListener("click", async () => {
  if (!stream || !video.videoWidth) return;
  const request = cameraRequest;
  capture.disabled = true;
  const blob = await captureFrame(document.createElement("canvas"), 1600, .9).catch(() => null);
  if (request !== cameraRequest) return;
  if (!blob) {
    capture.disabled = false;
    return announce("Không tạo được ảnh từ camera. Hãy chụp lại.", true);
  }
  stopCamera();
  selectFile(new File([blob], "anh-camera.jpg", {type: "image/jpeg"}), true);
});

function displayName(item) {
  return item.english && item.english !== item.label ? item.label + " (" + item.english + ")" : item.label;
}

function showFruitRanking(data) {
  const best = data.top3[0];
  heading.textContent = data.certain ? displayName(best) : "Chưa thể nhận diện chắc chắn. Vui lòng thử ảnh rõ hơn.";
  confidence.textContent = "Độ tin cậy cao nhất " + percent.format(best.probability);
  ranking.replaceChildren(...data.top3.map(item => {
    const li = document.createElement("li");
    const line = document.createElement("div");
    line.className = "score-line";
    const label = document.createElement("span");
    label.textContent = displayName(item);
    const score = document.createElement("strong");
    score.textContent = percent.format(item.probability);
    const bar = document.createElement("progress");
    bar.max = 1;
    bar.value = item.probability;
    bar.setAttribute("aria-label", item.label);
    line.append(label, score);
    li.append(line, bar);
    return li;
  }));
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (!selection || busy) return;
  setBusy(true);
  clearResult();
  announce("Đang nhận diện…");
  try {
    const data = await requestPrediction(selection, selection.name);
    fruitDetails.hidden = Boolean(data.non_fruit);
    if (data.non_fruit) heading.textContent = noFruitMessage;
    else showFruitRanking(data);
    resultPhoto.src = previewUrl;
    result.hidden = false;
    announce();
    heading.focus();
  } catch (error) {
    announce(error instanceof TypeError ? "Không kết nối được máy chủ. Hãy thử lại." : error.message, true);
  } finally {
    setBusy(false);
  }
});

function reset() {
  stopCamera();
  selectFile(null);
  if (cameraControls.hidden) photo.focus();
  else camera.focus();
}
remove.addEventListener("click", reset);
document.querySelector("#reset").addEventListener("click", reset);
window.addEventListener("pagehide", () => {
  stopCamera();
});
