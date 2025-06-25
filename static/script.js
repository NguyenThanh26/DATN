let subtitleEnabled = true;
let desiredTime = 0;
let seekInterval = null;

// Hiển thị trang được chọn
function showPage(pageNumber) {
    const pageVideos = document.getElementById('page-videos');
    const pageUpload = document.getElementById('page-upload');
    
    if (pageNumber === 1) {
        pageVideos.classList.remove('hidden');
        pageUpload.classList.add('hidden');
        fetchProcessedVideos();
    } else {
        pageVideos.classList.add('hidden');
        pageUpload.classList.remove('hidden');
    }
}

// Lấy danh sách video đã xử lý
async function fetchProcessedVideos() {
    const videoGrid = document.getElementById('video-grid');
    try {
        const response = await fetch('/list_processed_videos');
        const data = await response.json();
        videoGrid.innerHTML = '';
        if (data.videos.length === 0) {
            videoGrid.innerHTML = '<p>Chưa có video nào được xử lý.</p>';
            return;
        }
        data.videos.forEach((video, index) => {
            const videoItem = document.createElement('div');
            videoItem.className = 'video-item';
            let videoHtml = `
                <video class="grid-video" controls preload="metadata" id="video-${index}">
                    <source src="${video.path}" type="video/mp4">
                    Trình duyệt của bạn không hỗ trợ thẻ video.
            `;
            // Thêm phụ đề nếu có
            if (video.subtitle_path !== 'none') {
                const subtitleLang = video.language;
                videoHtml += `
                    <track id="subtitle-track-${index}" src="/download/${video.subtitle_path}" kind="subtitles" srclang="${subtitleLang}" label="${subtitleLang === 'zh' ? '中文' : subtitleLang === 'ko' ? '한국어' : subtitleLang === 'vi' ? 'Tiếng Việt' : 'English'}" default>
                `;
            }
            videoHtml += `</video>`;
            videoItem.innerHTML = `
                ${videoHtml}
                <div class="video-info">
                    <p>${video.filename} (Ngôn ngữ: ${video.language})</p>
                    <div class="video-controls">
                        ${video.subtitle_path !== 'none' ? `<button class="btn btn-secondary" onclick="toggleGridSubtitle(${index})" id="toggle-subtitle-${index}">Tắt phụ đề</button>` : ''}
                        <a href="${video.path}" download class="btn btn-primary">Tải video</a>
                        ${video.subtitle_path !== 'none' ? `<a href="/download/${video.subtitle_path}" download class="btn btn-primary">Tải phụ đề</a>` : ''}
                    </div>
                </div>
            `;
            videoGrid.appendChild(videoItem);
            // Bật phụ đề mặc định
            if (video.subtitle_path !== 'none') {
                const track = document.getElementById(`subtitle-track-${index}`);
                if (track && track.track) {
                    track.track.mode = 'showing';
                }
            }
        });
    } catch (error) {
        console.error('Error fetching processed videos:', error);
        videoGrid.innerHTML = '<p>Lỗi khi tải danh sách video.</p>';
    }
}

// Bật/tắt phụ đề cho video trong lưới
function toggleGridSubtitle(index) {
    const track = document.getElementById(`subtitle-track-${index}`);
    const button = document.getElementById(`toggle-subtitle-${index}`);
    if (track && track.track) {
        const isShowing = track.track.mode === 'showing';
        track.track.mode = isShowing ? 'hidden' : 'showing';
        button.textContent = isShowing ? 'Bật phụ đề' : 'Tắt phụ đề';
    }
}

// Xử lý form
document.getElementById('subtitle-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('file', document.getElementById('video-file').files[0]);
    const originLanguage = document.getElementById('origin-language').value;
    const translateLanguage = document.getElementById('translate-language').value;
    const useCorrection = document.getElementById('use-correction').checked;
    const embedSubtitle = document.getElementById('embed-subtitle').value;
    
    const resultDiv = document.getElementById('process-result');
    const videoPlayer = document.getElementById('video-player');
    const videoSection = document.getElementById('video-section');
    const resetButton = document.getElementById('reset-button');
    const toggleSubtitleButton = document.getElementById('toggle-subtitle');
    const rewindButton = document.getElementById('rewind-button');
    const fastForwardButton = document.getElementById('fast-forward-button');
    const subtitleInfo = document.getElementById('subtitle-info');
    const processLoading = document.getElementById('process-loading');
    
    resultDiv.innerHTML = '';
    processLoading.style.display = 'block';
    videoSection.classList.add('hidden');
    resetButton.classList.add('hidden');
    toggleSubtitleButton.classList.add('hidden');
    rewindButton.classList.add('hidden');
    fastForwardButton.classList.add('hidden');
    subtitleInfo.innerHTML = '';
    if (seekInterval) clearInterval(seekInterval);
    
    try {
        const response = await fetch(`/process?origin_language=${originLanguage}&translate_language=${translateLanguage}&use_correction=${useCorrection}${embedSubtitle ? `&embed_subtitle=${embedSubtitle}` : ''}`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        processLoading.style.display = 'none';
        if (response.ok) {
            let resultHtml = `${data.message}<br><a href="/download/${data.subtitle_path}">Tải file phụ đề gốc</a>`;
            if (data.translated_subtitle_path) {
                resultHtml += `<br><a href="/download/${data.translated_subtitle_path}">Tải file phụ đề ${data.translated_subtitle_path.includes('_corrected') ? 'đã hiệu chỉnh' : 'đã dịch'}</a>`;
            }
            if (data.video_with_subtitle) {
                resultHtml += `<br><a href="${data.video_url}">Tải video có phụ đề</a>`;
                videoPlayer.innerHTML = '';
                if (embedSubtitle === 'soft' && (data.translated_subtitle_path || data.subtitle_path)) {
                    const subtitlePath = data.translated_subtitle_path || data.subtitle_path;
                    const subtitleLang = data.translated_subtitle_path ? translateLanguage : originLanguage;
                    videoPlayer.innerHTML = `
                        <source src="${data.video_url}" type="video/mp4">
                        <track id="subtitle-track" src="/download/${subtitlePath}" kind="subtitles" srclang="${subtitleLang}" label="${subtitleLang === 'zh' ? '中文' : subtitleLang === 'ko' ? '한국어' : subtitleLang === 'vi' ? 'Tiếng Việt' : 'English'}" default>
                        Trình duyệt của bạn không hỗ trợ thẻ video.
                    `;
                    subtitleEnabled = true;
                    toggleSubtitleButton.textContent = 'Tắt phụ đề';
                    toggleSubtitleButton.classList.remove('hidden');
                    // Bật phụ đề mặc định
                    const track = videoPlayer.querySelector('track');
                    if (track && track.track) {
                        track.track.mode = 'showing';
                    }
                } else {
                    videoPlayer.innerHTML = `
                        <source src="${data.video_url}" type="video/mp4">
                        Trình duyệt của bạn không hỗ trợ thẻ video.
                    `;
                    toggleSubtitleButton.classList.add('hidden');
                }
                videoPlayer.load();
                videoSection.classList.remove('hidden');
                resetButton.classList.remove('hidden');
                rewindButton.classList.remove('hidden');
                fastForwardButton.classList.remove('hidden');
                subtitleInfo.innerHTML = `Phụ đề đã được nhúng ${embedSubtitle === 'hard' ? 'cứng' : embedSubtitle === 'soft' ? 'mềm' : 'không nhúng'}.`;
            } else {
                subtitleInfo.innerHTML = '<br><span style="color: red;">Không tạo được video có phụ đề.</span>';
            }
            resultDiv.innerHTML = resultHtml;
        } else {
            resultDiv.innerHTML = `Lỗi: ${data.detail}`;
        }
    } catch (error) {
        processLoading.style.display = 'none';
        resultDiv.innerHTML = `Lỗi: ${error.message}`;
    }
});

// Bật/tắt phụ đề cho video player ở Trang 2
function toggleSubtitle() {
    const videoPlayer = document.getElementById('video-player');
    const toggleSubtitleButton = document.getElementById('toggle-subtitle');
    const subtitleTrack = document.getElementById('subtitle-track');
    
    if (subtitleTrack) {
        subtitleEnabled = !subtitleEnabled;
        subtitleTrack.track.mode = subtitleEnabled ? 'showing' : 'hidden';
        toggleSubtitleButton.textContent = subtitleEnabled ? 'Tắt phụ đề' : 'Bật phụ đề';
    }
}

function seekVideo(newTime) {
    const videoPlayer = document.getElementById('video-player');
    const subtitleInfo = document.getElementById('subtitle-info');
    const wasPlaying = !videoPlayer.paused;
    if (videoPlayer.readyState >= 2) {
        const validTime = Math.max(0, Math.min(newTime, videoPlayer.duration || Infinity));
        if (!isNaN(validTime)) {
            videoPlayer.currentTime = validTime;
            if (wasPlaying) {
                videoPlayer.play();
            }
        } else {
            subtitleInfo.innerHTML += '<br><span style="color: orange;">Không thể tua. Đang chờ video tải...</span>';
        }
    } else {
        desiredTime = newTime;
        subtitleInfo.innerHTML += '<br><span style="color: orange;">Video chưa sẵn sàng. Đang chờ tải...</span>';
        if (!seekInterval) {
            seekInterval = setInterval(() => {
                if (videoPlayer.readyState >= 2) {
                    const validTime = Math.max(0, Math.min(desiredTime, videoPlayer.duration || Infinity));
                    if (!isNaN(validTime)) {
                        videoPlayer.currentTime = validTime;
                        if (wasPlaying) {
                            videoPlayer.play();
                        }
                        clearInterval(seekInterval);
                        seekInterval = null;
                        desiredTime = 0;
                    }
                }
            }, 500);
        }
    }
}

function rewindVideo() {
    const videoPlayer = document.getElementById('video-player');
    const newTime = Math.max(0, videoPlayer.currentTime - 10);
    seekVideo(newTime);
}

function fastForwardVideo() {
    const videoPlayer = document.getElementById('video-player');
    const newTime = Math.min(videoPlayer.duration || Infinity, videoPlayer.currentTime + 10);
    seekVideo(newTime);
}

function resetVideoPlayer() {
    const videoPlayer = document.getElementById('video-player');
    const videoSection = document.getElementById('video-section');
    const resetButton = document.getElementById('reset-button');
    const toggleSubtitleButton = document.getElementById('toggle-subtitle');
    const rewindButton = document.getElementById('rewind-button');
    const fastForwardButton = document.getElementById('fast-forward-button');
    const subtitleInfo = document.getElementById('subtitle-info');
    
    videoPlayer.pause();
    videoPlayer.removeAttribute('src');
    videoPlayer.innerHTML = '';
    videoPlayer.load();
    videoSection.classList.add('hidden');
    resetButton.classList.add('hidden');
    toggleSubtitleButton.classList.add('hidden');
    rewindButton.classList.add('hidden');
    fastForwardButton.classList.add('hidden');
    subtitleInfo.innerHTML = '';
    document.getElementById('process-result').innerHTML = '';
    document.getElementById('video-file').value = '';
    if (seekInterval) {
        clearInterval(seekInterval);
        seekInterval = null;
    }
}

// Tải danh sách video khi trang được tải
window.onload = () => {
    showPage(1);
};