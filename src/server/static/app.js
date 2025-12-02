/**
 * AnomalyCLIP Dashboard - Alpine.js State Management
 * Frame-synchronized video anomaly detection dashboard
 */

function dashboardState() {
  return {
    // Core state
    selectedFile: null,
    videoUrl: null,
    processing: false,
    isPlaying: false,
    videoFps: 25,
    currentFrameIndex: 0,
    totalFrames: 0,
    statusMessage: 'Initializing...',
    
    // Frame data buffer - indexed by frame number
    frameDataBuffer: {},
    currentFrameData: null,
    
    // Statistics
    anomalyCount: 0,
    anomalyHistory: [],
    peakAnomalyScore: 0,
    alerts: [],
    summary: '',
    
    // Video Summary (from Ollama)
    videoSummary: null,

    
    // Notification permission
    notificationsEnabled: false,
    
    // Computed properties
    get anomalyPercentage() {
      return this.totalFrames > 0 ? (this.anomalyCount / this.totalFrames) * 100 : 0;
    },
    
    get sortedAnomalyHistory() {
      return [...this.anomalyHistory].sort((a, b) => b.score - a.score);
    },
    
    get sortedAlerts() {
      const priorityOrder = { 'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3 };
      return [...this.alerts].sort((a, b) => {
        // First sort by priority
        const priorityDiff = priorityOrder[a.priority] - priorityOrder[b.priority];
        if (priorityDiff !== 0) return priorityDiff;
        // If same priority, sort by score (highest first)
        return b.score - a.score;
      });
    },
    
    /**
     * Initialize Alpine component
     */
    init() {
      console.log('AnomalyCLIP Dashboard initialized');
      this.requestNotificationPermission();
    },
    
    /**
     * Request browser notification permission
     */
    async requestNotificationPermission() {
      if (!('Notification' in window)) {
        console.warn('Browser does not support notifications');
        return;
      }
      
      if (Notification.permission === 'granted') {
        this.notificationsEnabled = true;
        console.log('✅ Notifications enabled');
      } else if (Notification.permission !== 'denied') {
        const permission = await Notification.requestPermission();
        this.notificationsEnabled = (permission === 'granted');
        console.log(`Notification permission: ${permission}`);
      }
    },
    
    /**
     * Get risk priority and message for anomaly type
     */
    getAnomalyRiskInfo(anomalyType, score) {
      const riskProfiles = {
        'Shooting': { priority: 'CRITICAL', icon: '🔫', color: '#dc2626', message: 'Active shooter detected' },
        'Explosion': { priority: 'CRITICAL', icon: '💥', color: '#dc2626', message: 'Explosion detected' },
        'Assault': { priority: 'HIGH', icon: '⚠️', color: '#ea580c', message: 'Physical assault in progress' },
        'Fighting': { priority: 'HIGH', icon: '🥊', color: '#ea580c', message: 'Violent altercation detected' },
        'Abuse': { priority: 'HIGH', icon: '🚨', color: '#ea580c', message: 'Abuse incident detected' },
        'Robbery': { priority: 'HIGH', icon: '💰', color: '#f59e0b', message: 'Robbery in progress' },
        'Burglary': { priority: 'MEDIUM', icon: '🏠', color: '#f59e0b', message: 'Burglary detected' },
        'Arson': { priority: 'CRITICAL', icon: '🔥', color: '#dc2626', message: 'Arson/fire detected' },
        'Arrest': { priority: 'MEDIUM', icon: '👮', color: '#3b82f6', message: 'Arrest in progress' },
        'RoadAccidents': { priority: 'HIGH', icon: '🚗', color: '#ea580c', message: 'Traffic accident detected' },
        'Shoplifting': { priority: 'LOW', icon: '🛒', color: '#facc15', message: 'Shoplifting suspected' },
        'Stealing': { priority: 'MEDIUM', icon: '👜', color: '#f59e0b', message: 'Theft in progress' },
        'Vandalism': { priority: 'LOW', icon: '🔨', color: '#facc15', message: 'Vandalism detected' },
      };
      
      const profile = riskProfiles[anomalyType] || {
        priority: 'MEDIUM',
        icon: '⚠️',
        color: '#f59e0b',
        message: `${anomalyType} detected`
      };
      
      // Add confidence level to message
      const confidence = score >= 0.8 ? 'High confidence' : score >= 0.6 ? 'Medium confidence' : 'Low confidence';
      profile.fullMessage = `${profile.message} (${confidence}, score: ${score.toFixed(2)})`;
      
      return profile;
    },
    
    /**
     * Show browser push notification
     */
    showPushNotification(anomalyType, score, frameIndex) {
      if (!this.notificationsEnabled || Notification.permission !== 'granted') {
        return;
      }
      
      const riskInfo = this.getAnomalyRiskInfo(anomalyType, score);
      
      const notification = new Notification(`${riskInfo.icon} ${riskInfo.priority} ALERT`, {
        body: `${riskInfo.fullMessage}\nFrame: ${frameIndex}\nTimestamp: ${(frameIndex / this.videoFps).toFixed(1)}s`,
        icon: '/static/anomaly-icon.png', // Optional: add icon file
        badge: '/static/badge-icon.png',  // Optional: add badge file
        tag: `anomaly-${frameIndex}`,     // Prevent duplicate notifications
        requireInteraction: riskInfo.priority === 'CRITICAL', // Critical alerts require user action
        silent: false,
        vibrate: riskInfo.priority === 'CRITICAL' ? [200, 100, 200] : [200],
      });
      
      // Click handler - seek to anomaly frame
      notification.onclick = () => {
        window.focus();
        const video = this.$refs.videoPlayer;
        if (video) {
          video.currentTime = frameIndex / this.videoFps;
          video.play();
        }
        notification.close();
      };
      
      // Auto-close after 10 seconds for non-critical alerts
      if (riskInfo.priority !== 'CRITICAL') {
        setTimeout(() => notification.close(), 10000);
      }
      
      console.log(`🔔 Push notification: ${riskInfo.priority} - ${riskInfo.message}`);
    },
    
    /**
     * Handle video upload and start processing
     */
    async handleUpload() {
      if (!this.selectedFile) {
        alert('Please select a video file');
        return;
      }
      
      // Validate MP4
      const isMP4 = this.selectedFile.type === 'video/mp4' || 
                    (this.selectedFile.name || '').toLowerCase().endsWith('.mp4');
      if (!isMP4) {
        alert('Please upload an MP4 video file');
        console.warn('Invalid file type:', this.selectedFile.type);
        return;
      }
      
      console.log('Starting upload:', this.selectedFile.name);
      this.processing = true;
      this.statusMessage = 'Uploading video...';
      
      // Reset state
      this.frameDataBuffer = {};
      this.currentFrameData = null;
      this.anomalyCount = 0;
      this.anomalyHistory = [];
      this.peakAnomalyScore = 0;
      this.alerts = [];
      this.totalFrames = 0;
      this.currentFrameIndex = 0;
      
      try {
        // Upload video and get session ID
        const formData = new FormData();
        formData.append('file', this.selectedFile);
        
        const uploadResponse = await fetch('/upload', { method: 'POST', body: formData });
        
        if (!uploadResponse.ok) {
          const errorData = await uploadResponse.json().catch(() => ({ detail: 'Upload failed' }));
          throw new Error(errorData.detail || 'Upload failed');
        }
        
        const { session_id } = await uploadResponse.json();
        this.sessionId = session_id;
        console.log('Upload successful, session ID:', session_id);
        
        // Set up video for local playback
        this.videoUrl = URL.createObjectURL(this.selectedFile);
        this.$refs.videoPlayer.src = this.videoUrl;
        this.statusMessage = 'Starting inference...';
        
        // Connect WebSocket for real-time inference
        this.connectWebSocket(session_id);
        
      } catch (error) {
        console.error('Upload error:', error);
        alert(`Upload failed: ${error.message}`);
        this.processing = false;
        this.statusMessage = '';
      }
    },
    
    /**
     * Connect WebSocket for streaming inference results
     */
    connectWebSocket(sessionId) {
      const wsUrl = `ws://${location.host}/ws/${sessionId}`;
      console.log('Connecting to WebSocket:', wsUrl);
      
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.statusMessage = 'Processing frames...';
      };
      
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleInferenceResult(data);
        } catch (error) {
          console.error('WebSocket message error:', error);
        }
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.statusMessage = 'Connection error';
      };
      
      this.ws.onclose = () => {
        console.log('WebSocket closed');
        this.processing = false;
        this.statusMessage = '';
      };
    },
    
    /**
     * Handle inference result from WebSocket
     */
    handleInferenceResult(data) {
      console.log('📥 Received WebSocket data:', {
        frame_index: data.frame_index,
        abnormal_score: data.abnormal_score,
        topk_count: data.topk?.length,
        has_alert: !!data.alert,
        has_meta: !!data.meta
      });
      
      if (data.error) {
        console.error('❌ Inference error:', data.error);
        alert(`Error: ${data.error}`);
        return;
      }
      
      // Handle warmup status
      if (data.status && data.status.warming_up !== undefined) {
        const { warming_up, needed, eta_s } = data.status;
        this.statusMessage = `Loading frames: ${warming_up}/${needed} (ETA: ${eta_s}s)`;
        console.log('⏳ Warmup:', this.statusMessage);
        return;
      }

      // Handle status messages
      if (data.type === 'status' && data.message) {
        this.statusMessage = data.message;
        console.log('📢 Status update:', data.message);
        return;
      }

      // Handle final video summary messages (no frame_index)
      if (data.type === 'summary' && data.summary) {
        this.videoSummary = data.summary;
        this.statusMessage = 'Analysis complete - summary ready';
        console.log('📝 Video summary received:', this.videoSummary);
        return;
      }
      
      // Extract metadata - SET FPS FIRST before buffering data
      if (data.meta && data.meta.fps && !this.videoFps) {
        this.videoFps = data.meta.fps;
        console.log('🎬 Video FPS detected:', this.videoFps);
      }
      
      const frameIndex = data.frame_index;
      if (typeof frameIndex !== 'number') {
        console.warn('⚠️ Invalid frame index:', data);
        return;
      }
      
      // Store frame data in buffer (indexed by frame number)
      this.frameDataBuffer[frameIndex] = {
        abnormal_score: data.abnormal_score || 0,
        topk: data.topk || [],
        timestamp: frameIndex / this.videoFps,
      };
      
      console.log(`✅ Buffered frame ${frameIndex}: score=${data.abnormal_score?.toFixed(3)}, topk=${data.topk?.length} classes`);
      
      // Update statistics
      if (data.abnormal_score > this.peakAnomalyScore) {
        this.peakAnomalyScore = data.abnormal_score;
        console.log('📈 New peak score:', this.peakAnomalyScore.toFixed(3));
      }
      
      if (data.abnormal_score >= 0.5) {
        this.anomalyCount++;
        this.anomalyHistory.push({
          index: frameIndex,
          score: data.abnormal_score,
        });
        console.log(`🚨 Anomaly #${this.anomalyCount} at frame ${frameIndex}: ${data.abnormal_score.toFixed(3)}`);
        
        // Trigger push notification with top anomaly class
        if (data.topk && data.topk.length > 0) {
          // Find highest probability non-normal class
          const topAnomaly = data.topk.find(item => item.label !== 'Normal');
          if (topAnomaly) {
            this.showPushNotification(topAnomaly.label, data.abnormal_score, frameIndex);
          }
        }
      }
      
      // Store rich alert data with risk info
      if (data.abnormal_score >= 0.5 && data.topk && data.topk.length > 0) {
        const topAnomaly = data.topk.find(item => item.label !== 'Normal');
        if (topAnomaly) {
          const riskInfo = this.getAnomalyRiskInfo(topAnomaly.label, data.abnormal_score);
          this.alerts.push({
            type: topAnomaly.label,
            score: data.abnormal_score,
            frame: frameIndex,
            timestamp: (frameIndex / this.videoFps).toFixed(1),
            priority: riskInfo.priority,
            icon: riskInfo.icon,
            message: riskInfo.fullMessage,
            color: riskInfo.color
          });
        }
      }
      
      // Handle text alerts from Ollama (if any)
      if (data.alert) {
        console.log('🔔 Ollama alert:', data.alert);
      }
      
      // Chart.js removed - using Ollama summary instead
      
      // Track total frames processed
      if (frameIndex > this.totalFrames) {
        this.totalFrames = frameIndex;
        console.log(`📏 Total frames: ${this.totalFrames}`);
      }
    },
    
    /**
     * Video loaded - initialize canvas overlay
     */
    onVideoLoaded() {
      console.log('🎬 Video loaded');
      const video = this.$refs.videoPlayer;
      const canvas = this.$refs.overlayCanvas;
      
      // Sync canvas dimensions with video
      if (canvas && video.videoWidth && video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        console.log(`🎬 Canvas sized: ${video.videoWidth}x${video.videoHeight}`);
      }
      
      // Try to autoplay
      video.play().then(() => {
        this.isPlaying = true;
        console.log('▶️ Video autoplay started');
      }).catch((error) => {
        console.warn('⚠️ Autoplay prevented:', error);
      });
    },
    
    /**
     * Video timeupdate event - sync dashboard with current frame
     * CRITICAL: Frame synchronization logic using frame index, NOT wall-clock time
     */
    onTimeUpdate() {
      const video = this.$refs.videoPlayer;
      if (!video || !this.videoFps) return;
      
      // Calculate current frame index from video time and FPS
      const prevFrameIndex = this.currentFrameIndex;
      this.currentFrameIndex = Math.floor(video.currentTime * this.videoFps);
      
      // Only log on frame change to avoid spam
      if (this.currentFrameIndex !== prevFrameIndex) {
        console.log(`🎬 Frame ${this.currentFrameIndex}: time=${video.currentTime.toFixed(2)}s, fps=${this.videoFps}`);
      }
      
      // Lookup frame data from buffer
      const frameData = this.frameDataBuffer[this.currentFrameIndex];
      
      if (frameData) {
        this.currentFrameData = frameData;
        
        if (this.currentFrameIndex !== prevFrameIndex) {
          console.log(`✅ Found data for frame ${this.currentFrameIndex}:`, {
            score: frameData.abnormal_score.toFixed(3),
            topk: frameData.topk.length
          });
        }
        
        // Update canvas overlay
        this.updateCanvasOverlay(frameData);
      } else {
        if (this.currentFrameIndex !== prevFrameIndex) {
          console.warn(`⚠️ No data for frame ${this.currentFrameIndex} (buffer has ${Object.keys(this.frameDataBuffer).length} frames)`);
        }
        
        // Frame not yet processed - clear overlay
        const canvas = this.$refs.overlayCanvas;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
    },
    
    /**
     * Draw canvas overlay for current frame
     * Red border for anomalies, transparent otherwise
     */
    updateCanvasOverlay(frameData) {
      const canvas = this.$refs.overlayCanvas;
      if (!canvas) return;
      
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const isAnomaly = frameData.abnormal_score >= 0.5;
      
      if (isAnomaly) {
        // Draw red border
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 8;
        ctx.strokeRect(4, 4, canvas.width - 8, canvas.height - 8);
        
        // Draw anomaly indicator
        ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
        ctx.fillRect(20, 20, 200, 40);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 20px sans-serif';
        ctx.fillText('ANOMALY DETECTED', 30, 46);
      }
    },
    
    /**
     * Toggle play/pause
     */
    togglePlayPause() {
      const video = this.$refs.videoPlayer;
      if (!video) return;
      
      if (video.paused) {
        video.play();
        this.isPlaying = true;
      } else {
        video.pause();
        this.isPlaying = false;
      }
    },
    
    /**
     * Change playback speed
     */
    changeSpeed(speed) {
      const video = this.$refs.videoPlayer;
      if (!video) return;
      
      video.playbackRate = parseFloat(speed) || 1;
      console.log(`Playback speed: ${speed}x`);
    },
  };
}
