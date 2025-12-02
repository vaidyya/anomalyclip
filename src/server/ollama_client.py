import asyncio
import base64
import logging
import time
from typing import Optional, Dict, Any, List, Tuple

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
# Using Moondream - lightweight VLM (1.7GB, efficient vision model)
# Switched from MiniCPM-V (7.6B) which was too heavy and crashed
MODEL_NAME = "moondream:latest"  # 1.7GB, vision-language model


def _make_ollama_request_with_retry(url: str, payload: Dict[str, Any], max_retries: int = 3,
                                   timeout: float = 120.0) -> Optional[Dict[str, Any]]:
    """Make HTTP request to Ollama with retry logic and exponential backoff."""
    for attempt in range(max_retries):
        try:
            logger.debug(f"Ollama request attempt {attempt + 1}/{max_retries}")
            response = httpx.post(url, json=payload, timeout=timeout)

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Ollama request failed with status {response.status_code}: {response.text[:200]}")

        except httpx.TimeoutException:
            logger.warning(f"Ollama request timed out (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            logger.warning(f"Ollama request failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")

        # Exponential backoff: 1s, 2s, 4s
        if attempt < max_retries - 1:
            sleep_time = 2 ** attempt
            logger.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

    return None


def _generate_fallback_description(segment_data: List[Dict[str, Any]], anomaly_type: str,
                                  max_score: float, avg_score: float) -> str:
    """Generate meaningful fallback description when VLM fails."""
    if not segment_data:
        return "No frame data available for analysis"

    # Get severity level
    if max_score >= 0.7:
        severity = "high"
    elif max_score >= 0.5:
        severity = "medium"
    else:
        severity = "low"

    # Build description based on available data
    parts = []

    if anomaly_type and anomaly_type != "Unknown":
        parts.append(f"Detected {anomaly_type.lower()}")

    parts.append(f"with {severity} confidence")

    # Add score information
    parts.append(f"(score: {max_score:.2f})")

    # Add frame count
    frame_count = len(segment_data)
    if frame_count > 1:
        parts.append(f"across {frame_count} frames")

    return " ".join(parts).capitalize()


def _group_anomalies_into_segments(anomaly_events: List[Dict[str, Any]], fps: float) -> List[Dict[str, Any]]:
    """Group anomaly events into temporal segments for summarization."""
    if not anomaly_events:
        return []

    # Sort by timestamp
    sorted_events = sorted(anomaly_events, key=lambda x: x['timestamp'])

    segments = []
    current_segment = {
        'start_time': sorted_events[0]['timestamp'],
        'end_time': sorted_events[0]['timestamp'],
        'events': [sorted_events[0]],
        'max_score': sorted_events[0]['score'],
        'dominant_type': sorted_events[0]['type']
    }

    # Group events within 5 seconds of each other
    time_threshold = 5.0

    for event in sorted_events[1:]:
        time_gap = event['timestamp'] - current_segment['end_time']

        if time_gap <= time_threshold:
            # Add to current segment
            current_segment['end_time'] = event['timestamp']
            current_segment['events'].append(event)
            current_segment['max_score'] = max(current_segment['max_score'], event['score'])
        else:
            # Start new segment
            segments.append(current_segment)
            current_segment = {
                'start_time': event['timestamp'],
                'end_time': event['timestamp'],
                'events': [event],
                'max_score': event['score'],
                'dominant_type': event['type']
            }

    segments.append(current_segment)

    # Update dominant types for each segment
    for segment in segments:
        type_counts = {}
        for event in segment['events']:
            event_type = event['type']
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        segment['dominant_type'] = max(type_counts.keys(), key=lambda x: type_counts[x]) if type_counts else "Unknown"

    return segments


def _generate_segment_summary_from_stats(segment: Dict[str, Any], fps: float) -> Dict[str, Any]:
    """Generate segment summary from statistical anomaly data."""
    start_time = segment['start_time']
    end_time = segment['end_time']
    events = segment['events']
    max_score = segment['max_score']
    dominant_type = segment['dominant_type']

    # Calculate severity
    if max_score >= 0.7:
        severity = "high"
    elif max_score >= 0.6:
        severity = "medium"
    else:
        severity = "low"

    # Generate descriptive summary
    event_count = len(events)
    duration = end_time - start_time

    if dominant_type != "Unknown":
        summary = f"Detected {event_count} {dominant_type.lower()} event{'s' if event_count > 1 else ''} "
        summary += f"with {severity} confidence over {duration:.1f} seconds."
    else:
        summary = f"Detected {event_count} anomalous event{'s' if event_count > 1 else ''} "
        summary += f"with {severity} confidence over {duration:.1f} seconds."

    # Get frame indices for the segment
    frame_indices = [event['frame_index'] for event in events]

    return {
        "timerange": f"{format_timestamp(start_time)}-{format_timestamp(end_time)}",
        "frames": frame_indices,
        "summary": summary,
        "max_anomaly": max_score,
        "anomaly_type": dominant_type,
        "visual_description": summary,  # Use same summary for visual description
        "event_count": event_count,
        "duration": duration,
        "severity": severity
    }


def _generate_executive_fallback(segment_summaries: List[Dict[str, Any]], anomaly_rate: float,
                                max_anomaly_frame: Dict[str, Any], video_metadata: Dict[str, Any]) -> str:
    """Generate executive summary fallback when LLM fails."""
    if not segment_summaries:
        return "Video analysis completed but summary generation failed. No anomaly segments detected."

    duration = video_metadata.get('duration', 0)

    # Get dominant anomaly type
    anomaly_types = [seg.get('anomaly_type', 'Unknown') for seg in segment_summaries if seg.get('anomaly_type', 'Unknown') != 'Unknown']
    dominant_type = max(set(anomaly_types), key=anomaly_types.count) if anomaly_types else "Unknown"

    # Get severity from max score
    max_score = max_anomaly_frame.get('abnormal_score', 0)
    if max_score >= 0.7:
        severity = "HIGH"
    elif max_score >= 0.5:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    # Build summary
    summary_parts = [
        f"SECURITY ALERT: {dominant_type} anomaly detected in {duration:.1f}s video.",
        f"Severity: {severity} (peak score: {max_score:.2f}).",
        f"Anomaly rate: {anomaly_rate:.1f}%.",
    ]

    # Add segment information
    segment_info = []
    for i, seg in enumerate(segment_summaries[:3]):  # Limit to first 3 segments
        seg_type = seg.get('anomaly_type', 'Unknown')
        seg_score = seg.get('max_anomaly', 0)
        segment_info.append(f"Segment {i+1}: {seg_type} ({seg_score:.2f})")

    if segment_info:
        summary_parts.append("Details: " + "; ".join(segment_info))

    summary_parts.append("RECOMMENDATION: Review video footage for verification.")

    return " ".join(summary_parts)


def summarize_event(
    summary_text: str,
    ref_image_b64: Optional[str] = None,
    timeout: float = 10.0
) -> str:
    """Call local Ollama LLM to generate a concise anomaly alert summary.

    Args:
        summary_text: Textual context (stats, top classes, timestamps)
        ref_image_b64: Optional PNG/JPEG base64 of a representative frame
        timeout: Request timeout in seconds
        
    Returns:
        Short summary string describing the anomaly, or fallback message if unavailable
    """
    request_payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "prompt": (
            "You are an on-device video anomaly assistant. "
            "Given the event context, produce a one-sentence alert with the likely anomaly type, severity (low/med/high), "
            "and a brief justification using the evidence provided. Keep under 30 words.\n\n"
            f"Context:\n{summary_text}\n"
        ),
        "stream": False,
    }

    if ref_image_b64:
        # Ollama multimodal API accepts base64 images in the 'images' array
        request_payload["images"] = [ref_image_b64]
        logger.info("Requesting summary with image from Ollama")
    else:
        logger.info("Requesting summary from Ollama (text only)")

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{OLLAMA_URL}/api/generate", json=request_payload)
            response.raise_for_status()
            response_data = response.json()
            
            # Responses commonly include 'response' or 'message'. Prefer 'response'.
            summary = response_data.get("response") or response_data.get("message", "")
            
            if summary:
                logger.info(f"Ollama summary generated: {summary[:50]}...")
                return summary
            else:
                logger.warning("Ollama returned empty response")
                return "(No summary generated)"
                
    except httpx.TimeoutException:
        logger.warning(f"Ollama request timed out after {timeout}s")
        return "(Summarizer timed out)"
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama HTTP error: {e.response.status_code} - {e.response.text}")
        return "(Summarizer error)"
    except httpx.ConnectError:
        logger.error(f"Cannot connect to Ollama at {OLLAMA_URL}. Is Ollama running?")
        return "(Ollama not available)"
    except Exception as e:
        logger.error(f"Unexpected error calling Ollama: {type(e).__name__}: {e}")
        return "(Summarizer unavailable)"


def select_keyframes(frame_results: List[Dict[str, Any]], max_keyframes: int = 10) -> List[int]:
    """Select representative keyframes based on anomaly events.
    
    Strategy:
    1. All high-confidence anomalies (score > 0.7)
    2. Segment video into time windows, pick highest anomaly per window
    3. Normal baseline frames (lowest scores) for context
    
    Args:
        frame_results: List of inference results with 'frame_index' and 'abnormal_score'
        max_keyframes: Maximum keyframes to select
        
    Returns:
        List of frame indices to summarize
    """
    if not frame_results:
        return []
    
    keyframes: List[int] = []
    
    # Sort by anomaly score (highest first)
    sorted_frames = sorted(frame_results, key=lambda x: x.get('abnormal_score', 0), reverse=True)
    
    # Step 1: High anomalies (top 40% of max_keyframes, score > 0.7)
    high_anomaly_count = max(1, int(max_keyframes * 0.4))
    high_anomalies = [f['frame_index'] for f in sorted_frames if f.get('abnormal_score', 0) > 0.7]
    keyframes.extend(high_anomalies[:high_anomaly_count])
    
    # Step 2: Temporal segmentation - pick highest per segment
    total_frames = len(frame_results)
    remaining_slots = max_keyframes - len(keyframes)
    
    if remaining_slots > 0:
        segment_count = remaining_slots
        segment_size = total_frames // segment_count if segment_count > 0 else total_frames
        
        for i in range(segment_count):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, total_frames)
            segment = frame_results[start_idx:end_idx]
            
            if segment:
                best_frame = max(segment, key=lambda x: x.get('abnormal_score', 0))
                frame_idx = best_frame['frame_index']
                if frame_idx not in keyframes:
                    keyframes.append(frame_idx)
    
    # Step 3: Add normal baseline frames if still have room
    if len(keyframes) < max_keyframes:
        normal_frames = [f['frame_index'] for f in sorted_frames[-3:] if f['frame_index'] not in keyframes]
        keyframes.extend(normal_frames[:max_keyframes - len(keyframes)])
    
    return sorted(keyframes)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


async def generate_video_summary(
    frame_results: List[Dict[str, Any]],
    video_metadata: Dict[str, Any],
    frame_images: Dict[int, str] = None,  # Dict of frame_idx -> base64 image (deprecated - not used)
    ollama_model: str = "phi:latest",  # Use LLM for text synthesis only
    max_keyframes: int = 5  # Reduced from 10 to limit compute
) -> Dict[str, Any]:
    """Generate comprehensive video summary using statistical analysis of anomaly detection results.

    ROOT CAUSE FIX: Eliminates slow VLM image analysis by generating summaries directly
    from the rich anomaly detection data already produced by AnomalyCLIP model.

    This approach is fundamentally faster because:
    1. No redundant image re-analysis (AnomalyCLIP already analyzed the content)
    2. No base64 encoding/decoding overhead
    3. No network latency to VLM service
    4. Direct synthesis from model outputs
    
    Args:
        frame_results: List of all frame inference results with anomaly scores and classifications
        video_metadata: Video info (duration, fps, etc.)
        frame_images: Not used in this optimized version
        ollama_model: LLM model for final text synthesis (not VLM)
        max_keyframes: Maximum keyframes to analyze (for backward compatibility)
        
    Returns:
        Dict with overall_summary, anomaly_segments, statistics, keyframes
    """
    logger.info(f"Generating video summary for {len(frame_results)} frames using statistical analysis")
    
    if not frame_results:
        return {
            "overall_summary": "No frames to analyze",
            "anomaly_segments": [],
            "statistics": {},
            "keyframes": []
        }
    
    # Extract video metadata
    video_fps = video_metadata.get('fps', 25.0)
    video_duration = len(frame_results) / video_fps

    # Phase 1: Statistical Analysis of Anomaly Detection Results
    anomaly_events = []
    anomaly_type_counts = {}
    severity_distribution = {"low": 0, "medium": 0, "high": 0}

    for frame_data in frame_results:
        abnormal_score = frame_data.get('abnormal_score', 0)

        # Only consider significant anomalies
        if abnormal_score >= 0.5:
            # Determine severity
            if abnormal_score >= 0.7:
                severity = "high"
            elif abnormal_score >= 0.6:
                severity = "medium"
            else:
                severity = "low"

            severity_distribution[severity] += 1

            # Extract anomaly type from model predictions
            anomaly_type = "Unknown"
            if frame_data.get('topk') and len(frame_data['topk']) > 0:
                top_prediction = frame_data['topk'][0]
                if top_prediction.get('label') and top_prediction['label'] != 'Normal':
                    anomaly_type = top_prediction['label']
                    anomaly_type_counts[anomaly_type] = anomaly_type_counts.get(anomaly_type, 0) + 1

            timestamp = frame_data.get('frame_index', 0) / video_fps

            anomaly_events.append({
                'timestamp': timestamp,
                'frame_index': frame_data.get('frame_index', 0),
                'score': abnormal_score,
                'severity': severity,
                'type': anomaly_type,
                'topk': frame_data.get('topk', [])
            })

    # Phase 2: Generate Segment Summaries from Statistical Data
    segment_summaries = []

    if anomaly_events:
        # Group anomalies into temporal segments
        segments = _group_anomalies_into_segments(anomaly_events, video_fps)

        for segment in segments[:2]:  # Limit to 2 segments for performance
            segment_summary = _generate_segment_summary_from_stats(segment, video_fps)
            segment_summaries.append(segment_summary)
    else:
        # No significant anomalies detected
        segment_summaries.append({
            "timerange": f"{format_timestamp(0)}-{format_timestamp(video_duration)}",
            "frames": [0, len(frame_results)-1],
            "summary": "No significant anomalies detected in the video footage.",
            "max_anomaly": 0.0,
            "anomaly_type": "Normal",
            "visual_description": "Normal surveillance footage with no detected security incidents."
        })

    # Phase 3: Compute Overall Statistics
    total_anomalies = len(anomaly_events)
    anomaly_rate = (total_anomalies / len(frame_results)) * 100 if frame_results else 0

    max_anomaly_frame = max(frame_results, key=lambda x: x.get('abnormal_score', 0))
    dominant_anomaly_type = max(anomaly_type_counts.keys(), key=lambda x: anomaly_type_counts[x]) if anomaly_type_counts else "None"
    
    # Phase 4: Generate Executive Summary using LLM
    duration = video_duration

    # Build context from statistical segment summaries
    aggregated_context = "\n".join([
        f"Segment {i+1} ({seg['timerange']}): {seg.get('summary', 'Unknown activity')}"
        for i, seg in enumerate(segment_summaries)
    ])

    # Security-focused prompt for executive summary
    final_prompt = f"""SECURITY INCIDENT SUMMARY

ANALYSIS RESULTS:
- Total Duration: {duration:.1f} seconds
- Anomaly Rate: {anomaly_rate:.1f}% ({total_anomalies}/{len(frame_results)} frames)
- Peak Anomaly Score: {max_anomaly_frame.get('abnormal_score', 0):.2f}
- Dominant Anomaly Type: {dominant_anomaly_type}

INCIDENT DETAILS:
{aggregated_context}

Generate a concise 2-3 sentence security summary for operators:
1. Describe the incident type and severity
2. Provide key metrics and timing
3. Recommend immediate action

Use professional security terminology."""

    overall_summary = "Summary generation not available"
    try:
        logger.info(f"Generating executive summary using statistical analysis")

        # Use Phi LLM for final text synthesis
        phi_payload = {
            "model": ollama_model,  # Now using phi:latest
            "prompt": final_prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,     # Low temp for consistent summaries
                "num_predict": 150,     # Allow comprehensive summary
                "num_ctx": 1024,        # Sufficient context
                "num_thread": 1,        # Single thread for stability
            }
        }

        response_data = _make_ollama_request_with_retry(
            f"{OLLAMA_URL}/api/generate",
            phi_payload,
            max_retries=2,  # Allow retries for LLM
            timeout=30.0    # Faster timeout since no images
        )

        if response_data:
            overall_summary = response_data.get("response", "Summary generation failed").strip()
            if overall_summary:
                logger.info(f"Executive summary generated ({len(overall_summary)} chars): {overall_summary[:100]}...")
            else:
                logger.warning("Executive summary empty, using fallback")
                overall_summary = _generate_executive_fallback(segment_summaries, anomaly_rate, max_anomaly_frame, video_metadata)
        else:
            logger.error("Executive summary generation failed after retries")
            overall_summary = _generate_executive_fallback(segment_summaries, anomaly_rate, max_anomaly_frame, video_metadata)
    except Exception as e:
        logger.error(f"Failed to generate executive summary: {type(e).__name__}: {e}")
        overall_summary = _generate_executive_fallback(segment_summaries, anomaly_rate, max_anomaly_frame, video_metadata)

    return {
        "overall_summary": overall_summary,
        "anomaly_segments": segment_summaries,
        "statistics": {
            "total_frames": len(frame_results),
            "total_anomalies": total_anomalies,
            "anomaly_rate_percent": round(anomaly_rate, 2),
            "peak_anomaly": {
                "score": max_anomaly_frame.get('abnormal_score', 0),
                "frame": max_anomaly_frame.get('frame_index', 0),
                "timestamp": format_timestamp(max_anomaly_frame.get('frame_index', 0) / video_fps)
            },
            "anomaly_type_distribution": anomaly_type_counts,
            "severity_distribution": severity_distribution
        },
        "keyframes": []  # No longer using keyframes in this optimized version
    }
