"""Shared utilities for processing job steps."""

import os

from loguru import logger


def scan_output_files(output_dir: str, extensions: tuple[str, ...] = None) -> list[dict]:
    """Scan an output directory and return metadata for each file.

    Args:
        output_dir: Path to the output directory to scan.
        extensions: Optional tuple of file extensions to filter (e.g. ('.mp4', '.png')).
                    If None, returns all files.

    Returns:
        List of dicts with keys: filename, path, size_bytes, extension.
    """
    results = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            filepath = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if extensions and ext not in extensions:
                continue
            results.append(
                {
                    "filename": f,
                    "path": filepath,
                    "size_bytes": os.path.getsize(filepath),
                    "extension": ext,
                }
            )
    logger.info(f"Scanned {len(results)} files from {output_dir}")
    return results


VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".avi", ".mov")


def get_video_metadata(video_path: str) -> dict[str, int | float]:
    """Extract metadata from a video file using moviepy.

    Returns width, height, fps, duration, and frame_count.
    On failure, returns -1 for all fields.
    """
    from moviepy import VideoFileClip

    try:
        clip = VideoFileClip(video_path)
        metadata = {
            "width": clip.w,
            "height": clip.h,
            "fps": clip.fps,
            "duration": clip.duration,
            "frame_count": int(clip.fps * clip.duration),
        }
        clip.close()
        logger.info(f"Video metadata for {video_path}: {metadata}")
        return metadata
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return {
            "width": -1,
            "height": -1,
            "fps": -1,
            "duration": -1,
            "frame_count": -1,
        }
