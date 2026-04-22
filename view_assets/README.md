> **Navigation:** [← Main README](../README.md)

# view_assets/

Streamlit app for browsing and viewing generated media assets (images, audio, videos) from pipeline S3 output buckets.

## Running

```bash
streamlit run view_assets/app.py
```

Requires `streamlit` (included in dev dependencies) and valid AWS credentials with S3 read access.

## What It Displays

The app reads pipeline config YAMLs to discover `construct_id` values, then lists matching S3 output buckets (`{account}-{region}-{prefix}{construct_id}-*-output-bucket`). The sidebar provides filters for:

- **Construct ID** — select which pipeline's outputs to browse
- **Step** — select which processing step's output bucket to view
- **Execution ID** — filter assets by SageMaker Pipeline execution
- **Media type** — filter by video, image, audio, or all

Assets are displayed inline with prev/next navigation. Supported formats:

- Video: `.mp4`, `.webm`, `.mkv`
- Image: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp`
- Audio: `.mp3`, `.wav`, `.ogg`, `.flac`, `.aac`

## Configuration

The app reads `AWS_ACCOUNT_ID` and `REGION` from `.env` (via `python-dotenv`) and `shared_prefix` from `config/cicd/cicd.yaml`.
