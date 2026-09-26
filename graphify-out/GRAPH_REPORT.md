# Graph Report - video enchancer  (2026-09-26)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 110 nodes · 199 edges · 13 communities (8 shown, 5 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.85)
- Token cost: 19,166 input · 1,039 output

## Graph Freshness
- Built from commit: `c0104d63`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Video Enhancement CLI Pipeline
- Model Loading and Management
- FastAPI Web Application
- API and Pipeline Tests
- Image Tile Processor
- RRDB Network Architecture
- Video Upload and Tasks
- Web API Endpoints
- System and GPU Information
- WebSocket Progress Endpoint
- AI Video Enhancer Core

## God Nodes (most connected - your core abstractions)
1. `TileProcessor` - 12 edges
2. `FFmpegReader` - 11 edges
3. `FFmpegWriter` - 11 edges
4. `get_realesrgan_model()` - 10 edges
5. `RRDBNet` - 9 edges
6. `main()` - 7 edges
7. `process_video_task()` - 7 edges
8. `ensure_models_dir()` - 5 edges
9. `ResidualDenseBlock_5C` - 4 edges
10. `load_realesrgan_weights()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `process_video_task()` --calls--> `FFmpegReader`  [EXTRACTED]
  app.py → utils/ffmpeg_handler.py
- `process_video_task()` --calls--> `FFmpegWriter`  [EXTRACTED]
  app.py → utils/ffmpeg_handler.py
- `test_rrdbnet_model()` --calls--> `RRDBNet`  [EXTRACTED]
  tests/test_model_loader.py → utils/rrdbnet.py
- `process_video_task()` --calls--> `TileProcessor`  [EXTRACTED]
  app.py → utils/tile_processor.py
- `main()` --calls--> `TileProcessor`  [EXTRACTED]
  enhance.py → utils/tile_processor.py

## Import Cycles
- None detected.

## Communities (13 total, 5 thin omitted)

### Community 0 - "Video Enhancement CLI Pipeline"
Cohesion: 0.14
Nodes (13): argparse, cv2, enhance_frame(), main(), parse_args(), AIVideoEnhancer - AI-powered video enhancement tool Supports real-time video…, Enhance a single frame using the model., numpy (+5 more)

### Community 1 - "Model Loading and Management"
Cohesion: 0.15
Nodes (18): math, requests, sys, test_ensure_models_dir(), test_rrdbnet_model(), test_tile_processor(), torch, torch_nn (+10 more)

### Community 2 - "FastAPI Web Application"
Cohesion: 0.14
Nodes (13): asyncio, fastapi, fastapi_middleware_cors, fastapi_responses, fastapi_staticfiles, json, pathlib, shutil (+5 more)

### Community 3 - "API and Pipeline Tests"
Cohesion: 0.20
Nodes (4): fastapi_testclient, os, pytest, tempfile

### Community 4 - "Image Tile Processor"
Cohesion: 0.39
Nodes (5): Module, ndarray, Process an image tile by tile. Args: img: Input image as numpy array (H, W, C),…, Handle tiling for VRAM-efficient processing of large images., TileProcessor

### Community 5 - "RRDB Network Architecture"
Cohesion: 0.29
Nodes (3): make_layer(), ResidualDenseBlock_5C, RRDB

### Community 6 - "Video Upload and Tasks"
Cohesion: 0.29
Nodes (7): notify_task_update(), process_video_task(), start_enhancement(), upload_video(), BackgroundTasks, post, UploadFile

### Community 7 - "Web API Endpoints"
Cohesion: 0.33
Nodes (6): get_output_video(), get_task_status(), get_uploaded_video(), list_models(), serve_ui(), get

## Knowledge Gaps
- **1 isolated node(s):** `AIVideoEnhancer`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 46 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TileProcessor` connect `Image Tile Processor` to `Video Enhancement CLI Pipeline`, `Model Loading and Management`, `FastAPI Web Application`, `Video Upload and Tasks`?**
  _High betweenness centrality (0.154) - this node is a cross-community bridge._
- **Why does `FFmpegWriter` connect `Video Enhancement CLI Pipeline` to `FastAPI Web Application`, `Video Upload and Tasks`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `get_realesrgan_model()` connect `Model Loading and Management` to `Video Enhancement CLI Pipeline`, `FastAPI Web Application`, `Video Upload and Tasks`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **What connects `AIVideoEnhancer` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Video Enhancement CLI Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.13768115942028986 - nodes in this community are weakly interconnected._
- **Should `FastAPI Web Application` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._