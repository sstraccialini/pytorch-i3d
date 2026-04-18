import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Default Paths (can be overridden by CLI args)
    "features_dir": "features/",           
    "output_dir": "labels/",               
    "class_map_file": "class_mapping.json",
    
    # CSV Column Names
    "col_video_id": "video_id",
    "col_label": "event",
    "col_start": "start_frame",
    "col_end": "end_frame",
    
    # Task specific constants
    "annotation_fps": 30.0,                # Original FPS of the video (used to convert frames to seconds)
    "feature_fps": 10.0,                   # FPS the features were extracted at
    "frames_per_segment": 16,
    "max_segments": 2500,
    "num_classes": 51,
    
    # Optional settings
    "debug_video_id": None,                # Set to a video_id string for verbose debug output
}

# Calculated constants
SECONDS_PER_SEGMENT = CONFIG["frames_per_segment"] / CONFIG["feature_fps"]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def determine_valid_segments(features: np.ndarray) -> int:
    """
    Infers the number of valid segments by finding the last non-zero row.
    Assumes features is padded with zeros.
    """
    if features.shape[0] < CONFIG["max_segments"]:
        return features.shape[0]
    
    # Find rows that are not entirely zero
    non_zero_rows = np.any(features != 0, axis=1)
    if not np.any(non_zero_rows):
        return 0
        
    last_non_zero_idx = np.where(non_zero_rows)[0][-1]
    return last_non_zero_idx + 1

def time_to_segments(start_time: float, end_time: float, valid_segments: int) -> Tuple[int, int]:
    """
    Converts a time interval to segment indices.
    A segment i covers [i * SECONDS_PER_SEGMENT, (i + 1) * SECONDS_PER_SEGMENT)
    """
    start_idx = int(start_time // SECONDS_PER_SEGMENT)
    # Use ceil equivalent logic for the end index to capture any overlap
    end_idx = int(np.ceil(end_time / SECONDS_PER_SEGMENT))
    
    # Clip to valid range
    start_idx = max(0, min(start_idx, valid_segments - 1))
    end_idx = max(0, min(end_idx, valid_segments))
    
    return start_idx, end_idx

# =============================================================================
# MAIN LOGIC
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare TSU Action Detection Labels from CSVs.")
    parser.add_argument("--features_dir", type=str, required=True, help="Path to the directory containing .npy feature files")
    parser.add_argument("--annotations_dir", type=str, required=True, help="Base path to TSU nested annotation CSVs (e.g. datasets/Annotation/)")
    parser.add_argument("--output_dir", type=str, default="labels/", help="Path to save the generated label matrices")
    args = parser.parse_args()
    
    CONFIG["features_dir"] = args.features_dir
    CONFIG["output_dir"] = args.output_dir

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # 1. Read nested CSV annotations
    print(f"Scanning for annotations in {args.annotations_dir}...")
    csv_paths = glob.glob(os.path.join(args.annotations_dir, "**", "*.csv"), recursive=True)
    
    if not csv_paths:
        raise ValueError(f"No CSV files found in {args.annotations_dir}")
        
    dfs = []
    print(f"Reading {len(csv_paths)} annotation files...")
    for csv_path in csv_paths:
        try:
            # Note: Update header logic here if CSVs lack headers. TSU sometimes lacks them.
            df_part = pd.read_csv(csv_path)
            
            # Since CSV is per video (e.g., P02T01C06.csv), inject video_id if missing
            video_name = os.path.splitext(os.path.basename(csv_path))[0]
            if CONFIG["col_video_id"] not in df_part.columns:
                df_part[CONFIG["col_video_id"]] = video_name
                
            dfs.append(df_part)
        except Exception as e:
            print(f"Warning: Failed to read {csv_path}: {e}")
            
    if not dfs:
        raise ValueError("No valid annotation data could be read.")
        
    df = pd.concat(dfs, ignore_index=True)
    
    # 2. Build class mapping mapping dynamically if needed, 
    # but TSU has 51 classes. We just extract unique lexicographically.
    unique_classes = sorted(df[CONFIG["col_label"]].dropna().unique().tolist())
    if len(unique_classes) > CONFIG["num_classes"]:
        print(f"Warning: Found {len(unique_classes)} classes, expected {CONFIG['num_classes']}")
        
    class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
    
    # Save class mapping
    class_map_path = os.path.join(CONFIG["output_dir"], CONFIG["class_map_file"])
    with open(class_map_path, 'w') as f:
        json.dump(class_to_idx, f, indent=4)
        
    # Group annotations by video
    video_groups = df.groupby(CONFIG["col_video_id"])
    
    # 3. Process Feature Files
    feature_files = glob.glob(os.path.join(CONFIG["features_dir"], "*.[nn][pp][yz]")) # .npy or .npz
    
    stats = {
        "processed": 0,
        "missing_features": 0,
        "unknown_labels": 0,
        "total_positive_segments": 0
    }
    
    print(f"Found {len(feature_files)} feature files. Processing...")
    
    processed_videos = set()
    
    for feat_path in feature_files:
        video_id = os.path.splitext(os.path.basename(feat_path))[0]
        
        # Load features
        try:
            if feat_path.endswith('.npz'):
                data = np.load(feat_path)
                features = data['arr_0'] if 'arr_0' in data else data[list(data.keys())[0]]
            else:
                features = np.load(feat_path)
        except Exception as e:
            print(f"Error loading {feat_path}: {e}")
            continue
            
        if features.shape[1] != 1024:
            print(f"Warning: {video_id} features have shape {features.shape}. Expected (..., 1024).")
            
        valid_segments = determine_valid_segments(features)
        
        # Initialize label matrix
        labels = np.zeros((CONFIG["max_segments"], CONFIG["num_classes"]), dtype=np.float32)
        
        # Populate labels if annotations exist
        if video_id in video_groups.groups:
            video_annos = video_groups.get_group(video_id)
            
            for _, row in video_annos.iterrows():
                action = row[CONFIG["col_label"]]
                
                # Convert frames to seconds
                start_t = row[CONFIG["col_start"]] / CONFIG["annotation_fps"]
                end_t = row[CONFIG["col_end"]] / CONFIG["annotation_fps"]
                
                if end_t <= start_t:
                    print(f"Warning: Invalid time range in {video_id}: {start_t} to {end_t}")
                    continue
                    
                if action not in class_to_idx:
                    stats["unknown_labels"] += 1
                    continue
                    
                c_idx = class_to_idx[action]
                s_idx, e_idx = time_to_segments(start_t, end_t, valid_segments)
                
                if s_idx < e_idx:
                    labels[s_idx:e_idx, c_idx] = 1.0
                    
                if CONFIG["debug_video_id"] == video_id:
                    print(f"DEBUG {video_id}: {action} [{start_t:.2f}s-{end_t:.2f}s] -> segments [{s_idx}:{e_idx})")
        else:
            # Features exist, but no annotations. Handled implicitly by zeros.
            pass
            
        # Save labels
        out_path = os.path.join(CONFIG["output_dir"], f"{video_id}.npy")
        np.save(out_path, labels)
        
        stats["processed"] += 1
        stats["total_positive_segments"] += np.sum(labels)
        processed_videos.add(video_id)
        
    # Check for videos in CSV that had no features
    csv_video_ids = set(video_groups.groups.keys())
    missing_features = csv_video_ids - processed_videos
    stats["missing_features"] = len(missing_features)
    
    # 4. Summary Output
    avg_pos = stats["total_positive_segments"] / max(1, stats["processed"])
    print("\n" + "="*40)
    print("PROCESSING SUMMARY")
    print("="*40)
    print(f"Videos processed: {stats['processed']}")
    print(f"Videos in CSV missing features: {stats['missing_features']}")
    print(f"Annotations with unknown labels: {stats['unknown_labels']}")
    print(f"Avg positive segments/video: {avg_pos:.2f}")
    if missing_features:
        print(f"Sample missing feature videos: {list(missing_features)[:5]}")

if __name__ == "__main__":
    main()