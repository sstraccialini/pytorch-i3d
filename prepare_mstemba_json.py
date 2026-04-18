import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # CSV Column Names
    "col_video_id": "video_id",
    "col_label": "event",
    "col_start": "start_frame",
    "col_end": "end_frame",
    
    # Task specific constants
    "annotation_fps": 30.0,                # Original FPS of the video 
    "feature_fps": 10.0,                   # FPS the features were extracted at
    "frames_per_segment": 16,
    "max_segments": 2500,
}

SECONDS_PER_SEGMENT = CONFIG["frames_per_segment"] / CONFIG["feature_fps"]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def determine_valid_segments(features: np.ndarray) -> int:
    """
    Infers the number of valid segments by finding the last non-zero row.
    Assumes features is padded with zeros.
    """
    if len(features.shape) == 1:
        return 0
        
    num_rows = features.shape[0]
    if num_rows < CONFIG["max_segments"]:
        # If it's shorter than max_segments, all rows are likely valid
        # unless it was padded to a smaller multiple. 
        # We still check for trailing zeros to be safe.
        pass
    
    # Find rows that are not entirely zero
    non_zero_rows = np.any(features != 0, axis=1)
    if not np.any(non_zero_rows):
        return 0
        
    last_non_zero_idx = np.where(non_zero_rows)[0][-1]
    return int(last_non_zero_idx + 1)

def load_features(feat_path: str) -> np.ndarray:
    """
    Robustly load features from .npy or .npz
    """
    if feat_path.endswith('.npz'):
        data = np.load(feat_path)
        # Try standard keys, fallback to first available
        if 'raw_features' in data:
            return data['raw_features']
        elif 'features' in data:
            return data['features']
        else:
            return data[list(data.keys())[0]]
    else:
        return np.load(feat_path)

# =============================================================================
# MAIN LOGIC
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert TSU CSVs to MS-Temba JSON format.")
    parser.add_argument("--features_dir", type=str, required=True, help="Directory containing .npy/.npz feature files")
    parser.add_argument("--annotations_dir", type=str, required=True, help="Directory containing nested annotation CSVs")
    parser.add_argument("--class_mapping", type=str, required=True, help="Path to class_mapping.json")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON path (e.g., MS-Temba/data/smarthome.json)")
    parser.add_argument("--split_file", type=str, default=None, help="Optional CSV mapping video_id to subset (training/testing)")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    
    # 1. Load class mapping (must exist and be fixed)
    print(f"Loading class mapping from {args.class_mapping}...")
    with open(args.class_mapping, 'r') as f:
        class_map = json.load(f)
        
    # 2. Discover features
    print(f"Scanning features in {args.features_dir}...")
    feat_paths = glob.glob(os.path.join(args.features_dir, "*.[nn][pp][yz]"))
    feat_dict = {Path(p).stem: p for p in feat_paths}
    print(f"Found {len(feat_dict)} feature files.")
    
    # 3. Load splits if provided
    split_dict = {}
    if args.split_file and os.path.exists(args.split_file):
        print(f"Loading splits from {args.split_file}...")
        split_df = pd.read_csv(args.split_file)
        # Assumes columns 'video_id' and 'subset'
        if 'video_id' in split_df.columns and 'subset' in split_df.columns:
            split_dict = dict(zip(split_df['video_id'], split_df['subset']))
        else:
            print("Warning: split_file must contain 'video_id' and 'subset' columns. Ignoring.")
    else:
        print("No split_file provided. Defaulting all videos to 'training'.")
        
    # 4. Discover and load CSV annotations
    print(f"Scanning annotations in {args.annotations_dir}...")
    csv_paths = glob.glob(os.path.join(args.annotations_dir, "**", "*.csv"), recursive=True)
    
    if not csv_paths:
        raise ValueError(f"No CSV files found in {args.annotations_dir}")
        
    dfs = []
    for csv_path in csv_paths:
        try:
            df_part = pd.read_csv(csv_path)
            video_name = Path(csv_path).stem
            if CONFIG["col_video_id"] not in df_part.columns:
                df_part[CONFIG["col_video_id"]] = video_name
            dfs.append(df_part)
        except Exception as e:
            print(f"Warning: Failed to read {csv_path}: {e}")
            
    df = pd.concat(dfs, ignore_index=True)
    df[CONFIG["col_label"]] = df[CONFIG["col_label"]].astype(str).str.strip()
    
    # 5. Process videos and build JSON structure
    output_data = {}
    stats = {
        "videos_written": 0,
        "actions_written": 0,
        "missing_features": 0,
        "unknown_labels": 0,
        "invalid_rows": 0
    }
    
    csv_video_ids = set(df[CONFIG["col_video_id"]].unique())
    unknown_labels_seen = set()
    videos_missing_features = set()
    
    # Group annotations by video
    video_groups = df.groupby(CONFIG["col_video_id"])
    
    print("\nProcessing actions and inferring durations...")
    # Iterate over features so we ensure we have the feature file available
    for video_id, feat_path in feat_dict.items():
        
        # Load feature to determine duration safely
        try:
            features = load_features(feat_path)
            valid_segs = determine_valid_segments(features)
            duration = valid_segs * SECONDS_PER_SEGMENT
        except Exception as e:
            print(f"Error reading feature {video_id}: {e}")
            continue
            
        subset = split_dict.get(video_id, "training")
        video_actions = []
        
        if video_id in video_groups.groups:
            video_annos = video_groups.get_group(video_id)
            
            for _, row in video_annos.iterrows():
                # Check NaNs
                if pd.isna(row[CONFIG["col_start"]]) or pd.isna(row[CONFIG["col_end"]]):
                    stats["invalid_rows"] += 1
                    continue
                    
                start_t = row[CONFIG["col_start"]] / CONFIG["annotation_fps"]
                end_t = row[CONFIG["col_end"]] / CONFIG["annotation_fps"]
                action = row[CONFIG["col_label"]]
                
                # Using < instead of <= to allow instantaneous events, acting on prev prompt feedback.
                if end_t < start_t:
                    stats["invalid_rows"] += 1
                    continue
                    
                if action not in class_map:
                    stats["unknown_labels"] += 1
                    unknown_labels_seen.add(action)
                    continue
                    
                class_id = class_map[action]
                
                # Format: [class_id, start_time, end_time]
                video_actions.append([class_id, start_t, end_t])
                
        # We always add the video to output if features exist and duration is known,
        # even if actions list is empty (supported by some repos, or it acts as a negative sample)
        output_data[video_id] = {
            "subset": subset,
            "duration": round(duration, 3), # Round to avoid float precision issues
            "actions": video_actions
        }
        
        stats["videos_written"] += 1
        stats["actions_written"] += len(video_actions)

    # 6. Validate and report
    videos_missing_features = csv_video_ids - set(feat_dict.keys())
    stats["missing_features"] = len(videos_missing_features)
    
    unreferenced_features = set(feat_dict.keys()) - csv_video_ids
    
    # Write JSON
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print("\n" + "="*50)
    print("CONVERSION SUMMARY")
    print("="*50)
    print(f"Target JSON saved to: {args.output_json}")
    print(f"Videos written: {stats['videos_written']}")
    print(f"Action annotations written: {stats['actions_written']}")
    print(f"Invalid bounding boxes skipped: {stats['invalid_rows']}")
    print(f"Annotations dropped (unknown label): {stats['unknown_labels']}")
    
    print("\n" + "-"*50)
    print("VALIDATION REPORT")
    print("-"*50)
    if videos_missing_features:
        print(f"[{len(videos_missing_features)}] Videos in CSV missing features (e.g. {list(videos_missing_features)[:3]})")
    else:
        print("[0] Videos in CSV missing features")
        
    if unreferenced_features:
        print(f"[{len(unreferenced_features)}] Feature files found with NO matching annotations (added as empty)")
    else:
        print("[0] Unreferenced feature files")
        
    if unknown_labels_seen:
        print(f"\n[{len(unknown_labels_seen)}] Labels in CSV missing from class_mapping.json:")
        for ul in sorted(unknown_labels_seen):
            print(f"  - '{ul}'")

if __name__ == "__main__":
    main()