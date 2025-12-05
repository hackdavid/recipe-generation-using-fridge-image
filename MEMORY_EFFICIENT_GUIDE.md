# Memory-Efficient Dataset Upload Guide

## Problem: Memory Errors

If you're getting memory errors when uploading the dataset, the script now supports two methods:

## Method 1: Batch Processing (Default)

Processes images in batches to reduce memory usage.

### Usage:
```bash
python upload_dataset_to_hf.py \
    --dataset_name "your-username/dataset-name" \
    --batch_size 500
```

### Adjust Batch Size:
- **Default**: 1000 images per batch
- **Low memory**: Use `--batch_size 500` or `--batch_size 200`
- **Very low memory**: Use `--batch_size 100`

### How it works:
1. Scans all image paths (doesn't load images yet)
2. Creates train/val split based on paths
3. Processes images in batches (loads → processes → releases memory)
4. Accumulates results and creates dataset

## Method 2: ImageFolder Format (Recommended for Large Datasets)

Uses HuggingFace's built-in imagefolder loader which is more memory efficient.

### Usage:
```bash
python upload_dataset_to_hf.py \
    --dataset_name "your-username/dataset-name" \
    --use_imagefolder \
    --temp_dir "temp_dataset"
```

### How it works:
1. Creates temporary train/val folder structure
2. Copies images to temp folders (maintains class structure)
3. Uses HuggingFace's efficient imagefolder loader
4. Cleans up temp directory after upload

### Advantages:
- More memory efficient
- Uses HuggingFace's optimized loading
- Better for very large datasets

### Disadvantages:
- Requires disk space for temporary files
- Takes longer (copying files)

## Memory Recommendations

### If you have < 8GB RAM:
```bash
# Use imagefolder format
python upload_dataset_to_hf.py \
    --dataset_name "your-username/dataset-name" \
    --use_imagefolder \
    --batch_size 200
```

### If you have 8-16GB RAM:
```bash
# Use batch processing with smaller batches
python upload_dataset_to_hf.py \
    --dataset_name "your-username/dataset-name" \
    --batch_size 500
```

### If you have > 16GB RAM:
```bash
# Default settings should work
python upload_dataset_to_hf.py \
    --dataset_name "your-username/dataset-name"
```

## Troubleshooting Memory Issues

### 1. Reduce Batch Size
```bash
--batch_size 100  # Process fewer images at once
```

### 2. Use ImageFolder Format
```bash
--use_imagefolder  # More memory efficient
```

### 3. Close Other Applications
- Close browser tabs
- Close other Python processes
- Free up RAM before running

### 4. Process on Smaller Machine/Cloud
- Use Google Colab (free GPU + RAM)
- Use Kaggle Notebooks
- Use cloud instances with more RAM

## Example: Low Memory Setup

```bash
# Set token
$env:HF_TOKEN = "your_token_here"

# Upload with imagefolder format and small batches
python upload_dataset_to_hf.py \
    --dataset_name "your-username/food-recognition" \
    --use_imagefolder \
    --temp_dir "temp_dataset" \
    --batch_size 200 \
    --hf_token $env:HF_TOKEN
```

## Monitoring Memory Usage

### Windows PowerShell:
```powershell
Get-Process python | Select-Object ProcessName, @{Name="Memory(MB)";Expression={[math]::Round($_.WS/1MB,2)}}
```

### Check available disk space:
```powershell
Get-PSDrive C | Select-Object Used,Free
```

## Tips

1. **Start small**: Test with `--batch_size 100` first
2. **Monitor progress**: Watch memory usage during processing
3. **Use imagefolder**: If batch processing fails, use `--use_imagefolder`
4. **Clean temp files**: Script automatically cleans up, but check if `temp_dataset` folder exists
5. **Resume capability**: If upload fails, you can resume (dataset is created incrementally)

## Expected Memory Usage

- **Batch processing (batch_size=1000)**: ~4-6GB RAM
- **Batch processing (batch_size=500)**: ~2-4GB RAM  
- **Batch processing (batch_size=200)**: ~1-2GB RAM
- **ImageFolder format**: ~500MB-1GB RAM (but needs disk space)

