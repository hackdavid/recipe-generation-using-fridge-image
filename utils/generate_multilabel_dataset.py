"""
Multi-Label Dataset Generator

This script generates a multi-label classification dataset by composing multiple
single-class images into realistic composite images.

Features:
- Multiple composition strategies (grid, overlay, natural arrangement)
- Class imbalance handling (oversampling rare classes)
- Realistic image composition with proper blending
- Automatic annotation generation
- Train/val/test split generation
- Dataset statistics and visualization

Usage:
    python data/generate_multilabel_dataset.py \
        --source_dir ./merged_dataset \
        --output_dir ./multilabel_dataset \
        --num_images 13000 \
        --min_labels 2 \
        --max_labels 5
"""

import os
import json
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class ImageComposer:
    """Handles realistic image composition"""
    
    def __init__(self, canvas_size=(512, 512), background_color=(255, 255, 255)):
        """
        Initialize image composer
        
        Args:
            canvas_size: Size of output composite image
            background_color: Background color (RGB)
        """
        self.canvas_size = canvas_size
        self.background_color = background_color
    
    def compose_grid(self, images: List[Image.Image], labels: List[str]) -> Tuple[Image.Image, List[str]]:
        """
        Compose images in a grid layout
        
        Args:
            images: List of PIL Images
            labels: List of class labels
        
        Returns:
            Composite image and labels
        """
        num_images = len(images)
        
        # Determine grid size
        if num_images == 2:
            grid_size = (1, 2)
        elif num_images == 3:
            grid_size = (1, 3)
        elif num_images == 4:
            grid_size = (2, 2)
        elif num_images == 5:
            grid_size = (2, 3)  # One empty space
        else:
            grid_size = (2, 3)
        
        # Calculate cell size
        cell_width = self.canvas_size[0] // grid_size[1]
        cell_height = self.canvas_size[1] // grid_size[0]
        
        # Create canvas
        canvas = Image.new('RGB', self.canvas_size, self.background_color)
        
        # Place images
        for idx, img in enumerate(images[:num_images]):
            row = idx // grid_size[1]
            col = idx % grid_size[1]
            
            # Resize image to fit cell (with padding)
            img_resized = self._resize_with_padding(img, (cell_width - 20, cell_height - 20))
            
            # Calculate position
            x = col * cell_width + 10
            y = row * cell_height + 10
            
            # Paste image
            canvas.paste(img_resized, (x, y))
        
        return canvas, labels
    
    def compose_overlay(self, images: List[Image.Image], labels: List[str]) -> Tuple[Image.Image, List[str]]:
        """
        Compose images with overlay/blending (more realistic)
        
        Args:
            images: List of PIL Images
            labels: List of class labels
        
        Returns:
            Composite image and labels
        """
        # Create canvas with subtle texture
        canvas = Image.new('RGB', self.canvas_size, self.background_color)
        
        # Add subtle background texture
        canvas = self._add_texture(canvas)
        
        # Shuffle images for random placement
        image_order = list(range(len(images)))
        random.shuffle(image_order)
        
        # Place images with random positions and sizes
        placed_regions = []
        
        for idx in image_order:
            img = images[idx]
            
            # Random size (40-70% of canvas)
            scale = random.uniform(0.4, 0.7)
            new_size = (
                int(self.canvas_size[0] * scale),
                int(self.canvas_size[1] * scale)
            )
            
            # Resize image
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Try to find non-overlapping position
            position = self._find_non_overlapping_position(
                new_size, placed_regions, max_attempts=50
            )
            
            if position:
                # Apply subtle transformations
                img_transformed = self._apply_transformations(img_resized)
                
                # Blend with canvas
                canvas = self._blend_image(canvas, img_transformed, position)
                
                placed_regions.append({
                    'position': position,
                    'size': new_size
                })
            else:
                # If can't find non-overlapping, place anyway with transparency
                position = (
                    random.randint(0, self.canvas_size[0] - new_size[0]),
                    random.randint(0, self.canvas_size[1] - new_size[1])
                )
                canvas = self._blend_image(canvas, img_transformed, position, alpha=0.7)
        
        return canvas, labels
    
    def compose_natural(self, images: List[Image.Image], labels: List[str]) -> Tuple[Image.Image, List[str]]:
        """
        Compose images in natural arrangement (most realistic)
        
        Args:
            images: List of PIL Images
            labels: List of class labels
        
        Returns:
            Composite image and labels
        """
        # Create canvas with natural background
        canvas = Image.new('RGB', self.canvas_size, self.background_color)
        canvas = self._add_natural_background(canvas)
        
        # Sort images by size (larger first for better composition)
        image_sizes = [(img.size[0] * img.size[1], i) for i, img in enumerate(images)]
        image_sizes.sort(reverse=True)
        
        placed_regions = []
        
        for _, idx in image_sizes:
            img = images[idx]
            
            # Determine size based on "importance" (first images larger)
            # Ensure sizes fit within canvas
            if len(placed_regions) == 0:
                scale = random.uniform(0.4, 0.6)  # Largest item (reduced max to ensure fit)
            elif len(placed_regions) == 1:
                scale = random.uniform(0.3, 0.5)  # Second largest
            else:
                scale = random.uniform(0.25, 0.4)  # Smaller items
            
            # Ensure scale doesn't exceed canvas
            max_scale_x = min(0.9, self.canvas_size[0] / max(img.size[0], 1))
            max_scale_y = min(0.9, self.canvas_size[1] / max(img.size[1], 1))
            scale = min(scale, max_scale_x, max_scale_y)
            
            new_size = (
                int(self.canvas_size[0] * scale),
                int(self.canvas_size[1] * scale)
            )
            
            # Ensure minimum size
            new_size = (max(50, new_size[0]), max(50, new_size[1]))
            
            # Resize with aspect ratio
            img_resized = self._resize_keep_aspect(img, new_size)
            
            # Ensure resized image fits in canvas
            if img_resized.size[0] > self.canvas_size[0] or img_resized.size[1] > self.canvas_size[1]:
                img_resized.thumbnail(self.canvas_size, Image.Resampling.LANCZOS)
            
            # Find natural position (prefer lower center for "table" effect)
            position = self._find_natural_position(
                img_resized.size, placed_regions
            )
            
            # Apply realistic transformations
            img_transformed = self._apply_realistic_transformations(img_resized)
            
            # Add shadow for depth
            img_with_shadow = self._add_shadow(img_transformed)
            
            # Blend naturally
            canvas = self._blend_naturally(canvas, img_with_shadow, position)
            
            placed_regions.append({
                'position': position,
                'size': img_resized.size
            })
        
        return canvas, labels
    
    def _resize_with_padding(self, img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize image with padding to maintain aspect ratio"""
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create new image with padding
        new_img = Image.new('RGB', target_size, (255, 255, 255))
        paste_x = (target_size[0] - img.size[0]) // 2
        paste_y = (target_size[1] - img.size[1]) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img
    
    def _resize_keep_aspect(self, img: Image.Image, max_size: Tuple[int, int]) -> Image.Image:
        """Resize image keeping aspect ratio"""
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        return img
    
    def _find_non_overlapping_position(
        self, size: Tuple[int, int], placed_regions: List[Dict], max_attempts: int = 50
    ) -> Optional[Tuple[int, int]]:
        """Find position that doesn't overlap with placed regions"""
        for _ in range(max_attempts):
            x = random.randint(0, max(1, self.canvas_size[0] - size[0]))
            y = random.randint(0, max(1, self.canvas_size[1] - size[1]))
            
            # Check overlap
            overlaps = False
            for region in placed_regions:
                if self._check_overlap(
                    (x, y), size,
                    region['position'], region['size']
                ):
                    overlaps = True
                    break
            
            if not overlaps:
                return (x, y)
        
        return None
    
    def _find_natural_position(
        self, size: Tuple[int, int], placed_regions: List[Dict]
    ) -> Tuple[int, int]:
        """Find natural position (prefer lower center area)"""
        # Prefer positions in lower 2/3 of canvas
        preferred_y_min = self.canvas_size[1] // 3
        preferred_y_max = max(preferred_y_min, self.canvas_size[1] - size[1])
        
        # Ensure valid range
        if preferred_y_max < preferred_y_min:
            # Image is too large for preferred area, use full canvas
            preferred_y_min = 0
            preferred_y_max = max(0, self.canvas_size[1] - size[1])
        
        # Try preferred area first
        for _ in range(30):
            x = random.randint(0, max(1, self.canvas_size[0] - size[0]))
            
            # Ensure valid y range
            if preferred_y_max >= preferred_y_min:
                y = random.randint(preferred_y_min, preferred_y_max)
            else:
                y = random.randint(0, max(0, self.canvas_size[1] - size[1]))
            
            overlaps = False
            for region in placed_regions:
                if self._check_overlap(
                    (x, y), size,
                    region['position'], region['size']
                ):
                    overlaps = True
                    break
            
            if not overlaps:
                return (x, y)
        
        # Fallback to any position
        fallback_pos = self._find_non_overlapping_position(size, placed_regions)
        if fallback_pos:
            return fallback_pos
        
        # Last resort: random position (may overlap)
        return (
            random.randint(0, max(1, self.canvas_size[0] - size[0])),
            random.randint(0, max(1, self.canvas_size[1] - size[1]))
        )
    
    def _check_overlap(
        self, pos1: Tuple[int, int], size1: Tuple[int, int],
        pos2: Tuple[int, int], size2: Tuple[int, int]
    ) -> bool:
        """Check if two rectangles overlap"""
        x1_min, y1_min = pos1
        x1_max, y1_max = x1_min + size1[0], y1_min + size1[1]
        x2_min, y2_min = pos2
        x2_max, y2_max = x2_min + size2[0], y2_min + size2[1]
        
        return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)
    
    def _apply_transformations(self, img: Image.Image) -> Image.Image:
        """Apply subtle transformations"""
        # Random rotation
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))
        
        # Random brightness
        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.9, 1.1))
        
        return img
    
    def _apply_realistic_transformations(self, img: Image.Image) -> Image.Image:
        """Apply realistic transformations for natural composition"""
        # Slight rotation
        if random.random() < 0.4:
            angle = random.uniform(-10, 10)
            img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))
        
        # Brightness adjustment
        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.95, 1.05))
        
        # Contrast adjustment
        if random.random() < 0.2:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.95, 1.05))
        
        # Slight blur for depth
        if random.random() < 0.1:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return img
    
    def _blend_image(
        self, canvas: Image.Image, img: Image.Image, position: Tuple[int, int], alpha: float = 0.95
    ) -> Image.Image:
        """Blend image onto canvas"""
        # Create alpha channel if needed
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Create alpha mask
        alpha_mask = Image.new('L', img.size, int(255 * alpha))
        img.putalpha(alpha_mask)
        
        # Paste with alpha
        canvas.paste(img, position, img)
        return canvas
    
    def _blend_naturally(
        self, canvas: Image.Image, img: Image.Image, position: Tuple[int, int]
    ) -> Image.Image:
        """Blend image naturally onto canvas"""
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Use high alpha for natural look
        alpha_mask = Image.new('L', img.size, 250)
        img.putalpha(alpha_mask)
        
        canvas.paste(img, position, img)
        return canvas
    
    def _add_texture(self, canvas: Image.Image) -> Image.Image:
        """Add subtle texture to background"""
        # Create noise texture
        noise = np.random.randint(240, 255, (canvas.size[1], canvas.size[0], 3), dtype=np.uint8)
        texture = Image.fromarray(noise)
        
        # Blend with canvas
        canvas = Image.blend(canvas, texture, 0.1)
        return canvas
    
    def _add_natural_background(self, canvas: Image.Image) -> Image.Image:
        """Add natural background (wooden table, white surface, etc.)"""
        # Create gradient background
        draw = ImageDraw.Draw(canvas)
        
        # Light gradient from top to bottom
        for y in range(canvas.size[1]):
            color_value = int(255 - (y / canvas.size[1]) * 10)
            color = (color_value, color_value, color_value)
            draw.line([(0, y), (canvas.size[0], y)], fill=color)
        
        # Add subtle texture
        canvas = self._add_texture(canvas)
        
        return canvas
    
    def _add_shadow(self, img: Image.Image) -> Image.Image:
        """Add shadow effect to image"""
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Create shadow (simplified - skip if too complex)
        try:
            shadow = Image.new('RGBA', (img.size[0] + 10, img.size[1] + 10), (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow)
            
            # Draw shadow ellipse
            shadow_draw.ellipse(
                [(5, img.size[1] - 5), (img.size[0] + 5, img.size[1] + 5)],
                fill=(0, 0, 0, 50)
            )
            
            # Blur shadow (may not be available in all PIL versions)
            try:
                shadow = shadow.filter(ImageFilter.GaussianBlur(radius=5))
            except:
                # Fallback: no blur
                pass
            
            # Composite shadow and image
            result = Image.new('RGBA', shadow.size, (255, 255, 255, 0))
            result.paste(shadow, (0, 0))
            result.paste(img, (0, 0), img)
            
            return result
        except:
            # Fallback: return original image
            return img


class MultiLabelDatasetGenerator:
    """Generates multi-label dataset from single-class images"""
    
    def __init__(
        self, source_dir: str, output_dir: str,
        min_labels: int = 2, max_labels: int = 5,
        canvas_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initialize dataset generator
        
        Args:
            source_dir: Directory with single-class folders
            output_dir: Output directory for multi-label dataset
            min_labels: Minimum labels per image
            max_labels: Maximum labels per image
            canvas_size: Size of composite images
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.min_labels = min_labels
        self.max_labels = max_labels
        self.canvas_size = canvas_size
        
        # Create output structure (no splits - single folder for HuggingFace)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'annotations').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        
        # Initialize composer
        self.composer = ImageComposer(canvas_size=canvas_size)
        
        # Load class information
        self.classes = self._load_classes()
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}
        self.id_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        
        # Load images per class
        self.images_per_class = self._load_images_per_class()
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'label_distribution': Counter(),
            'co_occurrence': defaultdict(int),
            'label_count_distribution': Counter()
        }
    
    def _load_classes(self) -> List[str]:
        """Load class names from source directory"""
        classes = sorted([d.name for d in self.source_dir.iterdir() if d.is_dir()])
        print(f"✓ Loaded {len(classes)} classes")
        return classes
    
    def _load_images_per_class(self) -> Dict[str, List[Path]]:
        """Load all image paths per class"""
        images_per_class = {}
        
        for cls in self.classes:
            class_dir = self.source_dir / cls
            image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
            images = [
                img_path for img_path in class_dir.iterdir()
                if img_path.suffix in image_extensions
            ]
            
            if len(images) == 0:
                print(f"⚠ Warning: No images found in {cls}")
                continue
            
            images_per_class[cls] = images
            print(f"  {cls}: {len(images)} images")
        
        return images_per_class
    
    def _get_label_count_distribution(self) -> Dict[int, float]:
        """Get distribution of label counts"""
        # Target distribution: 2 labels (40%), 3 labels (35%), 4 labels (20%), 5 labels (5%)
        return {
            2: 0.40,
            3: 0.35,
            4: 0.20,
            5: 0.05
        }
    
    def _sample_label_count(self) -> int:
        """Sample number of labels based on distribution"""
        dist = self._get_label_count_distribution()
        counts = list(dist.keys())
        probabilities = [dist[c] for c in counts]
        return np.random.choice(counts, p=probabilities)
    
    def _sample_classes(self, num_labels: int, oversample_rare: bool = True) -> List[str]:
        """Sample classes for composite image"""
        # Calculate class frequencies
        class_freqs = {cls: len(images) for cls, images in self.images_per_class.items()}
        total_images = sum(class_freqs.values())
        
        if oversample_rare:
            # Inverse frequency weighting (rare classes more likely)
            weights = {cls: 1.0 / (freq / total_images + 0.01) for cls, freq in class_freqs.items()}
            total_weight = sum(weights.values())
            probabilities = {cls: w / total_weight for cls, w in weights.items()}
        else:
            # Uniform sampling
            probabilities = {cls: 1.0 / len(self.classes) for cls in self.classes}
        
        # Sample without replacement
        classes = []
        available_classes = list(self.classes)
        
        for _ in range(num_labels):
            if not available_classes:
                break
            
            # Sample class
            probs = [probabilities[cls] for cls in available_classes]
            probs = np.array(probs)
            probs = probs / probs.sum()
            
            sampled_class = np.random.choice(available_classes, p=probs)
            classes.append(sampled_class)
            available_classes.remove(sampled_class)
        
        return classes
    
    def _load_image(self, image_path: Path) -> Image.Image:
        """Load and preprocess image"""
        try:
            img = Image.open(image_path).convert('RGB')
            # Resize if too large
            max_size = max(self.canvas_size)
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            return img
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def generate_composite(
        self, classes: List[str], composition_method: str = 'natural'
    ) -> Tuple[Optional[Image.Image], List[str]]:
        """
        Generate composite image from classes
        
        Args:
            classes: List of class names
            composition_method: 'grid', 'overlay', or 'natural'
        
        Returns:
            Composite image and labels
        """
        # Sample one image per class
        images = []
        valid_classes = []
        
        for cls in classes:
            if cls not in self.images_per_class:
                continue
            
            # Randomly sample image from class
            image_path = random.choice(self.images_per_class[cls])
            img = self._load_image(image_path)
            
            if img is not None:
                images.append(img)
                valid_classes.append(cls)
        
        if len(images) < self.min_labels:
            return None, []
        
        # Compose images
        if composition_method == 'grid':
            composite, labels = self.composer.compose_grid(images, valid_classes)
        elif composition_method == 'overlay':
            composite, labels = self.composer.compose_overlay(images, valid_classes)
        else:  # natural
            composite, labels = self.composer.compose_natural(images, valid_classes)
        
        return composite, labels
    
    def generate_dataset(
        self, num_images: int,
        composition_method: str = 'natural',
        oversample_rare: bool = True
    ):
        """
        Generate complete multi-label dataset (no splits - single folder for HuggingFace)
        
        Args:
            num_images: Total number of images to generate
            composition_method: Composition method ('grid', 'overlay', or 'natural')
            oversample_rare: Whether to oversample rare classes
        """
        print("\n" + "="*70)
        print("Generating Multi-Label Dataset")
        print("="*70)
        print(f"Generating {num_images} composite images...")
        print("(No splits - all images in single folder for HuggingFace upload)")
        
        annotations = {}
        failed = 0
        
        for idx in range(num_images):
            if (idx + 1) % 100 == 0:
                print(f"  Generated {idx + 1}/{num_images} images...")
            
            # Sample number of labels
            num_labels = self._sample_label_count()
            
            # Sample classes
            classes = self._sample_classes(num_labels, oversample_rare)
            
            # Generate composite
            composite, labels = self.generate_composite(classes, composition_method)
            
            if composite is None or len(labels) < self.min_labels:
                failed += 1
                continue
            
            # Save image (single folder, no split prefix)
            image_filename = f"composite_{idx:06d}.jpg"
            image_path = self.output_dir / 'images' / image_filename
            composite.save(image_path, 'JPEG', quality=95)
            
            # Create annotation
            label_ids = [self.class_to_id[cls] for cls in labels]
            label_ids.sort()
            
            annotation = {
                'labels': label_ids,
                'label_names': labels,
                'num_labels': len(labels)
            }
            
            annotations[image_filename] = annotation
            
            # Update statistics
            self.stats['total_images'] += 1
            self.stats['label_count_distribution'][len(labels)] += 1
            for label in labels:
                self.stats['label_distribution'][label] += 1
            
            # Track co-occurrence
            for i, label1 in enumerate(labels):
                for label2 in labels[i+1:]:
                    pair = tuple(sorted([label1, label2]))
                    self.stats['co_occurrence'][pair] += 1
        
        # Save single annotation file
        annotation_path = self.output_dir / 'annotations' / 'labels.json'
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Generated {len(annotations)} images (failed: {failed})")
        
        # Save metadata
        self._save_metadata(annotations)
        
        # Generate statistics
        self._generate_statistics()
        
        print("\n" + "="*70)
        print("Dataset Generation Complete!")
        print("="*70)
        print(f"Total images: {self.stats['total_images']}")
        print(f"Images saved to: {self.output_dir / 'images'}")
        print(f"Annotations saved to: {annotation_path}")
        print(f"\nReady for HuggingFace upload!")
    
    def _save_metadata(self, annotations: Dict):
        """Save dataset metadata"""
        metadata = {
            'dataset_name': 'multi_label_food_recognition',
            'num_classes': len(self.classes),
            'classes': self.classes,
            'class_to_id': self.class_to_id,
            'id_to_class': self.id_to_class,
            'generation_date': datetime.now().isoformat(),
            'canvas_size': self.canvas_size,
            'min_labels': self.min_labels,
            'max_labels': self.max_labels,
            'total_images': len(annotations),
            'note': 'No splits - ready for HuggingFace upload (splits can be created during upload)'
        }
        
        metadata_path = self.output_dir / 'metadata' / 'dataset_info.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved metadata to {metadata_path}")
    
    def _generate_statistics(self):
        """Generate and save dataset statistics"""
        stats_path = self.output_dir / 'metadata' / 'statistics.json'
        
        # Convert defaultdict to dict for JSON
        # Convert tuple keys to strings for JSON compatibility
        co_occurrence_dict = {
            f"{pair[0]}_{pair[1]}": count
            for pair, count in self.stats['co_occurrence'].items()
        }
        
        # Convert top co-occurrences (tuples) to string keys
        top_co_occurrences = {}
        for (cls1, cls2), count in Counter(self.stats['co_occurrence']).most_common(20):
            key = f"{cls1}_{cls2}"
            top_co_occurrences[key] = count
        
        stats_dict = {
            'total_images': self.stats['total_images'],
            'label_distribution': dict(self.stats['label_distribution']),
            'label_count_distribution': dict(self.stats['label_count_distribution']),
            'co_occurrence': co_occurrence_dict,
            'top_co_occurrences': top_co_occurrences
        }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved statistics to {stats_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("Dataset Statistics")
        print("="*70)
        print(f"\nLabel Count Distribution:")
        for count, freq in sorted(self.stats['label_count_distribution'].items()):
            percentage = (freq / self.stats['total_images']) * 100
            print(f"  {count} labels: {freq} ({percentage:.1f}%)")
        
        print(f"\nTop 10 Most Frequent Classes:")
        for cls, count in self.stats['label_distribution'].most_common(10):
            percentage = (count / self.stats['total_images']) * 100
            print(f"  {cls}: {count} ({percentage:.1f}%)")
        
        print(f"\nTop 10 Class Co-occurrences:")
        for (cls1, cls2), count in Counter(self.stats['co_occurrence']).most_common(10):
            print(f"  {cls1} + {cls2}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Generate multi-label dataset (no splits - for HuggingFace)')
    parser.add_argument('--source_dir', type=str, required=True,
                       help='Source directory with single-class folders')
    parser.add_argument('--output_dir', type=str, default='./multilabel_dataset',
                       help='Output directory for multi-label dataset')
    parser.add_argument('--num_images', type=int, default=13000,
                       help='Total number of images to generate (no splits)')
    parser.add_argument('--min_labels', type=int, default=2,
                       help='Minimum labels per image')
    parser.add_argument('--max_labels', type=int, default=5,
                       help='Maximum labels per image')
    parser.add_argument('--composition_method', type=str, default='natural',
                       choices=['grid', 'overlay', 'natural'],
                       help='Composition method')
    parser.add_argument('--canvas_size', type=int, nargs=2, default=[512, 512],
                       help='Canvas size (width height)')
    parser.add_argument('--no_oversample', action='store_true',
                       help='Disable oversampling of rare classes')
    
    args = parser.parse_args()
    
    # Create generator
    generator = MultiLabelDatasetGenerator(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        min_labels=args.min_labels,
        max_labels=args.max_labels,
        canvas_size=tuple(args.canvas_size)
    )
    
    # Generate dataset (no splits)
    generator.generate_dataset(
        num_images=args.num_images,
        composition_method=args.composition_method,
        oversample_rare=not args.no_oversample
    )
    
    print(f"\n✓ Dataset saved to: {args.output_dir}")
    print(f"  Ready for HuggingFace upload!")
    print(f"  Images: {args.output_dir}/images/")
    print(f"  Annotations: {args.output_dir}/annotations/labels.json")


if __name__ == '__main__':
    main()

