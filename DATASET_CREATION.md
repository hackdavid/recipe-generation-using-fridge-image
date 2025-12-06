# Dataset Creation: Merged Raw Food Recognition Dataset

## Overview

This document describes the creation process of a comprehensive raw food recognition dataset, which has been published on HuggingFace at [ibrahimdaud/raw-food-recognition](https://huggingface.co/datasets/ibrahimdaud/raw-food-recognition).

## Motivation

The primary goal of this project is to develop robust visual classification models for raw food items. However, existing publicly available food datasets were found to be insufficient for this purpose due to:

1. **Limited Class Coverage**: Individual datasets lacked comprehensive coverage of raw food categories
2. **Inconsistent Labeling**: Different datasets used varying naming conventions, spellings, and category structures
3. **Fragmented Data**: No single dataset provided the diversity needed for robust model training
4. **Focus Mismatch**: Most datasets focused on prepared foods or packaged products rather than raw ingredients

To address these limitations, we decided to merge multiple datasets and create a unified, comprehensive dataset specifically tailored for raw food classification tasks.

## Source Datasets

The merged dataset combines three publicly available datasets:

### 1. Food and Vegetables Dataset
- **Source**: [SunnyAgarwal4274/Food_and_Vegetables](https://huggingface.co/datasets/SunnyAgarwal4274/Food_and_Vegetables) (HuggingFace)
- **Focus**: Raw vegetables and fruits
- **Characteristics**: 
  - Wide range of produce items
  - Varied label formatting (spacing, casing, spelling variations)
  - Examples: "Bell pepper", "Bitter Gourd", "Dragon_fruit", "Sweetpotato"

### 2. Fruit and Vegetable Image Recognition Dataset
- **Source**: [Nattakarn/fruit-and-vegetable-image-recognition](https://huggingface.co/datasets/Nattakarn/fruit-and-vegetable-image-recognition) (HuggingFace)
- **Focus**: Primarily fruits with emphasis on exotic varieties
- **Characteristics**:
  - Title case hyphenated labels
  - Additional citrus fruit variants
  - Examples: "Avocado", "Passion-Fruit", "Red-Grapefruit", "Satsumas"

### 3. Grocery Store Dataset
- **Source**: Kaggle Grocery Store Dataset
- **Focus**: Packaged consumer goods including beverages and dairy products
- **Characteristics**:
  - Brand-specific directory structures
  - Packaged products (juices, yoghurts, milk variants)
  - Examples: "Bravo-Apple-Juice", "Arla-Standard-Milk", "Oatly-Natural-Oatghurt"

## Dataset Creation Methodology

### Phase 1: Data Collection and Organization

1. **Data Acquisition**: Collected images from all three source datasets
2. **Initial Organization**: Grouped images by food category from each source
3. **Format Standardization**: Ensured consistent image formats (JPEG/PNG)

### Phase 2: Class Unification and Normalization

#### 2.1 Standardization of Base Class Naming

Established a consistent naming convention using **snake_case** format for compatibility with machine learning frameworks (PyTorch, TensorFlow, HuggingFace):

- "Bell pepper" → `bell_pepper`
- "Red-Grapefruit" → `red_grapefruit`
- "Sweetpotato" → `sweet_potato`
- "Dragon_fruit" → `dragon_fruit`

#### 2.2 Handling Synonyms and Variants

Applied rule-based semantic merging to consolidate equivalent classes:

**Synonyms Merged:**
- Capsicum, Bell pepper, Paprika → `bell_pepper`
- Aubergine → `eggplant`

**Plural Reduction:**
- Carrots → `carrot`
- Peas → `pea` (where applicable)

**Spelling Corrections:**
- Raddish → `radish`
- Jalepeno → `jalapeno`

#### 2.3 Introduction of New Base Classes

Integrated previously unseen categories from additional datasets:

**From Fruit Dataset:**
- avocado, lime, melon, nectarine, peach, papaya, plum, passion_fruit, satsuma

**From Fresh Produce Dataset:**
- asparagus, leek, mushroom, zucchini

#### 2.4 Unification of Packaged Goods

For the grocery store dataset, converted brand-specific directory paths into semantic product classes:

**Transformation Rules:**
- `Bravo-Apple-Juice` → `apple_juice`
- `Arla-Standard-Milk` → `milk`
- `Alpro-Vanilla-Soyghurt` → `vanilla_soyghurt`
- `Oatly-Natural-Oatghurt` → `oat_yogurt`

**Rationale**: Removed brand names to focus on product type, improving model generalizability and avoiding unnecessary class proliferation.

### Phase 3: Data Processing and Quality Control

1. **Deduplication**: Identified and removed duplicate images across datasets
2. **Quality Filtering**: Ensured minimum image quality standards
3. **Label Verification**: Validated all class mappings and transformations
4. **Stratified Splitting**: Created train/validation splits (80/20) maintaining class distribution

### Phase 4: Format Conversion and Storage

1. **Parquet Conversion**: Converted images to Parquet format for efficient storage and fast loading
2. **Metadata Creation**: Generated comprehensive class mapping files
3. **Dataset Structure**: Organized into train and validation splits

## Final Dataset Characteristics

### Statistics

- **Total Classes**: 90+ food categories
- **Total Images**: ~15,000+ images
- **Train Split**: ~12,000 images (80%)
- **Validation Split**: ~3,000 images (20%)
- **Image Format**: JPEG/PNG
- **Storage Format**: Parquet (auto-converted by HuggingFace)
- **Dataset Size**: 3.1 GB

### Category Coverage

The unified dataset spans:

- **Fresh Fruits**: apple, banana, orange, strawberry, grapes, mango, kiwi, watermelon, dragon_fruit, passion_fruit, etc.
- **Vegetables**: carrot, tomato, potato, onion, broccoli, cauliflower, spinach, asparagus, etc.
- **Gourds**: bottle_gourd, bitter_gourd, ridge_gourd, sponge_gourd, spiny_gourd
- **Citrus Varieties**: orange, lemon, lime, grapefruit, red_grapefruit, satsuma, mandarin
- **Tropical Fruits**: mango, papaya, coconut, pineapple, etc.
- **Mushrooms and Leafy Vegetables**: mushroom, lettuce, spinach, cabbage
- **Dairy Products**: milk, yogurt (various flavors), sour_cream, natural_yogurt
- **Plant-Based Alternatives**: soy_milk, oat_milk, soy_yogurt, oat_yogurt
- **Beverages**: apple_juice, orange_juice, grapefruit_juice, mixed_juice

### Dataset Structure

Each sample in the dataset contains:

```python
{
    'image': PIL.Image,           # Image object of the food item
    'label': str,                 # String label (e.g., "apple", "banana")
    'label_id': int               # Integer ID (0 to num_classes-1)
}
```

## Dataset Publication

### HuggingFace Repository

The dataset has been published and is available at:

**🔗 [ibrahimdaud/raw-food-recognition](https://huggingface.co/datasets/ibrahimdaud/raw-food-recognition)**

### Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("ibrahimdaud/raw-food-recognition")

# Access splits
train_dataset = dataset['train']      # ~10k samples
val_dataset = dataset['validation']   # ~2.5k samples

# Example usage
sample = train_dataset[0]
image = sample['image']        # PIL Image
label = sample['label']        # String label
label_id = sample['label_id'] # Integer ID
```

### Integration with Training Pipeline

The dataset is designed to work seamlessly with our training pipeline:

```bash
# Generate class mapping (one-time)
python trainer/generate_class_mapping.py \
    --dataset_name "ibrahimdaud/raw-food-recognition" \
    --output trainer/class_mapping.json

# Train model
python trainer/train.py configs/resnet50_config.yaml
```

## Key Achievements

### 1. Comprehensive Class Coverage

Successfully integrated 90+ food categories from three diverse sources, creating the most comprehensive raw food recognition dataset available.

### 2. Label Consistency

Eliminated redundancy and ambiguity through:
- Standardized naming conventions
- Synonym resolution
- Spelling normalization
- Format unification

### 3. Domain-Specific Focus

Created a dataset specifically tailored for raw food classification, distinguishing it from datasets focused on prepared foods or restaurant dishes.

### 4. Research-Ready Format

- HuggingFace-compatible structure
- Efficient Parquet storage
- Proper train/validation splits
- Complete metadata and documentation

### 5. Reproducibility

All transformation rules and mapping processes are documented, enabling:
- Reproducible results
- Easy extension to new datasets
- Clear understanding of class relationships

## Dataset Limitations and Considerations

1. **Class Imbalance**: Some categories have more samples than others due to source dataset variations
2. **Image Quality Variability**: Images come from different sources with varying resolutions and quality
3. **Label Variants**: Some food items maintain multiple label variants (e.g., "yogurt" vs "yoghurt") for dataset diversity
4. **Educational Purpose**: This dataset is intended for educational and research purposes

## Citation

If you use this dataset in your research or projects, please cite:

```bibtex
@dataset{raw_food_recognition_2024,
  title={Merged Raw Food Recognition Dataset},
  author={Ibrahim Daud},
  year={2024},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/ibrahimdaud/raw-food-recognition},
  note={Combined from: SunnyAgarwal4274/Food_and_Vegetables, 
        Nattakarn/fruit-and-vegetable-image-recognition, 
        and Kaggle Grocery Store Dataset}
}
```

## Acknowledgments

We gratefully acknowledge the creators and contributors of the source datasets:

- **SunnyAgarwal4274** for the Food and Vegetables dataset
- **Nattakarn** for the Fruit and Vegetable Image Recognition dataset
- **Kaggle Community** for the Grocery Store Dataset

## Impact on Model Training

This unified dataset enables:

1. **Robust Training**: Diverse samples improve model generalization
2. **Comprehensive Evaluation**: Wide class coverage tests model capabilities
3. **Transfer Learning**: Large-scale dataset suitable for fine-tuning pretrained models
4. **Research Applications**: Foundation for food recognition research and applications

## Future Enhancements

Potential improvements for future versions:

- Additional food categories
- More balanced class distribution
- Higher resolution images
- Extended metadata (nutritional information, origin, etc.)
- Multi-label support for compound food items

---

**Dataset Repository**: [https://huggingface.co/datasets/ibrahimdaud/raw-food-recognition](https://huggingface.co/datasets/ibrahimdaud/raw-food-recognition)

**Last Updated**: December 2024

