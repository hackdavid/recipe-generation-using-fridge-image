"""
Configuration Loading Module

Handles loading and parsing YAML configuration files.
"""

import os
import yaml


def load_config(config_path):
    """
    Load and parse configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing flattened configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError("Config file is empty or invalid")
    
    # Flatten nested config structure
    cfg = {}
    
    # Data configuration
    data_cfg = config.get('data', {})
    cfg['data_source'] = data_cfg.get('data_source', 'huggingface')  # 'huggingface' or 'folder'
    cfg['dataset_name'] = data_cfg.get('dataset_name', None)  # HuggingFace dataset name
    cfg['train_split'] = data_cfg.get('train_split', 'train')
    cfg['val_split'] = data_cfg.get('val_split', 'validation')
    cfg['use_custom_split'] = data_cfg.get('use_custom_split', False)  # Create 80/20 split in streaming mode
    cfg['train_ratio'] = data_cfg.get('train_ratio', 0.8)  # Ratio for train split (default: 0.8 = 80%)
    cfg['data_dir'] = data_cfg.get('data_dir', './data')  # For folder-based datasets
    cfg['image_size'] = data_cfg.get('image_size', 224)
    cfg['num_workers'] = data_cfg.get('num_workers', 4)
    
    # Model configuration
    model_cfg = config.get('model', {})
    cfg['model'] = model_cfg.get('architecture', 'resnet50')
    cfg['num_classes'] = model_cfg.get('num_classes', None)
    cfg['pretrained'] = model_cfg.get('pretrained', True)
    cfg['se_reduction'] = model_cfg.get('se_reduction', 16)
    
    # Training configuration
    train_cfg = config.get('training', {})
    cfg['epochs'] = train_cfg.get('epochs', 50)
    cfg['batch_size'] = train_cfg.get('batch_size', 32)
    cfg['lr'] = train_cfg.get('learning_rate', 0.001)
    cfg['weight_decay'] = train_cfg.get('weight_decay', 1e-4)
    cfg['optimizer'] = train_cfg.get('optimizer', 'Adam')
    cfg['scheduler'] = train_cfg.get('scheduler', {})
    cfg['sgd'] = train_cfg.get('sgd', {})
    
    # Checkpoint configuration
    checkpoint_cfg = config.get('checkpoint', {})
    cfg['save_dir'] = checkpoint_cfg.get('save_dir', './checkpoints')
    cfg['resume'] = checkpoint_cfg.get('resume', None)
    
    # Wandb configuration
    wandb_cfg = config.get('wandb', {})
    cfg['use_wandb'] = wandb_cfg.get('use', False)
    cfg['wandb_api_key'] = wandb_cfg.get('api_key', None)
    cfg['wandb_project'] = wandb_cfg.get('project', 'ingredient-recognition')
    cfg['wandb_entity'] = wandb_cfg.get('entity', None)
    cfg['wandb_run_name'] = wandb_cfg.get('run_name', None)
    cfg['wandb_tags'] = wandb_cfg.get('tags', [])
    
    # Validate required fields
    if cfg['data_source'] == 'huggingface':
        if not cfg['dataset_name']:
            raise ValueError("dataset_name must be specified in config file when using HuggingFace dataset")
    elif cfg['data_source'] == 'folder':
        if not cfg['data_dir']:
            raise ValueError("data_dir must be specified in config file when using folder-based dataset")
    else:
        raise ValueError(f"Invalid data_source: {cfg['data_source']}. Must be 'huggingface' or 'folder'")
    
    if not cfg['model']:
        raise ValueError("model architecture must be specified in config file")
    
    if not cfg['num_classes']:
        raise ValueError("num_classes must be specified in config file")
    
    return cfg

