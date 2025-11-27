"""
Download LOL (Low-Light) dataset from Hugging Face or Kaggle

The LOL dataset is the standard dataset used in the Deep Retinex paper.
This script downloads it from Hugging Face or Kaggle and organizes it properly.

Hugging Face: https://huggingface.co/datasets/zhengli97/LOL-dataset
Kaggle: https://www.kaggle.com/datasets/tanhyml/lol-v2-dataset
"""

import os
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm


def download_lol_dataset(output_dir='data'):
    """
    Download LOL dataset using Hugging Face datasets library
    
    The LOL dataset contains:
    - Training: 485 pairs (low-light + normal-light)
    - Testing: 15 pairs
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not found.")
        print("Please install it with: pip install datasets")
        return False
    
    print("Downloading LOL dataset from Hugging Face...")
    print("This may take a few minutes...")
    
    try:
        # Download dataset
        dataset = load_dataset("zhengli97/LOL-dataset")
        
        # Create directory structure
        output_path = Path(output_dir)
        
        # Training data
        train_low_dir = output_path / 'train' / 'low'
        train_high_dir = output_path / 'train' / 'high'
        
        # Testing data
        test_low_dir = output_path / 'test' / 'low'
        test_high_dir = output_path / 'test' / 'high'
        
        # Evaluation data (same as test for LOL)
        eval_low_dir = output_path / 'eval' / 'low'
        eval_high_dir = output_path / 'eval' / 'high'
        
        # Create directories
        for dir_path in [train_low_dir, train_high_dir, test_low_dir, test_high_dir, 
                        eval_low_dir, eval_high_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("\nOrganizing training data...")
        # Save training images
        train_data = dataset['train']
        for idx, item in enumerate(tqdm(train_data)):
            # Low-light image
            low_img = item['low']
            low_img.save(train_low_dir / f'train_{idx:04d}.png')
            
            # High-light (normal) image
            high_img = item['high']
            high_img.save(train_high_dir / f'train_{idx:04d}.png')
        
        print("\nOrganizing test data...")
        # Save test images
        test_data = dataset['test']
        for idx, item in enumerate(tqdm(test_data)):
            # Low-light image
            low_img = item['low']
            low_img.save(test_low_dir / f'test_{idx:04d}.png')
            
            # High-light (normal) image
            high_img = item['high']
            high_img.save(test_high_dir / f'test_{idx:04d}.png')
        
        # Copy test data to eval (for validation during training)
        print("\nCreating evaluation split...")
        for low_file in test_low_dir.glob('*.png'):
            shutil.copy(low_file, eval_low_dir / low_file.name)
        
        for high_file in test_high_dir.glob('*.png'):
            shutil.copy(high_file, eval_high_dir / high_file.name)
        
        print("\n" + "=" * 50)
        print("Dataset downloaded successfully!")
        print(f"Location: {output_path.absolute()}")
        print("\nDataset structure:")
        print(f"  Train: {len(list(train_low_dir.glob('*.png')))} image pairs")
        print(f"  Test: {len(list(test_low_dir.glob('*.png')))} image pairs")
        print(f"  Eval: {len(list(eval_low_dir.glob('*.png')))} image pairs")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return False


def download_lol_v2_dataset_kaggle(output_dir='data_lol_v2'):
    """
    Download LOL-v2 dataset from Kaggle
    
    Requirements:
    - pip install kagglehub
    - Kaggle API credentials configured (kaggle.json)
    
    LOL-v2 has two subsets:
    - Real: Real captured images
    - Synthetic: Synthetically generated images
    """
    try:
        import kagglehub
    except ImportError:
        print("Error: 'kagglehub' library not found.")
        print("Please install it with: pip install kagglehub")
        return False
    
    print("Downloading LOL-v2 dataset from Kaggle...")
    print("Note: This requires Kaggle API credentials configured.")
    print("See: https://github.com/Kaggle/kaggle-api#api-credentials")
    print()
    
    try:
        # Download dataset from Kaggle
        print("Downloading from Kaggle (this may take a while)...")
        kaggle_path = kagglehub.dataset_download("tanhyml/lol-v2-dataset")
        print(f"Dataset downloaded to: {kaggle_path}")
        
        # Organize the downloaded files
        output_path = Path(output_dir)
        kaggle_path = Path(kaggle_path)
        
        print("\nOrganizing dataset structure...")
        
        # LOL-v2 structure from Kaggle
        # The dataset is inside a LOL-v2 subdirectory
        lol_v2_path = kaggle_path / 'LOL-v2'
        if not lol_v2_path.exists():
            lol_v2_path = kaggle_path  # Fallback if structure is different
        
        # Create directory structure
        for subset in ['real', 'synthetic']:
            for split in ['train', 'test', 'eval']:
                for img_type in ['low', 'high']:
                    dir_path = output_path / subset / split / img_type
                    dir_path.mkdir(parents=True, exist_ok=True)
        
        # Copy files from Kaggle structure to our structure
        if (lol_v2_path / 'Real_captured').exists():
            print("Processing Real subset...")
            organize_kaggle_lol_v2(lol_v2_path / 'Real_captured', output_path / 'real')
        
        if (lol_v2_path / 'Synthetic').exists():
            print("Processing Synthetic subset...")
            organize_kaggle_lol_v2(lol_v2_path / 'Synthetic', output_path / 'synthetic')
        
        print("\n" + "=" * 50)
        print("LOL-v2 dataset downloaded and organized successfully!")
        print(f"Location: {output_path.absolute()}")
        print("\nDataset structure:")
        for subset in ['real', 'synthetic']:
            subset_path = output_path / subset
            if subset_path.exists():
                train_low = len(list((subset_path / 'train' / 'low').glob('*')))
                test_low = len(list((subset_path / 'test' / 'low').glob('*')))
                print(f"  {subset.capitalize()}: {train_low} train pairs, {test_low} test pairs")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"Error downloading LOL-v2 dataset from Kaggle: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Install kagglehub: pip install kagglehub")
        print("2. Set up Kaggle API: https://github.com/Kaggle/kaggle-api#api-credentials")
        print("3. Ensure you have accepted the dataset terms on Kaggle website")
        return False


def organize_kaggle_lol_v2(kaggle_subset_path, output_subset_path):
    """
    Organize LOL-v2 files from Kaggle structure to our structure
    
    Kaggle structure:
        Train/Low/, Train/Normal/
        Test/Low/, Test/Normal/
    
    Our structure:
        train/low/, train/high/
        test/low/, test/high/
        eval/low/, eval/high/
    """
    # Map Kaggle names to our names
    split_map = {'Train': 'train', 'Test': 'test'}
    type_map = {'Low': 'low', 'Normal': 'high'}
    
    for kaggle_split, our_split in split_map.items():
        kaggle_split_path = kaggle_subset_path / kaggle_split
        if not kaggle_split_path.exists():
            continue
        
        for kaggle_type, our_type in type_map.items():
            kaggle_type_path = kaggle_split_path / kaggle_type
            if not kaggle_type_path.exists():
                continue
            
            # Copy to train/test
            our_type_path = output_subset_path / our_split / our_type
            our_type_path.mkdir(parents=True, exist_ok=True)
            
            # Copy all image files
            for img_file in tqdm(list(kaggle_type_path.glob('*')), 
                                desc=f"Copying {kaggle_split}/{kaggle_type}"):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    shutil.copy2(img_file, our_type_path / img_file.name)
        
        # Also copy test to eval
        if our_split == 'test':
            for our_type in ['low', 'high']:
                src_path = output_subset_path / 'test' / our_type
                dst_path = output_subset_path / 'eval' / our_type
                dst_path.mkdir(parents=True, exist_ok=True)
                
                for img_file in src_path.glob('*'):
                    if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        shutil.copy2(img_file, dst_path / img_file.name)


def download_lol_v2_dataset_huggingface(output_dir='data_lol_v2'):
    """
    Download LOL-v2 dataset from Hugging Face (may not be available)
    
    LOL-v2 has two subsets:
    - Real: Real captured images
    - Synthetic: Synthetically generated images
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not found.")
        print("Please install it with: pip install datasets")
        return False
    
    print("Downloading LOL-v2 dataset from Hugging Face...")
    print("Note: LOL-v2 may not be available on Hugging Face.")
    print("      Consider using --source kaggle instead.")
    print()
    
    try:
        # Download both subsets
        real_dataset = load_dataset("zhengli97/LOL-v2-dataset", name="real")
        synthetic_dataset = load_dataset("zhengli97/LOL-v2-dataset", name="synthetic")
        
        output_path = Path(output_dir)
        
        # Process Real subset
        print("\nProcessing LOL-v2 Real subset...")
        real_dir = output_path / 'real'
        process_lol_v2_subset(real_dataset, real_dir)
        
        # Process Synthetic subset
        print("\nProcessing LOL-v2 Synthetic subset...")
        synthetic_dir = output_path / 'synthetic'
        process_lol_v2_subset(synthetic_dataset, synthetic_dir)
        
        print("\n" + "=" * 50)
        print("LOL-v2 dataset downloaded successfully!")
        print(f"Location: {output_path.absolute()}")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"Error downloading LOL-v2 dataset from Hugging Face: {str(e)}")
        print("\nLOL-v2 is not available on Hugging Face.")
        print("Please use Kaggle instead:")
        print(f"  python download_dataset.py --dataset lol-v2 --source kaggle --output_dir {output_dir}")
        return False


def process_lol_v2_subset(dataset, output_dir):
    """Process LOL-v2 subset and save images"""
    train_low_dir = output_dir / 'train' / 'low'
    train_high_dir = output_dir / 'train' / 'high'
    test_low_dir = output_dir / 'test' / 'low'
    test_high_dir = output_dir / 'test' / 'high'
    eval_low_dir = output_dir / 'eval' / 'low'
    eval_high_dir = output_dir / 'eval' / 'high'
    
    for dir_path in [train_low_dir, train_high_dir, test_low_dir, test_high_dir,
                    eval_low_dir, eval_high_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Process training data
    if 'train' in dataset:
        train_data = dataset['train']
        for idx, item in enumerate(tqdm(train_data, desc="Train")):
            low_img = item['low']
            high_img = item['high']
            low_img.save(train_low_dir / f'train_{idx:04d}.png')
            high_img.save(train_high_dir / f'train_{idx:04d}.png')
    
    # Process test data
    if 'test' in dataset:
        test_data = dataset['test']
        for idx, item in enumerate(tqdm(test_data, desc="Test")):
            low_img = item['low']
            high_img = item['high']
            low_img.save(test_low_dir / f'test_{idx:04d}.png')
            high_img.save(test_high_dir / f'test_{idx:04d}.png')
            
            # Also copy to eval
            low_img.save(eval_low_dir / f'test_{idx:04d}.png')
            high_img.save(eval_high_dir / f'test_{idx:04d}.png')


def main():
    parser = argparse.ArgumentParser(
        description='Download LOL dataset from Hugging Face or Kaggle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download LOL from Hugging Face
  python download_dataset.py --dataset lol --output_dir data

  # Download LOL-v2 from Kaggle (recommended)
  python download_dataset.py --dataset lol-v2 --source kaggle --output_dir data_lol_v2

  # Download LOL-v2 from Hugging Face (may not work)
  python download_dataset.py --dataset lol-v2 --source huggingface --output_dir data_lol_v2

Note: For Kaggle downloads, you need:
  1. pip install kagglehub
  2. Kaggle API credentials: https://github.com/Kaggle/kaggle-api#api-credentials
        """
    )
    parser.add_argument('--dataset', type=str, default='lol',
                        choices=['lol', 'lol-v2'],
                        help='Which dataset to download (default: lol)')
    parser.add_argument('--source', type=str, default='auto',
                        choices=['huggingface', 'kaggle', 'auto'],
                        help='Download source (default: auto - uses best available)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for dataset (default: data)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("LOL Dataset Downloader")
    print("=" * 50)
    print(f"\nDataset: {args.dataset.upper()}")
    print(f"Source: {args.source}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    success = False
    
    if args.dataset == 'lol':
        # LOL is available on Hugging Face
        success = download_lol_dataset(args.output_dir)
    
    elif args.dataset == 'lol-v2':
        # LOL-v2: Choose source
        if args.source == 'kaggle' or args.source == 'auto':
            print("Using Kaggle as source (recommended for LOL-v2)")
            success = download_lol_v2_dataset_kaggle(args.output_dir)
        elif args.source == 'huggingface':
            print("Using Hugging Face as source")
            success = download_lol_v2_dataset_huggingface(args.output_dir)
            
            # If HF fails and auto mode, try Kaggle
            if not success and args.source == 'auto':
                print("\nHugging Face failed. Trying Kaggle...")
                success = download_lol_v2_dataset_kaggle(args.output_dir)
    
    if success:
        print("\n" + "=" * 50)
        print("SUCCESS! Dataset ready for training.")
        print("=" * 50)
        print("\nYou can now train the model with:")
        
        if args.dataset == 'lol':
            print(f"  python train.py --train_low_dir {args.output_dir}/train/low --train_high_dir {args.output_dir}/train/high")
        elif args.dataset == 'lol-v2':
            print(f"  # For Real subset:")
            print(f"  python train.py --train_low_dir {args.output_dir}/real/train/low --train_high_dir {args.output_dir}/real/train/high")
            print(f"\n  # For Synthetic subset:")
            print(f"  python train.py --train_low_dir {args.output_dir}/synthetic/train/low --train_high_dir {args.output_dir}/synthetic/train/high")
        
        print("\nOr test on the dataset:")
        if args.dataset == 'lol':
            print(f"  python test.py --checkpoint checkpoints/retinexnet_best.pth --input_dir {args.output_dir}/test/low --output_dir results/test")
        elif args.dataset == 'lol-v2':
            print(f"  python test.py --checkpoint checkpoints/retinexnet_best.pth --input_dir {args.output_dir}/real/test/low --output_dir results/test_real")
    else:
        print("\n" + "=" * 50)
        print("Dataset download failed.")
        print("=" * 50)
        print("\nTroubleshooting:")
        if args.dataset == 'lol-v2':
            print("1. Install kagglehub: pip install kagglehub")
            print("2. Set up Kaggle API credentials:")
            print("   - Go to https://www.kaggle.com/settings/account")
            print("   - Click 'Create New API Token'")
            print("   - Place kaggle.json in ~/.kaggle/")
            print("3. Try again with: python download_dataset.py --dataset lol-v2 --source kaggle")
        else:
            print("1. Check your internet connection")
            print("2. Install datasets: pip install datasets")
            print("3. Try again")


if __name__ == '__main__':
    main()
