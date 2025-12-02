# download lol low light dataset

import os
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm


def download_lol_dataset(output_dir='data'):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        return False
    
    print("Downloading LOL dataset from Hugging Face...")
    
    try:
        dataset = load_dataset("zhengli97/LOL-dataset")
        output_path = Path(output_dir)
        
        # create directories
        train_low_dir = output_path / 'train' / 'low'
        train_high_dir = output_path / 'train' / 'high'
        test_low_dir = output_path / 'test' / 'low'
        test_high_dir = output_path / 'test' / 'high'
        eval_low_dir = output_path / 'eval' / 'low'
        eval_high_dir = output_path / 'eval' / 'high'
        
        for dir_path in [train_low_dir, train_high_dir, test_low_dir, 
                        test_high_dir, eval_low_dir, eval_high_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save training images
        print("Saving training images...")
        for idx, item in enumerate(tqdm(dataset['train'])):
            item['low'].save(train_low_dir / f'train_{idx:04d}.png')
            item['high'].save(train_high_dir / f'train_{idx:04d}.png')
        
        # Save test images
        print("Saving test images...")
        for idx, item in enumerate(tqdm(dataset['test'])):
            item['low'].save(test_low_dir / f'test_{idx:04d}.png')
            item['high'].save(test_high_dir / f'test_{idx:04d}.png')
        
        # copy test to eval
        print("Creating evaluation split...")
        for f in test_low_dir.glob('*.png'):
            shutil.copy(f, eval_low_dir / f.name)
        for f in test_high_dir.glob('*.png'):
            shutil.copy(f, eval_high_dir / f.name)
        
        print(f"\nDataset downloaded to {output_path.absolute()}")
        print(f"  Train: {len(list(train_low_dir.glob('*.png')))} pairs")
        print(f"  Test: {len(list(test_low_dir.glob('*.png')))} pairs")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download LOL dataset')
    parser.add_argument('--output_dir', type=str, default='data')
    args = parser.parse_args()
    
    success = download_lol_dataset(args.output_dir)
    
    if success:
        print("\nYou can now train the model with:")
        print(f"  python train.py --train_low_dir {args.output_dir}/train/low --train_high_dir {args.output_dir}/train/high")


if __name__ == '__main__':
    main()
