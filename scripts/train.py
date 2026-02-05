#!/usr/bin/env python
"""Train all FPL prediction models."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import FPLPipeline


def main():
    print("=" * 60)
    print("FPL MODEL TRAINING")
    print("=" * 60)
    
    pipeline = FPLPipeline(data_dir='data')
    pipeline.load_data()
    pipeline.compute_features()
    pipeline.train()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Show feature importance for each model
    for name, model in pipeline.models.items():
        print(f"\n{name.upper()} - Top 5 Features:")
        fi = model.feature_importance()
        for _, row in fi.head(5).iterrows():
            print(f"  {row['feature']:30s} {row['importance']:.4f}")


if __name__ == '__main__':
    main()
