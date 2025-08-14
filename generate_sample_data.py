#!/usr/bin/env python3
"""
Example data generator for testing the viral genomics analysis platform.

This script creates sample data files in the expected format for testing
the dashboard functionality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import random


def create_sample_data(output_dir: Path, num_samples: int = 5) -> None:
    """
    Create sample data files for testing the analysis platform.

    Args:
        output_dir: Directory to create sample data in
        num_samples: Number of samples to generate
    """
    # Create directory structure
    dirs = [
        'consensus', 'coverage', 'depth', 'read_stats',
        'mapping', 'contamination', 'references'
    ]

    for dir_name in dirs:
        (output_dir / dir_name).mkdir(parents=True, exist_ok=True)

    # Generate sample IDs
    sample_ids = [f"sample_{i:03d}" for i in range(1, num_samples + 1)]
    segments = ['L', 'S']
    virus_types = ['LASV', 'HAZV']

    print(f"Generating data for {num_samples} samples...")

    # 1. Mapping statistics
    mapping_data = []
    for sample_id in sample_ids:
        total_reads = random.randint(500000, 2000000)
        mapped_reads = int(total_reads * random.uniform(0.6, 0.9))
        target_mapped = int(mapped_reads * random.uniform(0.7, 0.95))

        mapping_data.append({
            'sample_id': sample_id,
            'total_reads': total_reads,
            'mapped_reads': mapped_reads,
            'target_mapped_reads': target_mapped
        })

    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv(output_dir / 'mapping' / 'mapping_stats.tsv', sep='\t', index=False)

    # 2. Genome recovery data
    recovery_data = []
    for sample_id in sample_ids:
        for segment in segments:
            total_bases = 10000 if segment == 'L' else 3000
            covered_bases = int(total_bases * random.uniform(0.6, 0.98))

            recovery_data.append({
                'sample_id': sample_id,
                'segment': segment,
                'total_bases': total_bases,
                'covered_bases': covered_bases
            })

    recovery_df = pd.DataFrame(recovery_data)
    recovery_df.to_csv(output_dir / 'consensus' / 'genome_recovery.tsv', sep='\t', index=False)

    # 3. ANI comparison data
    ani_data = []
    for i, sample1 in enumerate(sample_ids):
        for j, sample2 in enumerate(sample_ids):
            if i < j:  # Only upper triangle
                ani_value = random.uniform(85.0, 99.5)
                ani_data.append({
                    'sample1': sample1,
                    'sample2': sample2,
                    'ani_value': ani_value
                })

    ani_df = pd.DataFrame(ani_data)
    ani_df.to_csv(output_dir / 'consensus' / 'ani_comparison.tsv', sep='\t', index=False)

    # 4. Coverage depth files
    for sample_id in sample_ids:
        depth_data = []
        for segment in segments:
            genome_length = 10000 if segment == 'L' else 3000
            for pos in range(1, genome_length + 1):
                # Simulate realistic coverage with some variation
                base_depth = random.randint(20, 200)
                # Add some noise and occasional dropouts
                if random.random() < 0.05:  # 5% chance of low coverage
                    depth = random.randint(0, 5)
                else:
                    depth = max(0, int(np.random.normal(base_depth, base_depth * 0.3)))

                depth_data.append({
                    'contig': f'{virus_types[0]}_{segment}',
                    'position': pos,
                    'depth': depth
                })

        depth_df = pd.DataFrame(depth_data)
        depth_df.to_csv(output_dir / 'depth' / f'{sample_id}.depth', sep='\t', index=False, header=False)

    # 5. UMI statistics
    umi_data = []
    for sample_id in sample_ids:
        total_umis = random.randint(10000, 100000)
        target_umis = int(total_umis * random.uniform(0.7, 0.95))

        umi_data.append({
            'sample_id': sample_id,
            'total_umis': total_umis,
            'target_umis': target_umis
        })

    umi_df = pd.DataFrame(umi_data)
    umi_df.to_csv(output_dir / 'read_stats' / 'umi_stats.tsv', sep='\t', index=False)

    # 6. Read counts per segment
    read_counts_data = []
    for sample_id in sample_ids:
        total_reads = mapping_df[mapping_df['sample_id'] == sample_id]['total_reads'].iloc[0]

        for segment in segments:
            segment_reads = int(total_reads * random.uniform(0.1, 0.4))
            read_counts_data.append({
                'sample_id': sample_id,
                'segment': segment,
                'total_reads': total_reads,
                'segment_reads': segment_reads
            })

    read_counts_df = pd.DataFrame(read_counts_data)
    read_counts_df.to_csv(output_dir / 'read_stats' / 'read_counts.tsv', sep='\t', index=False)

    # 7. Contamination data
    for virus_type in virus_types:
        contamination_data = []
        for sample_id in sample_ids:
            total_reads = mapping_df[mapping_df['sample_id'] == sample_id]['total_reads'].iloc[0]
            # Most samples have low contamination
            if random.random() < 0.2:  # 20% chance of higher contamination
                contaminant_reads = int(total_reads * random.uniform(0.01, 0.1))
            else:
                contaminant_reads = int(total_reads * random.uniform(0.0001, 0.01))

            contamination_data.append({
                'sample_id': sample_id,
                'total_reads': total_reads,
                'contaminant_reads': contaminant_reads
            })

        contamination_df = pd.DataFrame(contamination_data)
        contamination_df.to_csv(
            output_dir / 'contamination' / f'{virus_type.lower()}_contamination.tsv',
            sep='\t', index=False
        )

    # 8. Coverage summary
    coverage_summary_data = []
    for sample_id in sample_ids:
        for segment in segments:
            mean_depth = random.uniform(50, 200)
            coverage_percentage = random.uniform(70, 98)

            coverage_summary_data.append({
                'sample_id': sample_id,
                'segment': segment,
                'mean_depth': mean_depth,
                'coverage_percentage': coverage_percentage
            })

    coverage_summary_df = pd.DataFrame(coverage_summary_data)
    coverage_summary_df.to_csv(output_dir / 'coverage' / 'coverage_summary.tsv', sep='\t', index=False)

    # 9. Reference mapping info
    ref_mapping_data = []
    for sample_id in sample_ids:
        for segment in segments:
            ref_mapping_data.append({
                'sample_id': sample_id,
                'segment': segment,
                'reference_id': f'{virus_types[0]}_{segment}_REF',
                'mapping_quality': random.uniform(0.8, 0.99)
            })

    ref_mapping_df = pd.DataFrame(ref_mapping_data)
    ref_mapping_df.to_csv(output_dir / 'references' / 'reference_mapping.tsv', sep='\t', index=False)

    print(f"\nSample data created in: {output_dir}")
    print(f"Generated data for {num_samples} samples")
    print("\nCreated files:")

    # List all created files
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            rel_path = Path(root).relative_to(output_dir) / file
            print(f"  - {rel_path}")


def main():
    """Main function for the data generator."""
    parser = argparse.ArgumentParser(
        description="Generate sample data for viral genomics analysis platform"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="sample_data",
        help="Output directory for sample data (default: sample_data)"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to generate (default: 10)"
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)

    print("Viral Genomics Analysis Platform - Sample Data Generator")
    print("=" * 60)

    create_sample_data(output_path, args.num_samples)

    print("\n" + "=" * 60)
    print("Data generation complete!")
    print(f"\nTo test the dashboard with this data:")
    print(f"  python run_dashboard.py --data-path {output_path}")


if __name__ == "__main__":
    import os
    main()
