"""
Utility module for loading building metadata from Buildings-900K parquet files.

The metadata files contain EnergyPlus building model attributes including:
- Building geometry (square footage, etc.)
- HVAC system parameters (COP, setpoints, etc.)
- Building envelope properties
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Optional
import os


def load_building_metadata(
    building_id: str,
    dataset_path: Optional[Path] = None,
    building_type_and_year: Optional[str] = None,
    census_region: Optional[str] = None,
    puma_id: Optional[str] = None
) -> Dict:
    """Load metadata for a specific building from Buildings-900K metadata parquet.
    
    Args:
        building_id: Building identifier
        dataset_path: Path to Buildings-900K dataset. If None, uses BUILDINGS_BENCH env var.
        building_type_and_year: Building type and year (e.g., 'resstock_amy2018_release_1').
                                If None, will search common options.
        census_region: Census region (e.g., 'by_puma_northeast'). If None, will search.
        puma_id: PUMA ID. If None, will search.
    
    Returns:
        Dictionary containing building metadata. Keys include:
            - 'in.sqft': Building square footage
            - 'stat.average_dx_cooling_cop': Average cooling COP
            - 'in.tstat_clg_sp_f': Cooling setpoint in Fahrenheit
            - And other EnergyPlus attributes
    """
    if dataset_path is None:
        dataset_path = Path(os.environ.get('BUILDINGS_BENCH', ''))
    
    metadata_base = (
        dataset_path / 'Buildings-900K' / 'end-use-load-profiles-for-us-building-stock' / '2021'
    )
    
    # Common building types and years
    if building_type_and_year is None:
        building_types = [
            'comstock_tmy3_release_1',
            'resstock_tmy3_release_1',
            'comstock_amy2018_release_1',
            'resstock_amy2018_release_1'
        ]
    else:
        building_types = [building_type_and_year]
    
    # Common census regions
    if census_region is None:
        census_regions = ['by_puma_midwest', 'by_puma_south', 'by_puma_northeast', 'by_puma_west']
    else:
        census_regions = [census_region]
    
    # Try to find metadata file
    for bty in building_types:
        for cr in census_regions:
            metadata_path = metadata_base / bty / 'metadata' / 'metadata.parquet'
            
            if not metadata_path.exists():
                continue
            
            try:
                # Load metadata parquet
                metadata_df = pq.read_table(metadata_path).to_pandas()
                
                # Check if building_id is in the index or a column
                if building_id in metadata_df.index:
                    row = metadata_df.loc[building_id]
                elif 'building_id' in metadata_df.columns:
                    row = metadata_df[metadata_df['building_id'] == building_id].iloc[0]
                elif building_id in metadata_df.columns:
                    # Building ID might be a column name
                    row = metadata_df[building_id]
                else:
                    # Try to find by matching any column
                    matching_cols = [col for col in metadata_df.columns if building_id in str(col)]
                    if matching_cols:
                        row = metadata_df[matching_cols[0]]
                    else:
                        continue
                
                # Convert to dictionary
                if isinstance(row, pd.Series):
                    metadata = row.to_dict()
                else:
                    # If it's a single value, create a dict with the column name
                    metadata = {building_id: row}
                
                return metadata
                
            except (KeyError, IndexError, ValueError):
                continue
    
    # If not found, return default metadata
    print(f"Warning: Could not find metadata for building {building_id}. Using defaults.")
    return {
        'in.sqft': 2000.0,
        'stat.average_dx_cooling_cop': 3.0,
        'in.tstat_clg_sp_f': 72.0
    }


def load_metadata_for_buildings(
    building_ids: list,
    dataset_path: Optional[Path] = None
) -> Dict[str, Dict]:
    """Load metadata for multiple buildings.
    
    Args:
        building_ids: List of building identifiers
        dataset_path: Path to Buildings-900K dataset. If None, uses BUILDINGS_BENCH env var.
    
    Returns:
        Dictionary mapping building_id to metadata dictionary
    """
    metadata_dict = {}
    
    for building_id in building_ids:
        metadata_dict[building_id] = load_building_metadata(building_id, dataset_path)
    
    return metadata_dict


def get_default_metadata() -> Dict:
    """Get default metadata values for buildings without available metadata.
    
    Returns:
        Dictionary with default building parameters
    """
    return {
        'in.sqft': 2000.0,
        'stat.average_dx_cooling_cop': 3.0,
        'in.tstat_clg_sp_f': 72.0
    }

