"""
Copy from an input folder all dicom files to an output folder. In hte output folder the files
will be organized in a hierarchical structure based on the patient ID , study UID, and series UID.
"""
import os
import shutil
from pathlib import Path

from loguru import logger

from .dicom_utils import dcm_generator

from concurrent.futures import ThreadPoolExecutor
from pydicom import dcmread


def copy_and_organize(
    input_folder: Path,
    output_folder: Path,
    restructure: bool = True,
):
    """Copy from an input folder all DICOM files to an output folder. If `restructure` is True, the files
    in the output folder will be organized in a hierarchical structure based on the patient ID, study UID,
    and series UID.
    """
    cnt = 0
    dirs: dict[str, int] = {}
    # XXX: Should we order by InstanceNumber ??
    for dcm_info in dcm_generator(input_folder):
        current_output_folder = (
            (
                output_folder
                / dcm_info.patient_id
                / dcm_info.study_uid
                / dcm_info.series_uid
            )
            if restructure
            else output_folder / dcm_info.path.parent.relative_to(input_folder)
        )
        if current_output_folder not in dirs:
            # Since we walk the input directory in top down manner, we are sure that when we visit
            # a directory its parent directory has already been visited and created. So
            # parents=True, exist_ok=True are not needed but ..ok :-)
            current_output_folder.mkdir(parents=True, exist_ok=True)
            dirs[current_output_folder] = 1
        index = dirs[current_output_folder]
        dirs[current_output_folder] += 1
        output_file = (
            current_output_folder / f"{index:05d}.dcm"
            if restructure
            else current_output_folder / dcm_info.path.name
        )
        shutil.copy(dcm_info.path, output_file)
        cnt += 1
    msg = "Copied and organized hierarchically" if restructure else "Copied"
    logger.info(f"{msg} {cnt} DICOM files")

def process_single_file(file_path: str, input_folder: Path, output_folder: Path, restructure: bool, plain_dir: bool = False, file_number: int | None = None):
    """Worker function to process and copy a single DICOM file."""
    try:
        # Read metadata
        ds = dcmread(file_path, stop_before_pixels=True)
        
        # Determine destination
        if plain_dir:
            dest_dir = output_folder
            dest_file = dest_dir / f"{file_number}.dcm"
        elif restructure:
            dest_dir = (
                output_folder / 
                str(ds.PatientID) / 
                str(ds.StudyInstanceUID) / 
                str(ds.SeriesInstanceUID)
            )
            # Use InstanceNumber for filename or a hash to avoid collisions in parallel
            dest_file = dest_dir / f"{int(ds.InstanceNumber):05d}.dcm"
        else:
            rel_path = Path(file_path).parent.relative_to(input_folder)
            dest_dir = output_folder / rel_path
            dest_file = dest_dir / Path(file_path).name

        # Create directory (exist_ok=True is critical for multithreading)
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy(file_path, dest_file)
        return True
    except Exception as e:
        # Log or skip files that aren't valid DICOMs
        logger.error(f"Failed to copy {file_path} to output folder with error: {e}")
        return False

def copy_and_organize_parallel(
    input_folder: Path,
    output_folder: Path,
    restructure: bool = True,
    threads: int = 10,
    plain_dir: bool = False,
):
    # 1. Collect all file paths first
    all_files = []
    for root, _, files in os.walk(os.fspath(input_folder)):
        for file in files:
            all_files.append(os.path.join(root, file))

    # 2. Use a ThreadPoolExecutor to run the I/O tasks
    cnt = 0
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Map the worker function across all files
        results = list(executor.map(
            lambda f, idx: process_single_file(f, input_folder, output_folder, restructure, plain_dir=plain_dir, file_number=idx+1),
            all_files,
            range(len(all_files))
        ))
        cnt = sum(1 for r in results if r)

    msg = "Copied and organized hierarchically" if restructure else "Copied"
    print(f"{msg} {cnt} DICOM files")