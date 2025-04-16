import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt 
import torch
from scipy.ndimage import zoom

from pyronn.ct_reconstruction.geometry.geometry import Geometry
from pyronn.ct_reconstruction.helpers.filters.filters import shepp_logan_3D
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_3d
from pyronn.ct_reconstruction.layers.backprojection_3d import ConeBackProjection3D
from pyronn.ct_reconstruction.layers.projection_3d import ConeProjection3D

def load_dicom_series(dicom_dir, target_spacing=(1.0, 1.0, 1.0), target_shape=(256, 256, 128)):
    """
    Load DICOM files from a directory and generate a 3D volume with uniform spacing and shape.

    Parameters:
    - dicom_dir (str): Path to the directory containing DICOM files.
    - target_spacing (tuple): Desired voxel spacing (x, y, z).
    - target_shape (tuple): Target shape of the output volume.

    Returns:
    - volume_final (numpy.ndarray): Resampled 3D volume.
    - target_spacing (numpy.ndarray): The final spacing after resampling.
    """
    files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
    datasets = []
    
    for f in files:
        ds = pydicom.dcmread(f)
        if hasattr(ds, 'ImagePositionPatient') and hasattr(ds, 'PixelSpacing'):
            datasets.append(ds)
    
    if len(datasets) < 10:
        raise ValueError(f"Not enough valid DICOM files, {len(datasets)} valid files found.")
    
    # Sort by Z-axis position
    datasets.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    pixel_arrays = [ds.pixel_array for ds in datasets]
    
    # Extract spacing information
    pixel_spacing = datasets[0].PixelSpacing  # (X, Y)
    slice_thickness = abs(datasets[1].ImagePositionPatient[2] - datasets[0].ImagePositionPatient[2])  # Z
    
    # Ensure spacing is float64
    spacing = np.array((float(pixel_spacing[0]), float(pixel_spacing[1]), slice_thickness), dtype=np.float64)
    
    # Stack 3D volume
    volume = np.stack(pixel_arrays, axis=-1).astype(np.float64)  # (H, W, Z)
    
    # Resample based on physical spacing to make it uniform
    zoom_factors_spacing = spacing / np.array(target_spacing)
    volume_resampled = zoom(volume, zoom_factors_spacing, order=1)
    
    # Resize to target shape
    zoom_factors_shape = np.array(target_shape) / np.array(volume_resampled.shape)
    volume_final = zoom(volume_resampled, zoom_factors_shape, order=1)
    
    # Transpose dimensions to have Z-axis first
    volume_final = np.transpose(volume_final, (2, 0, 1))
    
    print(f"Original shape: {volume.shape}, Resampled shape: {volume_final.shape}")
    return volume_final, np.array(target_spacing, dtype=np.float64)

def process_and_save_all_dicom(main_dir, output_dir):
    """
    Process all DICOM directories, convert them to 3D volumes, and save them as NumPy arrays.

    Parameters:
    - main_dir (str): Path to the main directory containing DICOM folders.
    - output_dir (str): Directory where processed volumes and spacing files will be saved.

    Returns:
    - results (list): List of dictionaries containing patient ID, volume, and spacing.
    """

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for root, dirs, files in os.walk(main_dir):
        for sub_dir in dirs:
            dicom_dir = os.path.join(root, sub_dir)
            try:
                volume, spacing = load_dicom_series(dicom_dir)
                patient_id = os.path.basename(sub_dir)

                # Save volume and spacing
                volume_file = os.path.join(output_dir, f"{patient_id}_volume.npy")
                spacing_file = os.path.join(output_dir, f"{patient_id}_spacing.npy")
                np.save(volume_file, volume)
                np.save(spacing_file, spacing)

                print(f"Successfully processed and saved: {patient_id}")
                results.append({
                    "patient_id": patient_id,
                    "volume": volume,
                    "spacing": spacing
                })
            except Exception as e:
                print(f"Skipping {dicom_dir}, error: {e}")
    return results


def normalize_volume(volume):
    """
    Normalize the volume to the range [0,1].

    Parameters:
    - volume (numpy.ndarray): Input 3D volume.

    Returns:
    - normalized (numpy.ndarray): Normalized 3D volume.
    """
    min_val = np.min(volume)
    max_val = np.max(volume)
    normalized = (volume - min_val) / (max_val - min_val)
    return normalized



def create_geometry(volume,volume_spacing, detector_shape, detector_spacing, 
                    number_of_projections, angular_range, 
                    sid, sdd, trajectory=circular_trajectory_3d):
    """
    Create and configure a Geometry object.

    Parameters:
    - volume (numpy.ndarray): The 3D volume.
    - volume_spacing (tuple): Spacing of the volume.
    - detector_shape (tuple): Shape of the detector.
    - detector_spacing (tuple): Spacing of the detector.
    - number_of_projections (int): Number of projection angles.
    - angular_range (float): Total angular range for projections.
    - sid (float): Source-to-isocenter distance.
    - sdd (float): Source-to-detector distance.
    - trajectory (function): Function defining the scan trajectory.

    Returns:
    - geometry (Geometry): Configured Geometry object.
    """

    volume_shape = volume.shape

    geometry = Geometry()
    geometry.init_from_parameters(
        volume_shape=volume_shape,
        volume_spacing=volume_spacing,
        detector_shape=detector_shape,
        detector_spacing=detector_spacing,
        number_of_projections=number_of_projections,
        angular_range=angular_range,
        trajectory=trajectory,
        source_isocenter_distance=sid,
        source_detector_distance=sdd
    )
    return geometry

def create_conebeam_projection(volume, geometry, patient_id, output_dir, detector_shape):
    """
    Perform forward cone-beam projection on the given volume.

    Parameters:
    - volume (numpy.ndarray): 3D volume data.
    - geometry (Geometry): CT scan geometry.
    - patient_id (str): Unique identifier for the patient.
    - output_dir (str): Directory to save the projection output.
    - detector_shape (tuple): Shape of the detector.

    Returns:
    - sinogram_np (numpy.ndarray): The computed sinogram.
    """
    
    phantom = torch.tensor(
        np.expand_dims(volume.astype(np.float32), axis=0).copy(), dtype=torch.float32
    ).cuda()
    
    # Perform forward projection
    sinogram = ConeProjection3D().forward(phantom, **geometry)
    sinogram_np = sinogram.cpu().numpy()

    base_name = patient_id
    sinogram_output_path = os.path.join(
        output_dir, f"{base_name}_sinogram_{detector_shape[0]}x{detector_shape[1]}.npy"
    )
    np.save(sinogram_output_path, sinogram_np)
    print(f"Sinogram data saved at: {sinogram_output_path}")

    return sinogram_np


def create_conebeam_backprojection( sinogram_np, geometry, patient_id, output_dir, detector_shape): 
    """
    Perform cone-beam backprojection to reconstruct the 3D volume from the sinogram.

    Parameters:
    - sinogram_np (numpy.ndarray): The sinogram data in NumPy format.
    - geometry (object): The geometry configuration for backprojection.
    - patient_id (str): The unique identifier for the patient.
    - output_dir (str): The directory where the reconstructed volume will be saved.
    - detector_shape (tuple): The shape of the detector (width, height).

    Returns:
    - reconstructed_np (numpy.ndarray): The reconstructed 3D volume.
    """

    sinogram = torch.tensor(sinogram_np, dtype=torch.float32).cuda()

    # FBP
    reco_filter = torch.tensor(
        shepp_logan_3D(
            geometry.detector_shape,
            geometry.detector_spacing,
            geometry.number_of_projections,
        ),
        dtype=torch.float32,
    ).cuda()
    
    sinogram_fft = torch.fft.fft(sinogram, dim=-1, norm="ortho")
    filtered_sinogram = torch.multiply(sinogram_fft, reco_filter)
    filtered_sinogram = torch.fft.ifft(filtered_sinogram, dim=-1, norm="ortho").real

    reconstructed_volume = ConeBackProjection3D().forward(filtered_sinogram.contiguous(), **geometry)
    reconstructed_np = reconstructed_volume.cpu().numpy()

    base_name = patient_id
    reconstruction_output_path = os.path.join(
        output_dir, f"{base_name}_reconstruction_{detector_shape[0]}x{detector_shape[1]}.npy"
    )
    np.save(reconstruction_output_path, reconstructed_np)
    print(f"Reconstructed volume saved at: {reconstruction_output_path}")


    return reconstructed_np


def create_dynamic_rectangle_mask( height, width, center, width_fraction):
    """
    Create a dynamic rectangular mask that applies to the (width, height) dimensions.

    Parameters:
    - height (int): The height of the detector (sinogram's third dimension).
    - width (int): The width of the detector (sinogram's second dimension).
    - center (tuple): The center point (x, y) of the rectangle in pixel coordinates.
    - width_fraction (float): The fraction of width to be retained (0 to 1).

    Returns:
    - mask (numpy.ndarray): A boolean mask with shape (width, height).
    """

    y, x = np.ogrid[:height, :width]  
    cx, _ = center  
    half_width = (width_fraction * width) / 2

    # Apply mask only on width
    mask = (x >= cx - half_width) & (x <= cx + half_width)
    return mask


def create_truncated_proj(full_sinogram, scale_factor, patient_id, output_dir):
    """
    Apply a rectangular mask to the sinogram, truncating data in the width direction.
    The processed sinogram is then restored to batch size 1 and saved.

    Parameters:
    - full_sinogram (numpy.ndarray): The 4D sinogram with shape (batch, views, width, height).
    - scale_factor (float): The fraction of width to be retained (0 to 1).
    - patient_id (str): The unique identifier for the patient.
    - output_dir (str): The directory where the processed sinogram will be saved.

    Returns:
    - sinogram_processed (numpy.ndarray): The processed sinogram with shape (1, views, width, height).
    - mask (numpy.ndarray): The boolean mask applied to the sinogram.
    """

    os.makedirs(output_dir, exist_ok=True)
    
    if full_sinogram.ndim == 4 and full_sinogram.shape[0] == 1:
        sinogram = full_sinogram[0]  # shape: (views, width, height)
    else:
        sinogram = full_sinogram[0]
    
    views, height, width = sinogram.shape
    sinogram_processed = np.zeros_like(sinogram)
    
    center = (width // 2, height // 2)
    
    mask = create_dynamic_rectangle_mask(height, width, center, scale_factor)
    
    for view in range(views):
        sinogram_processed[view] = np.where(mask, sinogram[view], 0)

    sinogram_processed = np.expand_dims(sinogram_processed, axis=0)
    
    base_name = patient_id
    truncated_sinogram_output_path = os.path.join(
        output_dir, f"{base_name}_{scale_factor}_truncated_sinogram_{detector_shape[0]}x{detector_shape[1]}.npy"
    )
    np.save(truncated_sinogram_output_path, sinogram_processed)
    print(f"Sinogram data saved at: {truncated_sinogram_output_path}")

    return sinogram_processed, mask
"""
def zero_pad_sinogram(truncated_sinogram, original_shape):
    """
    Apply zero-padding to a truncated sinogram to restore its original detector shape.

    Parameters:
    - truncated_sinogram (numpy.ndarray): The truncated sinogram with shape (1, views, truncated_width, height).
    - original_shape (tuple): The original detector shape (original_width, original_height).

    Returns:
    - padded_sinogram (numpy.ndarray): The zero-padded sinogram with original dimensions.
    """

    _, views, truncated_width, height = truncated_sinogram.shape
    original_width, original_height = original_shape

    # Compute padding sizes
    pad_left = (original_width - truncated_width) // 2
    pad_right = original_width - truncated_width - pad_left

    padded_sinogram = np.pad(truncated_sinogram, 
                              ((0, 0), (0, 0), (pad_left, pad_right), (0, 0)),
                              mode='constant', constant_values=0)

    return padded_sinogram
"""


"""test"""
test_input = "/home/hpc/iwi5/iwi5241h/CT/output/check/check_input"
test_output_volume = "/home/hpc/iwi5/iwi5241h/CT/output/check/check_volume"
test_output_fullsino = "/home/hpc/iwi5/iwi5241h/CT/output/check/check_full_sinogram"
test_output_fullreco = "/home/hpc/iwi5/iwi5241h/CT/output/check/check_output_rcon"
test_output_trunc_sino = "/home/hpc/iwi5/iwi5241h/CT/output/check/check_truncated_sinogram"
test_output_trunc_recon = "/home/hpc/iwi5/iwi5241h/CT/output/check/check_output_truncate_recon"

sdd = 1000
sid = 500
detector_shape = (750, 700)
detector_spacing = (1, 1)
volume_spacing = (1, 1, 1)
number_of_projections = 180
angular_range = np.pi

scale_factors = [0.75, 0.5, 0.25]  

all_volumes = process_and_save_all_dicom(test_input, test_output_volume)

for patient_data in all_volumes:
    patient_id = patient_data["patient_id"]
    test_volume = patient_data["volume"]

    # Create geometry parameters
    geometry = create_geometry(test_volume, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, sid, sdd, trajectory=circular_trajectory_3d)

    # 1. **Full projection (Sinogram generation)**
    full_sinogram = create_conebeam_projection(test_volume, geometry, patient_id, test_output_fullsino, detector_shape)

    # 2. **Reconstruction from full projection**
    full_reconstruction = create_conebeam_backprojection(full_sinogram, geometry, patient_id, test_output_fullreco, detector_shape)

    # 3. **Iterate over different scale factors for truncation, padding, and reconstruction**
    for scale_factor in scale_factors:
        print(f"Processing scale factor: {scale_factor} for patient {patient_id}")

        # 3.1 **Truncate the projection**
        truncated_sinogram, mask = create_truncated_proj(full_sinogram, scale_factor, patient_id, test_output_trunc_sino)

        # 3.2 **Apply zero-padding to the truncated projection**
        #padded_truncated_sinogram = zero_pad_sinogram(truncated_sinogram, detector_shape)

        # 3.3 **Reconstruct from the truncated projection**
        truncated_reconstruction = create_conebeam_backprojection(truncated_sinogram, geometry, 
                                                                  f"{patient_id}_scale_{int(scale_factor * 100)}",  
                                                                  test_output_trunc_recon, detector_shape)

    print(f"Processing completed for patient: {patient_id}")
