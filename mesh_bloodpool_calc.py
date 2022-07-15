import glob
import os

import numpy as np
import pandas as pd
import pyvista as pv
import vtkmodules.all as vtk
from natsort import natsorted


def extract_closest_point_region(polydata, point=(0, 0, 0)):
    """Function that finds the closest surface given a seed coordinate

    Args:
        polydata (object): Surface mesh
        point (tuple): Seed coordinate for finding subsequent closest enclosed region

    Returns:
        connect.GetOutput(): Extracted surface mesh.
    """
    surfer = vtk.vtkDataSetSurfaceFilter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        surfer.SetInputData(polydata)
    else:
        surfer.SetInput(polydata)
    surfer.Update()
    cleaner = vtk.vtkCleanPolyData()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        cleaner.SetInputData(surfer.GetOutput())
    else:
        cleaner.SetInput(surfer.GetOutput())
    cleaner.Update()
    connect = vtk.vtkPolyDataConnectivityFilter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        connect.SetInputData(cleaner.GetOutput())
    else:
        connect.SetInput(cleaner.GetOutput())
    connect.SetExtractionModeToClosestPointRegion()
    connect.SetClosestPoint(point)
    connect.Update()
    return connect.GetOutput()


def center_of_mass(surface):
    """ Get center of mass of Polydata """
    center_filter = vtk.vtkCenterOfMass()
    center_filter.SetInputData(surface)
    center_filter.SetUseScalarsAsWeights(False)
    center_filter.Update()
    center = center_filter.GetCenter()
    return center


def refine_polydata(polydata, iterations=30, set_passband=False):
    """Adjusts mesh point positions using a windowed sinc function interpolation kernel.

        The effect is to "relax" the mesh, making the cells better shaped and the vertices more evenly distributed.
    Args:
        polydata (object): Input surface mesh.
        iterations (int): Number of iterations to perform refinement for.
        set_passband (bool): Flag of whether to apply a passband filter at 0.05 threshold.

    Returns:
        refiner.GetOutput(): Refined mesh
    """

    refiner = vtk.vtkWindowedSincPolyDataFilter()
    refiner.SetInputData(polydata)
    refiner.SetNumberOfIterations(iterations)
    refiner.NonManifoldSmoothingOn()
    if set_passband:
        refiner.SetPassBand(0.05)
    refiner.NormalizeCoordinatesOff()
    refiner.GenerateErrorScalarsOff()
    refiner.Update()
    return refiner.GetOutput()


def cell_threshold(polydata, arrayname, start=0, end=1):
    """Function to extract desired labels from a multi label mesh.

    Args:
        polydata (object): Input surface mesh.
        arrayname (str): How the scalars are defined in the mesh data. (Same as label_tag in get_endocardial_surface)
        start (int): Starting threshold range for labels.
        end (int): Ending threshold range for labels.

    Returns:
        surfer.GetOutput(): Extracted mesh.
    """
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, arrayname)
    threshold.SetLowerThreshold(start)
    threshold.SetUpperThreshold(end)
    threshold.Update()
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputConnection(threshold.GetOutputPort())
    surfer.Update()
    return surfer.GetOutput()


def volume_calc(triangle_in):
    """Performs volume calculation given 3D coordinates as proposed in 'Efficient feature extraction for 2D/3D objects
    in mesh representation' by Zhang and Chen 2001

    Args:
        triangle_in (np.array):

    Returns:
        v_i (float): volume for individual triangle
    """
    triangle_in = np.squeeze(triangle_in)
    x = triangle_in[0, :]
    y = triangle_in[1, :]
    z = triangle_in[2, :]

    v_i = (1 / 6.) * (-x[2] * y[1] * z[0] +
                      x[1] * y[2] * z[0] +
                      x[2] * y[0] * z[1] -
                      x[0] * y[2] * z[1] -
                      x[1] * y[0] * z[2] +
                      x[0] * y[1] * z[2])
    return v_i


def calc_bloodpool_vol(in_mesh):
    """Calculates the volume of a given surface mesh

    Args:
        in_mesh (object): Extracted surface to calculate a volume for.

    Returns:

    """
    number_of_cells = in_mesh.GetNumberOfCells()

    cell_ids = vtk.vtkIdList()
    total_volume = []
    for cellIndex in range(number_of_cells):  # for every cell
        in_mesh.GetCellPoints(cellIndex, cell_ids)
        triangle = []
        for i in range(0, cell_ids.GetNumberOfIds()):  # for every point of the given cell
            coord = in_mesh.GetPoint(cell_ids.GetId(i))
            triangle.append(coord)

        triangle_volume = volume_calc(triangle)
        total_volume.append(triangle_volume)
    return abs(np.sum(total_volume))


def save_volumes2excel(save_path, volume_data, calculated_labels, case_paths):
    """Save calculated volumes to an Excel file (xlsx format specifically)

    Args:
        save_path (str): Path to save file to.
        volume_data (list): Calculated volumes of dataset.
        calculated_labels (list): Labels of the anatomy that was used in calculations.
        case_paths (list): All file paths of used meshes, used for clear differentiation in file.

    Returns:
        df (object): Dataframe containing all information that will be saved to file.
    """
    column_labels = ['volume_' + str(label) for label in calculated_labels]
    row_labels = [case_path.split(os.sep)[-1].split('.')[0] for case_path in case_paths]

    df = pd.DataFrame(np.squeeze(volume_data), index=row_labels, columns=column_labels)
    df.to_csv(save_path, sep=',', index=True)
    return df


def get_endocardial_surface(in_mesh, label_tag, label_number):
    """Function to find the inner (endocardial) surface of the cardiac mesh for each desired chamber.

    Args:
        in_mesh (object): Loaded Pyvista mesh object.
        label_tag (str): How the scalars are defined in the mesh data.
        label_number (int): Chamber label of the cardiac mesh.

    Returns:
        endo_surface (object): Inner surface of the desired chamber to calculate the volume of.
    """
    mesh_surface = in_mesh.extract_surface()
    extracted_chamber = cell_threshold(mesh_surface,
                                       label_tag,
                                       label_number,
                                       label_number)
    com_chamber = center_of_mass(extracted_chamber)
    endo_surface = pv.wrap(extract_closest_point_region(mesh_surface, com_chamber))
    if refine_mesh:
        endo_surface = refine_polydata(endo_surface)
    return endo_surface


def get_bloodpool_volume(in_path, selected_labels, label_tag='elemTag',
                         plot_chamber_volumes=False):  # use 'ID' for marinas cases
    """Function to extract the bloodpool volume(s) of a given mesh

    Args:
        in_path (str): Path of vtk mesh to be loaded.
        selected_labels (list): Chamber labels of the cardiac mesh to calculate a volume for.
        label_tag (str): How the scalars are defined in the mesh data.
        plot_chamber_volumes (bool): Whether to plot each of the extracted endocardial surfaces in an external window.

    Returns:
        calculated_volumes (list): Calculated volumes for each of the desired labels.

    """
    mesh = pv.read(in_path)
    calculated_volumes = []
    for label in selected_labels:
        endocardium = get_endocardial_surface(in_mesh=mesh, label_tag=label_tag, label_number=label)

        if plot_chamber_volumes:
            plotter = pv.Plotter()
            plotter.add_mesh(endocardium, scalars="elemTag", smooth_shading=True, show_scalar_bar=False)
            plotter.show()

        bloodpool_volume = calc_bloodpool_vol(endocardium) / 1e3
        calculated_volumes.append(bloodpool_volume)
    return calculated_volumes


if __name__ == "__main__":
    '''
    Labels for chambers in https://zenodo.org/record/4506930#.YtFzPNLMJ1M
    01. LV myocardium (endo + epi)
    02. RV myocardium (endo + epi)
    03. LA myocardium (endo + epi)
    04. RA myocardium (endo + epi)
    '''

    data_path = r'/home/ds17/Documents/final_data_binary/'
    labels = [1, 2, 3, 4]
    show_chamber_volumes = False
    refine_mesh = False
    chamber_volume_save_path = os.path.join(os.getcwd(), 'chamber_volumes.csv')


    all_volumes = []
    paths = natsorted(glob.glob(data_path + '*.vtk'))
    for path in paths:
        all_case_volumes = get_bloodpool_volume(path,
                                                selected_labels=labels,
                                                label_tag='elemTag',
                                                plot_chamber_volumes=show_chamber_volumes)
        all_volumes.append(all_case_volumes)
        print(path)

    chamber_volume_df = save_volumes2excel(save_path=chamber_volume_save_path,
                                           volume_data=all_volumes,
                                           calculated_labels=labels,
                                           case_paths=paths)
