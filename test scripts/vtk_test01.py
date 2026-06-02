# Source - https://stackoverflow.com/a
# Posted by Nico Vuaille
# Retrieved 2025-11-25, License - CC BY-SA 4.0
import vtk
from vtk.util.numpy_support import numpy_to_vtk

VTK_data.SetName("VELOCITY")
data.GetPointData().AddArray(VTK_data)

writer = vtk.vtkUnstructuredGridWriter()
writer.SetFileName("Output.vtk")
writer.SetInputData(data)
writer.Update()
writer.Write()