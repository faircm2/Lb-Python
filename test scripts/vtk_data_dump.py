import numpy as np

try:
    import vtk
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk
except ImportError as e:
    print(f"Error importing vtk: {e}")
    raise
import os


class VTKDataDumper:
    """
    Dump 3D scalar or vector data to VTK ImageData (.vti) files for ParaView.
    """
    def __init__(self, xn, yn, zn, output_dir="output", spacing=(1.0, 1.0, 1.0), data_order="xyz"):
        """
        Args:
            xn, yn, zn: Grid dimensions (nodes)
            output_dir: Directory for .vti files (default: 'output')
            spacing: Grid spacing (x, y, z) (default: (1.0, 1.0, 1.0))
            data_order: Simulation array order, e.g., 'xyz' (default) or 'zyx' (transpose to VTK x,y,z)
        """
        self.xn = xn
        self.yn = yn
        self.zn = zn
        self.output_dir = output_dir
        self.spacing = spacing
        self.data_order = data_order  # 'xyz' or 'zyx'
        os.makedirs(self.output_dir, exist_ok=True)

    def dump_to_vti(self, data, iteration, field_name="field"):
        """
        Dump 3D scalar or vector array to .vti file.
        Args:
            data: 3D NumPy array (xn, yn, zn) for scalar or (3, xn, yn, zn) for vector
            iteration: Iteration number for filename
            field_name: Field name in ParaView (default: 'field')
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be a NumPy array, got {type(data)}")

        # Ensure C-order and check for invalid values
        data = np.ascontiguousarray(data)
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print(f"Warning: {field_name} contains NaN or Inf values")
        print(f"{field_name} range: min={np.min(data):.4f}, max={np.max(data):.4f}, shape={data.shape}")

        # Check if data is vector (4D) or scalar (3D)
        is_vector = data.ndim == 4 and data.shape[0] == 3
        expected_shape = (3, self.xn, self.yn, self.zn) if is_vector else (self.xn, self.yn, self.zn)

        if data.shape != expected_shape:
            raise ValueError(f"{field_name} shape {data.shape} != expected {expected_shape}")

        image_data = vtk.vtkImageData()
        image_data.SetDimensions(self.xn, self.yn, self.zn)
        image_data.SetSpacing(*self.spacing)
        image_data.SetOrigin(0.0, 0.0, 0.0)

        if is_vector:
            # Vector: transpose (3, xn, yn, zn) to (xn, yn, zn, 3) for VTK
            if self.data_order == "zyx":
                data_processed = data.transpose(1, 2, 3, 0)[:, :, ::-1]  # Map (x,y,z) to VTK (z,y,x)
            else:
                data_processed = data.transpose(1, 2, 3, 0)  # Keep (x,y,z)
            data_flat = data_processed.reshape(-1, 3)
            vtk_data = numpy_to_vtk(data_flat, deep=True, array_type=vtk.VTK_DOUBLE)
            vtk_data.SetNumberOfComponents(3)
            vtk_data.SetName(field_name)
            image_data.GetPointData().SetVectors(vtk_data)
        else:
            # Scalar: transpose (xn, yn, zn) to (zn, yn, xn) if zyx
            if self.data_order == "zyx":
                data_processed = data.transpose(2, 1, 0)  # Map (x,y,z) to VTK (z,y,x)
            else:
                data_processed = data
            data_flat = data_processed.ravel(order='C')
            vtk_data = numpy_to_vtk(data_flat, deep=True, array_type=vtk.VTK_DOUBLE)
            vtk_data.SetName(field_name)
            image_data.GetPointData().SetScalars(vtk_data)

        filename = os.path.join(self.output_dir, f"{field_name}_{iteration:06d}.vti")
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(image_data)
        writer.Write()
        print(f"Saved {filename}")


    # Assuming self.xn, self.yn, density, velocity_x, velocity_y, force_x, force_y, gravity are defined globally
    # and IBM_object is a class with attributes num_nodes, node (list of objects with x, y attributes), and is_wall
    def write_fluid_vtk(self, time, rho):
        """
        Write fluid state to a VTK file at each t_disk step.
        Data written: density difference (density - 1), x-component of velocity, y-component of velocity.
        Designed to be read by ParaView.
        """
        # Create filename
        output_filename = f"vtk/fluid_t{time}.vtk"
        
        # Ensure vtk directory exists
        os.makedirs("vtk", exist_ok=True)
        
        # Open file
        with open(output_filename, 'w') as output_file:
            # Write VTK header
            output_file.write("# vtk DataFile Version 3.0\n")
            output_file.write("fluid_state\n")
            output_file.write("ASCII\n")
            output_file.write("DATASET RECTILINEAR_GRID\n")
            output_file.write(f"DIMENSIONS {self.xn} {self.yn - 2} 1\n")
            
            # Write X coordinates
            output_file.write(f"X_COORDINATES {self.xn} float\n")
            for X in range(self.xn):
                output_file.write(f"{X + 0.5} ")
            output_file.write("\n")
            
            # Write Y coordinates
            output_file.write(f"Y_COORDINATES {self.yn - 2} float\n")
            for Y in range(1, self.yn - 1):
                output_file.write(f"{Y - (self.yn - 1) / 2.0} ")
            output_file.write("\n")
            
            # Write Z coordinates
            output_file.write("Z_COORDINATES 1 float\n")
            output_file.write("0\n")
            
            # Write point data
            output_file.write(f"POINT_DATA {self.xn * (self.yn - 2)}\n")
            
            # Write density difference
            output_file.write("SCALARS density_difference float 1\n")
            output_file.write("LOOKUP_TABLE default\n")
            for Y in range(1, self.yn - 1):
                for X in range(self.xn):
                    output_file.write(f"{rho[X][Y] - 1.0}\n")
            
            # Write velocity vectors
            output_file.write("VECTORS velocity_vector float\n")
            for Y in range(1, self.yn - 1):
                for X in range(self.xn):
                    vx = velocity_x[X][Y] + 0.5 * (force_x[X][Y] + gravity) / rho[X][Y]
                    vy = velocity_y[X][Y] + 0.5 * force_y[X][Y] / rho[X][Y]
                    output_file.write(f"{vx} {vy} 0\n")


    def write_particle_vtk(self, time, boundary):
        """
        Write wall state (node positions) to a VTK file at each t_disk step.
        Designed to be read by ParaView.
        boundary: List of IBM_object instances with num_nodes, node (list of objects with x, y), and is_wall.
        """
        # Ensure vtk directory exists
        os.makedirs("vtk", exist_ok=True)
        
        for i in range(len(boundary)):
            # Create filename
            output_filename = f"vtk/wall_t{time}.vtk"
            
            # Open file
            with open(output_filename, 'w') as output_file:
                # Write VTK header
                output_file.write("# vtk DataFile Version 3.0\n")
                output_file.write("wall_state\n")
                output_file.write("ASCII\n")
                output_file.write("DATASET POLYDATA\n")
                
                # Write node positions
                output_file.write(f"POINTS {boundary[i].num_nodes} float\n")
                for n in range(boundary[i].num_nodes):
                    output_file.write(f"{boundary[i].node[n].x} {boundary[i].node[n].y - (self.yn - 2) / 2.0} 0\n")
                
                # Write lines between neighboring nodes
                if boundary[i].is_wall:
                    output_file.write(f"LINES {boundary[i].num_nodes - 2} {3 * (boundary[i].num_nodes - 2)}\n")
                    for n in range(boundary[i].num_nodes // 2 - 1):
                        output_file.write(f"2 {n} {n + 1}\n")
                    for n in range(boundary[i].num_nodes // 2, boundary[i].num_nodes - 1):
                        output_file.write(f"2 {n} {n + 1}\n")
                else:
                    output_file.write(f"LINES {boundary[i].num_nodes} {3 * boundary[i].num_nodes}\n")
                    for n in range(boundary[i].num_nodes):
                        output_file.write(f"2 {n} {(n + 1) % boundary[i].num_nodes}\n")
                
                # Write vertices
                output_file.write(f"VERTICES 1 {boundary[i].num_nodes + 1}\n")
                output_file.write(f"{boundary[i].num_nodes} ")
                for n in range(boundary[i].num_nodes):
                    output_file.write(f"{n} ")
                output_file.write("\n")