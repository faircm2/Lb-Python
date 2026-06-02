import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


def extract_simulation_data(directory):
    simulation_data = []
    
    # Regular expression to parse the line in the file
    line_pattern = re.compile(r"Simulation with (\d+) nodes, grid (\d+)x(\d+); Run-time: ([\d.]+) s")
    
    # Regular expression to parse grid and nodes from filename
    filename_pattern = re.compile(r"runtime-grid-(\d+)x(\d+)-nodes(\d+)-L(\d+)\.txt")
    
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return simulation_data
    
    # Pattern to match files like runtime-grid-50x*-nodes*-L*.txt
    file_pattern = os.path.join(directory, "runtime-grid-50x*-nodes*-L1.txt")
    
    for filepath in glob.glob(file_pattern):
        # Extract grid info and nodes from filename
        filename = os.path.basename(filepath)
        filename_match = filename_pattern.match(filename)
        if not filename_match:
            print(f"Warning: File {filename} does not match expected pattern, skipping.")
            continue
        
        grid_x_filename = int(filename_match.group(1))
        grid_y_filename = int(filename_match.group(2))
        nodes_filename = int(filename_match.group(3))
        l_value = int(filename_match.group(4))
        
        # Read file content
        try:
            with open(filepath, 'r') as file:
                for line in file:
                    match = line_pattern.match(line.strip())
                    if match:
                        nodes = int(match.group(1))
                        grid_x = int(match.group(2))
                        grid_y = int(match.group(3))
                        runtime = float(match.group(4))
                        
                        # Verify consistency between filename and file content
                        if (grid_x_filename != grid_x or 
                            grid_y_filename != grid_y or 
                            nodes_filename != nodes):
                            print(f"Warning: Mismatch in {filename}: "
                                  f"Filename grid {grid_x_filename}x{grid_y_filename}, nodes {nodes_filename}; "
                                  f"Content grid {grid_x}x{grid_y}, nodes {nodes}")
                        
                        simulation_data.append({
                            'nodes': nodes,
                            'grid_x': grid_x,
                            'grid_y': grid_y,
                            'runtime': runtime,
                            'filename': filename,
                            'directory': directory,
                            'L': l_value
                        })
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    return simulation_data


if __name__ == "__main__":
    # Root directory (update as needed)
    base_dir = r"C:\Dirk\temp\bwcluster\20250523.1"
    #base_dir = r"C:\Dirk\temp\bwcluster\20250531.1"
    
    data = extract_simulation_data(base_dir)
    
    if not data:
        print("No data found to plot. Please check the directory and file patterns.")
        exit(1)
    
    # Print extracted data for verification
    for entry in data:
        print(f"Directory: {entry['directory']}, File: {entry['filename']}, "
              f"Nodes: {entry['nodes']}, Grid: {entry['grid_x']}x{entry['grid_y']}, "
              f"L: {entry['L']}, Runtime: {entry['runtime']} s")
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv("simulation_data.csv", index=False)
    
    # Plot runtime vs grid_y for each number of nodes
    plt.figure(figsize=(10, 6))
    nodes_values = sorted(set(entry['nodes'] for entry in data))
    
    for nodes in nodes_values:
        node_data = [entry for entry in data if entry['nodes'] == nodes]
        grid_y_values = [entry['grid_y'] for entry in node_data]
        grid_x_values = [entry['grid_x'] for entry in node_data]
        runtimes = [entry['runtime'] for entry in node_data]
        
        # Sort by grid_y for consistent plotting
        sorted_pairs = sorted(zip(grid_y_values, runtimes, grid_x_values))
        grid_y_values, runtimes, grid_x_values = zip(*sorted_pairs)
        
        # Plot the line and points
        plt.plot(grid_y_values, runtimes, marker='o', label=f'nodes-{nodes}')
        
        # Add labels for each point in [Yn x (96 x NoXnBlocks x 50)] format
        for x, y, grid_x in zip(grid_y_values, runtimes, grid_x_values):
            # Calculate NoXnBlocks = grid_y / 96 / 50
            no_xn_blocks = x / 96 / 50
            # Format NoXnBlocks: integer if whole number, else 2 decimal places
            no_xn_blocks_str = f"{int(no_xn_blocks)}" if no_xn_blocks.is_integer() else f"{no_xn_blocks:.2f}"
            # Offset the label slightly
            label_x = x + 0.02 * (max(grid_y_values) - min(grid_y_values))  # Small offset in x
            label_y = y + 0.02 * (max(runtimes) - min(runtimes))  # Small offset in y
            plt.text(label_x, label_y, f'[{grid_x} x (96 x {no_xn_blocks_str} x 50)]', fontsize=8, ha='left', va='bottom')
    
    plt.xlabel('Grid Size [Xn]')
    plt.ylabel('Runtime [s]')
    plt.title('Runtime vs Grid Size [50 x Xn] by Nodes')
    plt.legend()
    plt.grid(True)
    plt.savefig('runtime_vs_grid_size.png')
    plt.show()