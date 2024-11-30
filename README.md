# 3d_Well_Trajectory_Surveying 

## Overview

This project is a 3D Well Trajectory Visualization Tool designed to facilitate precise planning and visualization of directional drilling profiles. It enables users to input key parameters and generate accurate 3D representations of well trajectories, enhancing decision-making and operational efficiency.

The tool also includes a feature for plotting multiple well trajectories consecutively, allowing users to analyze their spatial relationship within a reservoir. A built-in proximity alert system ensures safety and collision avoidance by flagging instances where well trajectories approach within a critical distance of 20 feet, enabling users to make real-time adjustments to drilling parameters.

## Features

- **Interactive Input**: Users can define the type of trajectory, such as:
  - J-Bend
  - S-Bend
  - Single & Double Build Horizontal
- **Parameter Specification**: Input target coordinates, surface coordinates, build-up angle, and other essential drilling parameters.
- **3D Visualization**: Generates and displays a 3D well trajectory profile based on user inputs.
- **Multi-Well Visualization**: Allows users to plot **two consecutive well trajectories** within a given reservoir for side-by-side analysis.
- **Collision Avoidance System**: 
  - Includes a **flagging system** that alerts the user if two well trajectories come within a close proximity of **20 feet**.
  - Enables users to modify well parameters in real-time to avoid intersection or collision of wells.
- **Accurate Calculations**: Utilizes standard directional drilling equations to ensure precise trajectory modeling.

## Use Cases

- Directional drilling design and planning.
- Collision risk analysis for multi-well setups in shared reservoirs.
- Academic or training purposes for petroleum engineering students.
- Visualization of complex well profiles for better understanding.

## Input Parameters

The tool requires the following inputs:
1. **Surface Coordinates**: (X, Y, Z) of the well's starting point.
2. **Target Coordinates**: (X, Y, Z) of the desired endpoint.
3. **Build-Up Angle**: The inclination rate used to transition from vertical to the desired trajectory.
4. **Trajectory Type**: Build-and-Hold, Horizontal, or others.
5. **Second Well Parameters** (optional): For plotting two trajectories together.
6. Additional data as per trajectory requirements.

## Output

- A **3D plot** showing the trajectory of one or two wells from surface to target.
- **Collision Warning Flags**: A visual or textual alert if the proximity between two trajectories falls below **20 feet**.
- Optional: Export trajectory data as a CSV or text file for further analysis.

## Screenshots
![Figure_1](https://github.com/user-attachments/assets/1d340e7d-930c-4e35-9899-c1c0b6c1678f)
![b](https://github.com/user-attachments/assets/1e8443d2-659b-458c-aeaf-ad75586c029d)



## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - `matplotlib` for 3D plotting
  - `numpy` for mathematical computations
  - Optionally: `pandas` for data handling, `seaborn` for enhanced visuals
 
