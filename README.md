# Polyhedral Meshes
This project is an implementation of the cutting edge algorithm has been explored by Dr. Roi Porrane.

The paper solves the next problem:

Given a Polyhedral-Mesh, deform its vertices by reserving the Mesh-Planarity property or in simple words, it is still a Polyhedral-Mesh at the output.

[![Simple Polyhendral Mesh](https://github.com/itaycsguy/Polyhedral_Meshes/blob/master/Doc/simple_3D_cube.png)]

# Installation
## Option 1
1. Download Blender v2.81 from: [Configured Blender v2.81](https://drive.google.com/file/d/1mGoSfcI1Mg-K9_gTOd3h2hfrxLv4XJom/view?usp=sharing)
2. Download Python v3.7 from: [Configured Python With External Libraries]()
3. Make git clone to another directory
4. Unzip [Configurations/mesh_realization.zip](https://github.com/itaycsguy/Polyhedral_Meshes/blob/master/Configurations/mesh_realization.zip)
5. Open mesh_realization.blend 

## Option 2
1. Download Blender v2.81 from: [Release Blender v2.81](https://www.blender.org/download/releases/2-81/)
2. Make git clone to another directory
3. Download external libraries from: [Exernal Libraries]()
3. Copy [Src/planarization.pyd](https://github.com/itaycsguy/Polyhedral_Meshes/blob/master/Src/planarization.pyd) to the blender DLLs directory
4. cd Polyhedral_Meshes
5. Unzip [Configurations/mesh_realization.zip](https://github.com/itaycsguy/Polyhedral_Meshes/blob/master/Configurations/mesh_realization.zip)
6. Open Configurations/mesh_realization.blend

## Note
* If you change the c++ code, verify your configurations using [Doc/CPP Core Using Python Interface.pdf](https://github.com/itaycsguy/Polyhedral_Meshes/blob/master/Doc/CPP%20Core%20Using%20Python%20Interface.pdf)
