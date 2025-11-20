## Introduction

Compiling FFVship on windows has been relatively complex due to lib placement not being known to compiler, zimg being non trivial to build and include files missing. As such, I upload below a dev kit containing pre compiled dependencies.

[Windows_devkit.zip](https://codeberg.org/Line-fr/Vship/releases/download/v4.0.0/Windows_devkit.zip)

## Steps

- Unzip the devkit to get a folder containing various files

- Copy these files and dump them in vship folder

- Use `make buildFFVSHIP` (amd) or `make buildFFVSHIPcuda` (cuda) or `make buildFFVSHIPall` (amd all arch) or `make buildFFVSHIPcudaall` (cuda all arch) depending on your system and goal
