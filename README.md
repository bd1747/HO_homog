<!-- # Project Title

One Paragraph of project description goes here -->

# HO_homog

**TODO** : *One Paragraph of project description goes here*

## Getting Started

The **HO_homog** project is developped on Ubuntu 18.04 with the programming language Python3.

The current stable release of the **HO_homog** Python library can be downloaded from this [repository](https://gitlab.enpc.fr/baptiste.durand/HO_homog/tree/master).

### Prerequisites

Before using the **HO_homog** tools, FEniCS and the gmsh Python API must be installed.

#### FEniCS

To install the FEniCS on Ubuntu, run the following commands:

```bash
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install --no-install-recommends fenics
```

For more information about installation or instructions for other platforms see the [fenicsproject website](https://fenicsproject.org/download/) and the [FEniCS reference manual](https://fenics.readthedocs.io/en/latest/installation.html#debian-ubuntu-packages).

#### Gmsh

The python API of gmsh is part of the gmsh Software Development Kit (SDK).
The exectuble releases of the Gmsh SDK can be download from the [Gmsh website](http://gmsh.info/).

Alternatively, the python API of gmsh can be downloaded and installed via a `pip` command thanks to the [gmsh-sdk package](https://github.com/mrkwjc/gmsh-sdk) created by [Marek Wojciechowski](https://github.com/mrkwjc).
See the [project description](https://pypi.org/project/gmsh-sdk/) for more information about gmsh-sdk.

Installing the executable Gmsh as well is not necessary.

Please note that the gmsh SDK library must be accessible by Python.
On an Ubuntu platform, if you run Python via a Terminal you can complete the PYTHONPATH by means of a `.bashrc` file. It should be placed in your home directory `~/` and should contain a command similar to this one :

```bash
export PYTHONPATH=$PYTHONPATH:/usr/lib/gmsh-4.0.6-Linux64-sdk/lib
```

Alternatively, you can explicitely indicate the path to `gmsh.py` in the **HO_homog** library. To do that, you can use the following command of the `geometry.py` file : `sys.path.insert(0,'/usr/lib/gmsh-4.0.6-Linux64-sdk/lib')`

<!--
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo
-->

<!--
## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

<!-- ## Deployment

Add additional notes about how to deploy this on a live system -->

## Built With

The HO_homog tools have been developped with Python 3.6.7.
They make use of the following applications. The numbers in brackets represent the versions used for the development.

- Python 3 (3.6.7);
- [Gmsh](http://gmsh.info/) - Used to generate finite element meshes, by means of its API for Python (4.0.6);
- [FEniCS](https://fenicsproject.org/) - Finite element computing platform (2018.1.0)
  
<!-- 
## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).  -->

## Authors

- **Baptiste Durand** - *PhD student* - Laboratoire NAVIER, UMR 8205 - [@baptiste.durand](https://gitlab.enpc.fr/baptiste.durand) - [baptiste.durand@enpc.fr](mailto:baptiste.durand@enpc.fr)

See also the graphs of all [contributions](https://gitlab.enpc.fr/baptiste.durand/HO_homog/graphs/master) to this project.

<!-- ## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project. -->

<!-- ## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details -->

<!-- ## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc -->

## Ressources pour la rédaction du fichier README

- [Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) for writing in markdown.
- [Template](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2) of a README<span></span>.md file.
- [Examples](https://github.com/matiassingers/awesome-readme) of readme files.