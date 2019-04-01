<!-- # Project Title

One Paragraph of project description goes here -->

# HO_homog

**TODO** : *One Paragraph of project description goes here*

## Getting Started

The **HO_homog** project is developped on Ubuntu 18.04 with the programming language Python3.

The current stable release of the **HO_homog** Python library can be downloaded from this [repository](https://gitlab.enpc.fr/baptiste.durand/HO_homog/tree/master).

### Prerequisites

Before using the **HO_homog** tools FEniCS and `pip` (or setuptools) must be installed.

#### FEniCS

To install the FEniCS on Ubuntu, run the following commands:

```bash
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install --no-install-recommends fenics
```

For more information about installation or instructions for other platforms see the [fenicsproject website](https://fenicsproject.org/download/) and the [FEniCS reference manual](https://fenics.readthedocs.io/en/latest/installation.html#debian-ubuntu-packages).


#### Virtual Environment
Using a virtual environment for Python projects is recommended. **HO_homog** and Gmsh can be installed inside a virtual environment but FEniCS cannot.

Make a new virtual environment :
```bash
python3 -m venv /path/to/project/project_name
```
Then activate it :
```bash
source /path/to/project/project_name/bin/activate
```
Now, the HO_homog package and its dependencies can be installed inside this virtual environment (see instructions below).

Supposing FEniCS has been already installed on the system, the settings of the virtual environment can be modified to make it accessible. In `/path/to/project/project_name/pyenv.cfg` : `include-system-site-packages = true`.

### Installing
The **HO_homog** Python package can be installed with `pip`.
- Install it from the gitlab repository directly (preferred way):

```bash
pip3 install git+https://baptiste.durand@gitlab.enpc.fr/baptiste.durand/HO_homog.git#egg=ho_homog
```

- Or [download](https://gitlab.enpc.fr/baptiste.durand/HO_homog/repository/archive.tar?ref=master) the HO_homog repository files. Then, move to the HO_homog directory and use `pip` : 

```bash
 pip3 install . --no-cache-dir
```

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
- **Arthur Lebée** - *Researcher* - Laboratoire NAVIER, UMR 8205 - [@arthur.lebee](https://gitlab.enpc.fr/arthur.lebee) - [Researcher page on Laboratoire Navier website](https://navier.enpc.fr/LEBEE-ARTHUR,144)

See also the graphs of all [contributions](https://gitlab.enpc.fr/baptiste.durand/HO_homog/graphs/master) to this project.

<!-- ## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project. -->

<!-- ## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details -->

## Acknowledgments

The authors acknowledge the support of the French Agence Nationale de la Recherche (ANR), under grant ANR-17-CE08-0039 ([project ArchiMatHOS](http://www.agence-nationale-recherche.fr/Projet-ANR-17-CE08-0039)).
<img src="/img/ArchiMATHOS_ANR.png" height="250">


<!--
## Ressources pour la rédaction du fichier README

- [Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) for writing in markdown.
- [Template](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2) of a README<span></span>.md file.
- [Examples](https://github.com/matiassingers/awesome-readme) of readme files.
-->