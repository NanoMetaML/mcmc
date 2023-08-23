<p align=center>
<picture><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Purdue_Boilermakers_logo.svg/1200px-Purdue_Boilermakers_logo.svg.png" alt="Purdue Logo" width=200></picture>
</p>
<p style="font-size:40px;" align=center>
NanoML Github Template 
</p>

![Purdue](https://img.shields.io/badge/Purdue-University-cfb991?style=for-the-badge) 
![Version](https://img.shields.io/static/v1?label=Version&message=0.0.1&color=cfb991&style=for-the-badge)
![License](https://img.shields.io/static/v1?label=License&message=MIT&color=cfb991&style=for-the-badge)

<p style="font-size:30px;" align=center>
Team Languages 
</p>

![Python](https://img.shields.io/badge/-Python-cfb991?logo=python&logoColor=white&style=for-the-badge)
![Cpp](https://img.shields.io/badge/-C++-cfb991?logo=cplusplus&logoColor=white&style=for-the-badge)
![C](https://img.shields.io/badge/-C-cfb991?logo=c&logoColor=white&style=for-the-badge)
![Rust](https://img.shields.io/badge/-Rust-cfb991?logo=rust&logoColor=white&style=for-the-badge)


We implement 3 pretty standard systems for managing our repos.

1. Virtual environments
2. Version Control
3. Packaging

Our goal is to make our code as easy as

    pip install our-package


## Why do we as academics care?

Academia's goal is to further humanity's knowledge and provide access to education with as few hurdles as possible. By making our code easy to use, we bring academia's goal closer to reality and open doors for anyone to come into science. Virtual environments, version control, and proper packaging make our code easily accessible and give access closer to anyone.

## Version Control

Git helps us track changes in our code, collaborate with other team members, and manage releases. It also allows others to easily contribute to our projects and fork them for their own use.

## Virtual Environments

Virtual environments are essential for isolating project dependencies and avoiding conflicts between different Python packages. We use virtual environments to manage and install project-specific packages, ensuring a consistent and reproducible development environment for all contributors.

## Packaging

Packaging code is a widely used package management tool for Python. We use it to define, package, and distribute our code as reusable modules or libraries. This makes it easy for others to install and use our code without worrying about dependencies or compatibility issues.

# Before we get started...

Make sure that you have your ClearML credentials configured at [clear.ml](https:\\clear.ml). You will need these to establish connections with the ClearML servers. This will be elaborated more in the following steps. 

# Getting Started

To get started with our NanoML Github Template, follow these steps:

1. Clone the repository:

        git clone https://github.com/nanometaml/template.git

2. Change directory to the cloned repository:

        cd template

3. Create a virtual environment and activate it:


        python -m venv .venv
        source .venv/bin/activate  # For Linux/Mac
        .venv\Scripts\activate     # For Windows

4. Install the required packages:

        python -m pip install -r requirements.txt

### Why python -m? 
Good question! If you type in `pip` into the cmd, it's not guaranteed to be linked to the same python binary found on your cmd path. The only way to guarantee that the package is installed to the correct python, you have to run pip using python's module mode, i.e., 'python -m pip'.

5. Configure ClearML using the following command:

        clearml-init

This command will initialize the ClearML setup wizard for the server. After this command is run, your terminal should stall at `Paste copied configuration here:`. Now, in your ClearML profile, go to Settings >> Workspace >> App Credentials, and click "Create New Credentials". Your credentials should look like this:

```
api { 
    # Vea Iyer's workspace
    web_server: https://app.clear.ml
    api_server: https://api.clear.ml
    files_server: https://files.clear.ml
    credentials {
        "access_key" = "XXXxxx"
        "secret_key"  = "XXXxxx"
    }
}
```

Paste the API information in your terminal. If everything works, you should get a confirmation message. Now, your ClearML server should be ready to go!

6. Install the package in editable mode:

        python -m pip install -e .

Note: This means this package is now available on your local machine! Access it in your scripts for testing. 

7. Go into test/test.py and change the `my_name` variable to a string with your name.

8. Run test/test.py using the following

        cd test
        python test.py
        >>> Hello from the nanometaml team!
        >>> ...
        >>> <Clearml Output>
        >>> ...
        >>> Done!

If you see this message, everything is working properly!

## Setup Clearml

Go to [clear.ml](https:\\clear.ml) and setup an account. Then email Blake to get access to the NanoML workspace. You'll need to be invited to the NanoML workspace and then use your credentials from the workspace when you run `clearml-init` for the first time.

Now, you're ready to start developing your own NanoML projects using this template.

## Using ClearML

The easiest way to run a clearml script is to run the following in a git repo:

    `clearml-task --name <Task name> --project <project name> --script <entrance script> --queue default`

Note, you need to prepare the script by adding a Task init method from the clearml package and have a requirements.txt file for the environment to be preapred properly in the claerml agent.

# Workflow

 A machine learning experiment is built using three core components: 

 1. Datasets
 2. Models
 3. Trainers

The datasets contain all the data we need for an experiment. We pull in datasets from clearml and locally.

Models are modularized into components and pytorch-lightning modules.

Trainers control the training of the model. In these modules, you define checkpointing, logging, and running the experiment.

All three of these components are loaded using nanomodules.

## NanoModules

`nanomodules` is a lightweight package (literally one function) that calls the `load` function in a module at a path. That means we can keep our models, datasets, and trainers in self contained modules and reused across projects. Our goal with nanomodules is to not reinvent the wheel for every similar nano project.

## Datasets

Datasets are available on kaggle (public facing) and clearml (private). 

## Models 

Models are available on the official NanoMetaML github. 

## Trainer

Preconfigured pytorch-lightning trainers are available on the NanoMetaML github.

### Optional

Become a collaborator on the NanoMetaML organization!!

[https://github.com/NanoMetaML](https://github.com/NanoMetaML)

# Contributing

We welcome contributions from the community! If you would like to contribute to our projects or start a collaboration with one of our gifted team members, please follow these steps:

1. Fork the repository.

2. Create a new branch for your changes.

3. Make your changes and commit them, making sure to follow our commit message guidelines.

4. Push your changes to your fork.

5. Create a pull request against our repository, describing your changes and any relevant information.

Our team will review your pull request and provide feedback or merge it if approved.

# License

All NanoML projects are released under the MIT License. This permissive license allows you to use, modify, and distribute our code, as long as you include the original copyright and license information.
