# aind-smartspim-segmentation

Large-scale detection of cell-like structures. The algorithms included in this package are:

**Traditional algorithm**: Method based on the Laplacian of Gaussians technique. These are the image processing steps that happen in every chunk:
1. Laplacian of Gaussians to enhance regions where the intensity changes dramatically (higher gradient).
2. Percentile to get estimated background image.
3. Combination of logical ANDs to filter the LoG image using threshold values and non-linear maximum filter.
4. After identifying initial spots (ZYX points) from 1-3 steps, we prune blobs close to each other within a certain radius $$r$$ using a kd-tree.
5. We take each of these pruned spots and get their contexts which is a 3D image of size $$context = (radius + 1, radius + 1, radius + 1)$$.
6. Finally, we fit a gaussian to each of the spots using its context to be able to prune false positives.

This is a traditional-based algorithm, therefore, parameter tunning is required to make it work.

The output of this algorithm is a CSV file with the following columns:

- Z: Z location of the spot.
- Y: Y location of the spot.
- X: X location of the spot.
- Z_center: Z center of the spot during the guassian fitting, useful for demixing.
- Y_center: Y center of the spot during the guassian fitting, useful for demixing.
- X_center: X center of the spot during the guassian fitting, useful for demixing.
- dist: Euclidean distance or L2 norm of the ZYX center vector, $`norm = \sqrt{z^2 + y^2 + x^2}`$.
- r: Pearson correlation coefficient between integrated 3D gaussian and the 3D context where the spot is located.
- fg: Mean foreground of the proposal.
- bg: Mean background of the proposal.

The output folder structure looks like:

cell_Ex_XXX_Em_XXX/
    metadata/
        processing.json
        proposals.log
    visualization/
        precomputed
        neuroglancer_config.json
    cell_likelihoods.csv
    
The metadata folder includes aind-data-schema metadata useful to track image processing parameters. The log registers every step the algorithm took to get the final result. In the visualization folder you will find the assets useful to visualize the results in neuroglancer. Finally, the cell_likelihoods.csv contains the identified proposals as well as metrics useful to represent them.

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)

Tool for whole brain cell couting using dask and cellfinder in the cloud.

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e .[dev]
```

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```bash
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```bash
black .
```

- Use **isort** to automatically sort import statements:
```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o doc_template/source/ src 
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html doc_template/source/ doc_template/build/html
```
More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).
