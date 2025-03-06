# aind-smartspim-segmentation

## Large-scale Detection of Cell-like Structures

aind-smartspim-segmentation provides an algorithm for detecting cell-like structures in large-scale microscopy images. The package currently includes:

### Traditional Algorithm (Laplacian of Gaussians-based Method)

A classical approach using image processing techniques to identify cell-like structures in 3D images. The processing pipeline includes:

1. **Laplacian of Gaussians (LoG)**: Enhances regions with high intensity changes (high gradient).
2. **Background Estimation**: Uses percentile filtering to estimate background intensity.
3. **Filtering**: Logical AND operations combine thresholding and non-linear maximum filtering on the LoG image.
4. **Blob Pruning**: Uses a kd-tree to remove blobs that are too close to each other within a radius \(r\).
5. **Context Extraction**: Extracts a 3D sub-image of size \((r + 1, r + 1, r + 1)\) around each pruned spot.
6. **Gaussian Fitting**: Fits a Gaussian model to each extracted context to reduce false positives.

Since this method relies on predefined parameters, **parameter tuning is required** for optimal performance.

## Output

The algorithm outputs a CSV file with detected spots and relevant metrics:

| Column                              | Description                                                                   |
| ----------------------------------- | ----------------------------------------------------------------------------- |
| **Z, Y, X**                         | Spot location coordinates                                                     |
| **Z\_center, Y\_center, X\_center** | Refined spot center after Gaussian fitting (useful for demixing)              |
| **dist**                            | Euclidean distance of the ZYX center (\(\sqrt{z^2 + y^2 + x^2}\))             |
| **r**                               | Pearson correlation coefficient between fitted Gaussian and extracted context |
| **fg**                              | Mean foreground intensity                                                     |
| **bg**                              | Mean background intensity                                                     |

### Folder Structure

```
cell_Ex_XXX_Em_XXX/
    metadata/
        processing.json
        proposals.log
    visualization/
        precomputed/
        neuroglancer_config.json
    cell_likelihoods.csv
```

- **metadata/**: Contains `processing.json` (algorithm parameters) and `proposals.log` (processing logs).
- **visualization/**: Includes assets for **Neuroglancer** visualization.
- **cell\_likelihoods.csv**: Contains detected spots and associated metrics.

## Features

- **High-throughput** processing of whole-brain images
- **Cloud-compatible** with **Dask** and **CellFinder**
- **Neuroglancer integration** for visualization

## Installation

To install the package:

```bash
pip install -e .
```

For development:

```bash
pip install -e .[dev]
```

## Contributing

### Code Quality & Testing

Use the following tools to maintain code quality:

- **Unit Testing & Coverage**
  ```bash
  coverage run -m unittest discover && coverage report
  ```
- **Documentation Coverage**
  ```bash
  interrogate .
  ```
- **Code Style (PEP 8, Formatting, Imports Sorting)**
  ```bash
  flake8 .
  black .
  isort .
  ```

### Pull Requests

- Internal members should create a branch.
- External contributors should fork the repository and open a pull request.
- Follow **Angular-style commit messages**:
  ```text
  <type>(<scope>): <short summary>
  ```
  **Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `build`

## Documentation

To generate documentation:

```bash
sphinx-apidoc -o doc_template/source/ src
```

Then build HTML documentation:

```bash
sphinx-build -b html doc_template/source/ doc_template/build/html
```

More details on Sphinx installation [here](https://www.sphinx-doc.org/en/master/usage/installation.html).

---

&#x20;&#x20;

