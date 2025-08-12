Fits a sparse linear regression model using variational inference with parameter explosion and spike-and-slab priors.

# Intallation

```{r setup, message=FALSE, warning=FALSE}
# Check for and install required packages
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

if (!requireNamespace("spxlvb", quietly = FALSE)) {
  remotes::install_github("peterolejua/spxlvb")
}

library(spxlvb)
```
