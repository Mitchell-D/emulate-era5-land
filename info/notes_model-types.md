# model types

A list of model types and strategies that may apply to my case, and
the rationale behind them.

## architectures

### distributional regression

**variational encoder/decoder**

Model maps inputs to arbitrary-dimensional latent vectors, one for
each parameter in a parametric distribution (ie gaussian). Samples
are drawn from this latent distribution, and decoded to an ensemble
of predictions. During training, the reparameterization trick is
utilized to facilitate backpropagation through the random variable.

## learning strategies

### distributional regression

**engression**

Introduce noise as an input, and use multi-objective optimization to
incentivize the model to generate an ensemble of outputs that
approach a well-calibrated uncertainty distribution.

**quantile regression**

A quantile in (0,1) is provided as an argument to the model and loss
function, and the model is trained to minimize a modified version of
MAE that more weakly penalizes error in the direction of the
selected quantile, with an amount proportional to the quantile.

### equitable sub-distribution learning

**dynamic loss weighting**

Use a function during sampling to map samples to loss weights based
on any kind of property (ie static, analytic, histogram position).

**sampling probability**

Like dynamic loss weighting, but map samples to likelihoods of
selection (ie float [0,1]). During dataset chunk replenish, apply
mapping to each sample, select a random value, and subset to samples
with probabilities above the selected value.

**subdomain clustering**

Input embeddings can be clustered using techniques like t-SNE, UMAP,
and last-layer features, then the groupings can be used to assign
sample weights/probabilities, implement domain adversarial methods,
or perhaps train separate models for distinct domains.

### a-priori model structure

Need to find literature on best way to train for multiple objectives
in a system of modular model components. My intuition tells me:

- Pre-train on land surface temperature profile (no soil moisture)

- Freeze LST model and use it within model predicting snow cover
  - potentially use true LST values (ie teacher forcing) for training

- Freeze and use LST and snow models to train flux model
  - again, possible teacher forcing.

- Freeze and use LST and snow models to train soil water model

- Simultaneously train the whole assembly of models.
  - probably use a low learning rate
  - consider systematically freezing/unfreezing sub-models
  - combine with equitable sub-distribution learning techniques.
