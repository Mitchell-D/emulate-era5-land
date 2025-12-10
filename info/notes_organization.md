# organization

My goal with this iteration of the project is to make a data
pipeline, model training, model evaluation, and evaluation plotting
system that is robust to different spatial domains, temporal
resolutions, model architectures, input/output data structures,
evaluation types, and plots per evaluation type.

One of the main challenges with this is exposing all the information
each stage needs from the other stages.

## abstract looks at each step's IO

I'm starting to think I need a reliable abstraction for the input
and output domains of models and generators. This would be similar
to my CoordFeatDataset / CoordFeatArray construction and would enable
both a priori verification of a transform's validity.

CoordFeatArray (CFA):

- depiction of shape, ex ("B", "Sh", "Fd", "M")
- feature labels, ex "Fd":["temp", "dwpt", "pres", "apcp"]

CoordFeatTransform (CFT):

A CFT contains an input CFD

In place of a CFA, a CFD with

CoordFeatDataset (CFD):

Although some axes (esp. batch size) may change between instances of
a CFD, within a single CFD everything should be consistent.

CFDs are collections of CFA and CFT objects

### SparseTimegridSampleDataset

- source files (timegrids)
- output dataset features + whether differentiated (window, horizon,
  target, static, static int, derived, auxiliary)
- output dataset shapes (window size, horizon size,
  temporal resolution)
- normalization coefficients
- valid domain (need to finish implementing)

All of the input parameters are serializable.

### PredictionDataset

- overridden configuration options

