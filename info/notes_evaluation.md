# evaluation

##

#### testbed/eval\_sequences.py

 1. select list of model weights, batch size, number of batches,
    output feat, and str-defined functional conditions on horizon,
    prediction, or static datasets.
 2. Define a series of dict-based arguments for evaluator objects.
 3. Loop over the selected weights, calling `eval_model_on_sequence`
    with each of them.

#### testbed/eval\_sequences.eval\_model\_on\_sequences

 1. initialize a `ModelDir` object based on the weights path
 2. initialize a prediction generator using the model dir config
 3. for each of the eval getter arg dicts, get the evals by calling
    `get_sequence_evaluator_objects` with the model dir, sequence
    generator arguments, and eval getter args.
 4. for each generated batch, for each evalutaor, call `add_batch`
    with the inputs, true states, and predictions
 5. generate a pkl file for each evaluator

#### testbed/eval\_sequences.get\_sequence\_evaluator\_objects

 1. get horizon indeces for commonly-used feature indeces.
 2. list evaluator types that distinguish between absolute error
    and error bias.
 3. list eval types that contain all feats despite feat parameter
 4. instantiate an evaluator object of each preconfigured type.
 5. collect a list of all initialized `Evaluator` instances along
    with a string name that includes the eval name, eval metric, and
    other qualifiers.

#### evaluator object arguments

 - every evaluator has an "attrs" dict with arguments for the model
   config dict, the generator args dict, and a plot\_spec dict.
 - `EvalJointHist` axis args are either:
    - 2-tuple ((dataset, feature), (hist\_min, hist\_max, hist\_res))
    - 3-tuple (((ds,feat) for ingredient in derived\_func),
               derived\_func, (hist\_min, hist\_max, hist\_res))

`EvalHorizon` *horizon* (pred\_coarseness)

 - only one variant; all target feats

`EvalTemporal` *temporal* (use\_absolute\_error)

 - all target feats; split by error type

`EvalStatic` *static-combos* (soil\_idxs, use\_absolute\_error)

 - all target feats; split by error type

`EvalEfficiency` *efficiency* (pred\_feat\_idx, pred\_coarseness)

 - independent per feat

`EvalJointHist` *hist-true-pred* (ax1\_args, ax2\_args)

 - independent per feat

`EvalJointHist` *hist-saturation-error* (ax1\_args, ax2\_args,
use\_absolute\_error)

 - independent per feat; split by error type

`EvalJointHist` *hist-infiltration* (ax1\_args, ax2\_args,
ignore\_nan, covariate\_feature, pred\_coarseness,
coarse\_reduce\_func)

 - independent per feat.

`EvalJointHist` *hist-state-increment* (ax1\_args, ax2\_args,
covariate\_feature, use\_absolute\_error, ignore\_nan,
pred\_coarseness)

 - independent per feat

`EvalJointHist` *hist-humidity-temp* (ax1\_args, ax2\_args,
coarse\_reduce\_func, covariate\_feature, use\_absolute\_error,
ignore\_nan, pred\_coarseness)

 - independent per feat; split by error type

#### new evaluator configuration

 - each Evaluator instance category is configured to have an
   Evaluator type string, a set of required arguments, and a dict of
   pre-defined arguments.

 - When dispatching, specify tuple of the instance category string
   followed by positional arguments matching the list order of
   required arguments.

 - dict of evaluator options in the `evaluators` module provides
   mapping for pre-configuration Evalutor type strings.

 - inside `get_sequence_evaluator_objects` analog, zip positionals
   with the required arguments and assemble full `__init__` function
   kwarg dict in order to dynamically create a list of evaluators
   and corresponding unique pkl file paths.

 - break EvalJointHist axis args into ax1\_dataset, ax2\_dataset, ax1\_feat,
   ax2\_feat, ax1\_hbounds, ax2\_hbounds, ax1\_hres, ax2\_hres so that the
   default and required arguments make more sense.
    - for derived values like infiltration ratio, later define a dataset called
      "derived" and enable separate derived feat recipes to be specified in a
      dict as a default argument.

 -
