"""
ModelSet and ModelDir are abstractions on the directory structure that is
enforced in the tracktrain framework.

ModelSet is a generalization of a number of models that share a parent
directory, and facilitates bulk access to all of the models' data.
This includes training metrics, configuration, and the models themselves.
"""
import numpy as np
import json
from pathlib import Path
from collections.abc import Callable
import matplotlib.pyplot as plt

class ModelDir:
    """
    ModelDir is an abstraction for a directory minimally containing a config
    file sufficient to create a compilable Model object, and provides methods
    for interfacing with the model's configuration and training metrics.
    """
    def __init__(self, model_dir:Path):
        """ Initialize a ModelDir from an existing directory """
        if type(model_dir) == str:
            model_dir = Path(model_dir)
        self.name = model_dir.name
        self.dir = model_dir
        children = [
                f for f in model_dir.iterdir()
                if (f.name.split("_")[0]==self.name and not f.is_dir())
                ]
        self.path_config = self.dir.joinpath(f"{self.name}_config.json")
        self.req_files = (self.path_config,)
        self.path_prog = self.dir.joinpath(f"{self.name}_metrics_simple.json")
        self._check_req_files()
        self._prog = None
        self._summary = None
        self._config = None

    def __str__(self):
        return f"ModelDir({self.dir})"
    def __repr__(self):
        return str(self)

    def prog(self):
        """
        Return the progress csv as a 2-tuple (labels:list, data:ndarray)
        data is a (E,M) array of M metrics' data per epoch E; each metric is
        labeled by the corresponding string in 'labels'.
        """
        if self._prog is None:
            self._prog = self.load_prog(as_array=True)
        return self._prog

    @property
    def metric_labels(self):
        return self.prog()[0]
    @property
    def metric_data(self):
        """
        Returns a (E,M) shaped ndarray of M metric values over E epochs.
        """
        return self.prog()[1]
    @property
    def config(self):
        """ Returns the model config dictionary.  """
        if self._config == None:
            self._config = self._load_config()
        return self._config

    def get_metric(self, metric):
        """"
        Returns the per-epoch metric data array for one or more metrics.

        If a single str metric label is provided, a (E,) array of that metric's
        data over E epochs is returned.

        If a list of M str metric labels is provided, a (E,M) array of the
        corresponding metrics' data are provided (in the order of the labels).

        :@param metric: String metric label or list of metrics
        """
        if type(metric) is str:
            assert metric in self.metric_labels
            return self.metric_data[:,self.metric_labels.index(metric)]
        assert all(m in self.metric_labels for m in metric)
        idxs = np.array([self.metric_labels.index(m) for m in metric])
        return self.metric_data[:,idxs]

    def _check_req_files(self):
        """
        Verify that all files created by the build() function exist in the
        model directory (ie _summary.txt and _config.json).
        """
        try:
            assert all(f.exists() for f in self.req_files)
        except:
            raise FileNotFoundError(
                f"All of these files must be in {self.dir.as_posix()}:\n",
                tuple(f.name for f in self.req_files))
        return True

    def load_prog(self):
        """ Load the simple metric progress json as a nested dict of arrays """
        if self.path_prog is None:
            raise ValueError(
                    "Cannot return progress csv. "
                    f"File not found: {self.path_prog.as_posix()}"
                    )
        prog = json.load(self.path_prog.open("r"))
        return {
                "train":{k:np.array(prog["train"][k])
                    for k in prog["train"].keys()},
                "val":{k:np.array(prog["val"][k])
                    for k in prog["val"].keys()},
                "train_epochs":np.array(prog["train_epochs"]),
                "val_epochs":np.array(prog["val_epochs"]),
                "lr":np.array(prog["lr"]),
                }

    def _load_config(self):
        """
        Load the configuration JSON associated with a specific model as a dict.
        """
        self._config = json.load(self.path_config.open("r"))
        return self._config

    def update_config(self, update_dict:dict):
        """
        Update the config json to have the new keys, replacing any that exist.

        CAREFUL! Overwrites and reloads the json configuration file.

        This is useful for retroactively updating json files that must meet
        a newly-enforced standard, or for recategorizing models.

        :@param update_dict: dict mapping string config field labels to new
            json-serializable values.

        :@return: the config dict after being serialized and reloaded
        """
        ## Get the current configuration and update it
        cur_config = self.config
        cur_config.update(update_dict)
        ## Overwrite the json with the new version
        json.dump(cur_config, self.path_config.open("w"), indent=4)
        ## reset the config and reload the json by returning the property
        self._config = None
        return self.config

class ModelSet:
    """
    A ModelSet abstracts collection of model instances
    """
    @staticmethod
    def from_dir(model_parent_dir:Path):
        """
        Assumes every subdirectory of the provided Path is a ModelDir-style
        model directory
        """
        model_dirs = [
                ModelDir(d)
                for d in Path(model_parent_dir).iterdir()
                if d.is_dir()
                ]
        return ModelSet(model_dirs=model_dirs)

    def __init__(self, model_dirs:list, check_valid=True):
        """ """
        assert all(type(m) is ModelDir for m in model_dirs)
        ## Validate all ModelDir objects unless check_valid is False
        assert check_valid or all(m._check_req_files() for m in model_dirs)
        self._model_dirs = tuple(model_dirs)

    def __str__(self):
        return ", ".join(list(map(str,self._model_dirs)))
    def __repr__(self):
        return str(self)

    @property
    def model_dirs(self):
        """ return the model directories as a tuple """
        return self._model_dirs
    @property
    def model_names(self):
        """ Return the string names of all ModelDir objects in the ModelSet """
        return tuple(m.name for m in self.model_dirs)

    def subset(self, rule:Callable=None, substr:str=None, check_valid=True):
        """
        Return a subset of the ModelDir objects in this ModelSet based on
        one or both of:

        (1) A Callable taking the ModelDir object and returning True or False.
        (2) A substring that must be included in the model dir's name property.

        :@param rule: Function taking a ModelDir as the first positional arg,
            and returning True iff the ModelDir should be in the new ModelSet
        :@param substr: String that must be included in the ModelDir.name
            string property of all directories in the returned ModelSet

        :@return: ModelSet with all ModelDir objects meeting the conditions
        """
        subset = self.model_dirs
        if not rule is None:
            subset = tuple(filter(rule, subset))
        if not substr is None:
            subset = tuple(filter(lambda m:substr in m.name, subset))
        return ModelSet(subset, check_valid=check_valid)

    def scatter_metrics(self,xmetric:str,ymetric:str,fig_path=None,show=False,
                     use_notes=False, plot_spec={}):
        """  """
        ps = {"xlabel":"epoch", "ylabel":"", "title":"", "cmap":"Set1",
              "text_size":12, "norm":"linear", "logx":False, "figsize":(16,12),
              "plot_kwargs":{}, "xlim":None, "ylim":None, "line_width":2,
              "facecolor":"white", "legend_cols":1, "legend_size":8}
        #line_styles = ("-", ":", "--", "-.")
        ps = {**ps, **plot_spec}
        fig,ax = plt.subplots()

        model_xmetrics = []
        model_ymetrics = []
        model_labels = []
        for md in sorted(self.model_dirs,key=lambda m:m.dir.name):
            model_xmetrics.append(md.get_metric(xmetric)[-1])
            model_ymetrics.append(md.get_metric(ymetric)[-1])
            model_labels.append(f"{md.name}")
            if use_notes:
                model_labels[-1] += " - " + md.config.get("notes")

        print(len(model_labels))

        cmap = plt.cm.get_cmap(ps.get("cmap"), len(model_labels))
        print(len(model_xmetrics), len(model_ymetrics), len(model_labels))
        scatter = plt.scatter(
                model_xmetrics,
                model_ymetrics,
                c=list(range(len(model_labels))),
                cmap=cmap,
                )
        leg_elem = scatter.legend_elements(num=len(model_labels))
        ax.legend(handles=leg_elem[0],
                  ncols=ps.get("legend_cols"),
                  labels=model_labels,
                  prop={"size":ps.get("legend_size")},
                  )
        if ps["logx"]:
            plt.semilogx()

        ax.set_title(ps.get("title"))
        ax.set_xlabel(ps.get("xlabel"))
        ax.set_ylabel(ps.get("ylabel"))
        ax.set_facecolor(ps.get("facecolor"))
        if not ps.get("xlim") is None:
            ax.set_xlim(*ps["xlim"])
        if not ps.get("ylim") is None:
            ax.set_ylim(*ps["ylim"])
        if show:
            plt.show()
        if not fig_path is None:
            fig.set_size_inches(*ps.get("figsize"))
            fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)

    def plot_metrics(self,metrics:list,fig_path=None,show=False,
                     use_notes=False, plot_spec={}):
        """  """
        ps = {"xlabel":"epoch", "ylabel":"", "title":"", "cmap":"Set1",
              "text_size":12, "norm":"linear", "logx":False, "figsize":(16,12),
              "plot_kwargs":{}, "xlim":None, "ylim":None, "line_width":2,
              "facecolor":"white", "legend_cols":1, "fontsize_legend":12,
              "fontsize_title":16, "fontsize_labels":12,}
        line_styles = ("-", ":", "--", "-.")
        ps = {**ps, **plot_spec}
        fig,ax = plt.subplots()
        cmap = plt.cm.get_cmap(ps.get("cmap"), len(self.model_dirs))
        for i,md in enumerate(sorted(self.model_dirs,key=lambda m:m.dir.name)):
            for j,m in enumerate(metrics):
                if m not in md.metric_labels:
                    raise ValueError(f"{md} doesn't support metric {m}")
                label = f"{md.name} - "
                if use_notes:
                    label += md.config.get("notes") + " - "
                label += m
                lineplot = ax.plot(
                        md.get_metric("epoch"),
                        md.get_metric(m),
                        linewidth=ps.get("line_width"),
                        label=label,
                        color=cmap(i),
                        linestyle=line_styles[j%len(line_styles)],
                        **ps.get("plot_kwargs"),
                        )
        legend = ax.legend(
                ncols=ps.get("legend_cols"),
                fontsize=ps.get("fontsize_legend"),
                )
        if ps["logx"]:
            plt.semilogx()
        ax.set_title(ps.get("title"), fontsize=ps.get("fontsize_title"))
        ax.set_xlabel(ps.get("xlabel"), fontsize=ps.get("fontsize_labels"))
        ax.set_ylabel(ps.get("ylabel"), fontsize=ps.get("fontsize_labels"))
        ax.set_facecolor(ps.get("facecolor"))
        if not ps.get("xlim") is None:
            ax.set_xlim(*ps["xlim"])
        if not ps.get("ylim") is None:
            ax.set_ylim(*ps["ylim"])
        if show:
            plt.show()
        if not fig_path is None:
            fig.set_size_inches(*ps.get("figsize"))
            fig.savefig(fig_path.as_posix(), bbox_inches="tight", dpi=80)

