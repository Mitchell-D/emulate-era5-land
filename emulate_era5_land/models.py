import torch
import torch.nn as tnn
import numpy as np

class AccLSTM(tnn.Module):
    """
    Sequence-to-sequence style LSTM that shares weights between the window and
    horizon sequences, and enables output cycling, accumulation of
    differentiated outputs, as well as teacher forcing from target values.

    The model is structured in a window-horizon configuration where the window
    includes known output states that aren't available in the horizon, however
    since weights are shared between the window and horizon, all these states
    or the differentiated version of the states must be the target variables
    for model prediction.

    If a differentiated target variable is cycled by accumulated
    integration (the only option for cycling), its normalization must be
    converted to the normalized units of the accumulated variable.

           g = (x - b) / m              ## norm'd magnitude of state x
          dr = (dx - c) / u             ## norm'd increment change in state x
          x' = x + dx                   ## time step change in x
    g' m + b = (g m + b) + (dr u + c)   ## expand to normalized data variables
          g' = g + (dr u + c) / m       ## solve for normalized next-step
          g' = g + dr (u/m) + (c/m)     ## next-step given norm'd increment

    """
    def __init__(self, window_feats, horizon_feats, target_feats,
            static_feats, static_int_feats, static_embed_maps={},
            static_int_encoding_size=6, norm_coeffs={}, num_hidden_feats=32,
            num_hidden_layers=1, lstm_kwargs={}, normalized_inputs=True,
            normalized_outputs=True, cycle_targets=[], teacher_forcing=False,
            sample_retain_frequency=None):
        """

        :@param *_feats: number of feature dimensions in terminal layers
            associated with each input or output category
        :@param static_embed_maps: Dict mapping static feature names to vector
            sizes of their one-hot static integer embeddings
        :@param static_int_encoding_size: Static integer one-hot encoded inputs
            are concatenated and encoded to a float vector of this size.
        :@param norm_coeffs: Dict mapping feature names to 2-tuples
            (mean, standard deviation) used to normalize or de-normalize the
            data coordinates.
        :@param num_hidden_feats: Dimensionality of hidden layers.
        :@param num_hidden_layers: Integer number of hidden LSTM layers to use.
        :@param normalized_inputs: If True, the inputs will not be normalized
            prior to executing the model (expected to already be norm'd).
        :@param normalized_outputs: If True, the outputs will remain
            normalized, otherwise they will be converted back to data coords.
        :@param cycle_targets: List a subset of the target variables to cycle
            back into the inputs. For now, any differentiated (increment)
            outputs will be integrated to their accumulated state values (*)
            before be concatenated to the input sequence. Even so, specify
            the cycled feat name like "diff vsm-10" to prevent ambiguity.
        :@param teacher_forcing: Cycle in the 'true' target states as inputs
            to horizon prediction steps during training for rather than
            self-cycling previous outputs, which might better emphasize the
            relationship between current states and subsequent transitions.
            This only applies to feat specified in cycle_targets.
        :@param lstm_kwargs: additional arguments to lstm available from:
            https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        :@param sample_retain_frequency: If an integer not None, the AccLSTM
            will store as an attribute the first sample and model outputs once
            every provided number of batches.
        """
        super(AccLSTM, self).__init__()
        self._norm_in = normalized_inputs
        self._norm_out = normalized_outputs
        self._teacher_forcing = teacher_forcing
        self._nhnodes = num_hidden_feats
        self._nhlayers = num_hidden_layers

        self.feats = {
                "w":window_feats, "h":horizon_feats, "y":target_feats,
                "s":static_feats, "si":static_int_feats, "c":cycle_targets,
                }

        ## Identify target features to cycle as inputs at each step, and which
        ## ones will need to be integrated.
        base_cycle_feats = [tfk.split(" ")[-1] for tfk in cycle_targets]
        ## make sure window and horizon features match since weights are shared
        assert tuple(window_feats)==(*horizon_feats, *base_cycle_feats), \
                "Window feats must match and be ordered the same as the "\
                "concatenation of horizon feats with undifferentiated cycled "\
                "feats since model shares weights between window & horizon\n"\
                f"{window_feats=}\n{horizon_feats=}\n{base_cycle_feats=}"

        ## for each differentiated cycled feat, get the normalization coeffs
        ## for both the differentiated and integrated components
        dcf_info = [
            (i,norm_coeffs.get(fk,(0,1)),norm_coeffs.get(bt[-1],(0,1)))
            for i,fk in enumerate(cycle_targets)
            if (bt:=fk.split(" "))[0]=="diff"
            ]
        if len(dcf_info) == 0:
            self._diff_cycle_ixs = []
            self._dcf_norms = None
        else:
            self._diff_cycle_ixs,dnorms,inorms = list(zip(*dcf_info))
            ## use the norm coeffs to calculate the values for linearly
            ## rescaling a normalized differential to normalized integrated
            ## coordinates as shown in the class doc string
            self._dcf_norms = torch.Tensor(np.array([
                    (md/si, sd/si) for ((md,sd),(mi,si)) in zip(dnorms, inorms)
                    ]).T)
        print(f"{self._dcf_norms = }")

        ## cycled feat indices wrt model output vector
        self._cycle_ixs = tuple(target_feats.index(fk) for fk in cycle_targets)

        ## Get a dict of all the normalization vectors
        self.norms = {
                k:torch.Tensor([norm_coeffs.get(fk, (0,1)) for fk in fl]).T
                for k,fl in self.feats.items()
                }
        ## make a trainable affine embedding matrix for all static int inputs
        self._embed = tnn.Linear(
                sum([len(static_embed_maps[sik]) for sik in static_int_feats]),
                static_int_encoding_size,
                )
        ## declare the lstm
        input_size = len(self.feats["w"]) + len(self.feats["s"]) \
                + static_int_encoding_size
        self._lstm = tnn.LSTM(
                input_size=input_size,
                hidden_size=self._nhnodes,
                num_layers=self._nhlayers,
                batch_first=True,
                proj_size=0, ## don't use proj; non-recurrent layer is separate
                **lstm_kwargs,
                )
        ## initialize a non-recurrent affine projection to the output space
        self._proj = tnn.Linear(self._nhnodes, len(self.feats["y"]))

        self._srf = sample_retain_frequency
        self._nb = 0 ## number of batches
        self.samples = []

    def forward(self, window, horizon, static, static_int,
            target=None, random_init_state=False, device=None):
        """
        Run the RNN on a batch of sequence samples.

        :@param window: (B, Sw, Fh)
        """
        _dt = window.dtype
        ## if normalized inputs not expected, normalize them now.
        if not self._norm_in:
            window = (window-self.norm["w"][0])/self.norm["w"][1]
            horizon = (horizon-self.norm["h"][0])/self.norm["h"][1]
            static = (static-self.norm["s"][0])/self.norm["s"][1]
            if not target is None:
                target = (target-self.norm["y"][0])/self.norm["y"][1]

        window = window.to(_dt).to(device)
        horizon = horizon.to(_dt).to(device)
        static = static.to(_dt).to(device)
        static_int = torch.cat(static_int, axis=-1).to(device)

        ## create static vectors, embed them, and concatenate with window input
        si_embed = self._embed(static_int)
        s = torch.cat([static,si_embed], axis=-1)[:,None]
        w = torch.cat([s.expand(-1,window.size(1),-1), window], axis=-1)

        if not self._dcf_norms is None and self._dcf_norms.device != device:
            self._dcf_norms = self._dcf_norms.to(device)

        if random_init_state:
            C_init = torch.rand(
                    self._nhlayers, window.shape[0], self._nhnodes,
                    device=device, dtype=_dt)
            L_init = torch.rand(
                    self._nhlayers, window.shape[0], self._nhnodes,
                    device=device, dtype=_dt)
        else:
            C_init = torch.zeros(
                    self._nhlayers, window.shape[0], self._nhnodes,
                    device=device, dtype=_dt)
            L_init = torch.zeros(
                    self._nhlayers, window.shape[0], self._nhnodes,
                    device=device, dtype=_dt)

        ## get the hidden state and context from the window encoder
        _,(L,C) = self._lstm(w, (L_init, C_init))
        ## only persist the last hidden state in the sequence
        L = L[:,-1,:]
        C = C[:,-1,:]

        ## last feats from last window seq; guaranteed to be base cycle feats
        pc_init = torch.full(
                (horizon.size(0), horizon.size(1), len(self.feats["c"])),
                float("nan"),
                device=device,
                dtype=_dt,
                )
        pc_init[:,0,:] = w[:,-1,-len(self.feats["c"]):]
        if self._teacher_forcing and self.training:
            ## extract all cycled output features
            tforce = target[...,self._cycle_ixs]
            if len(self._diff_cycle_ixs) != 0:
                ## renorm just the diff'd output feats to integrated coords
                dcf = tforce[...,self._diff_cycle_ixs]
                ## from above documentation: dg = dr (u/m) + (c/m)
                dcf = dcf * self._dcf_norms[1] + self._dcf_norms[0]
                ## accumulate differentials and scale to start with init state
                tforce[...,self._diff_cycle_ixs] = \
                        pc_init[:,0,self._diff_cycle_ixs][:,None,:] \
                        + torch.cumsum(dcf, dim=1).to(_dt)

        P = torch.zeros(horizon.size(0),horizon.size(1),len(self.feats["y"]))
        P = P.to(device)
        ## input_seq without the cycled features
        input_seq = torch.cat([s.expand(-1,horizon.size(1),-1), horizon], -1)
        for ix in range(horizon.size(1)):
            ## run all samples/layers of the LSTM for this timestep
            tmpp,(L,C) = self._lstm(
                    torch.cat([input_seq[:,ix,:], pc_init[:,ix,:]], axis=-1),
                    (L.contiguous(),C.contiguous())
                    )
            ## project the lstm output to the norm'd prediction target values
            tmpp = self._proj(tmpp)
            ## load the predictions into the output tensor for this timestep
            P[:,ix,:] = tmpp

            ## don't try to calculate the cycled inputs in the last loop
            if ix==horizon.size(1)-1:
                break
            ## calculate the next cycled input
            if self._teacher_forcing and self.training:
                ## if teacher forcing used, grab the next pre-calculated step
                pc_init[:,ix+1] = tforce[:,ix+1]
            else:
                ## extract all outputs that are to be cycled back in
                cfeats = tmpp[...,self._cycle_ixs]
                if len(self._diff_cycle_ixs) != 0:
                    ## subset to only the differentiated cycled features
                    dcf = cfeats[...,self._diff_cycle_ixs]
                    ## convert norm'd increment to integral coordinates
                    dcf = dcf * self._dcf_norms[1] + self._dcf_norms[0]
                    ## accumulate previous state with updated one
                    cfeats[...,self._diff_cycle_ixs] = \
                            pc_init[:,ix,self._diff_cycle_ixs] + dcf
                ## update new initial state as the calculated cycled feats.
                pc_init[:,ix+1] = cfeats

        ## if output normalization isn't requested, rescale to data coords
        if not self._norm_out:
            P = P*self.norms["y"][1].to(device)+self.norms["y"][0].to(device)

        self._nb += 1
        if not self._srf is None and self._nb % self._srf == 0:
            ## Note that pc_init is the integrated target differentials if
            ## teacher_forcing, else the integrated model outputs.
            self.samples.append((
                    (window[0], horizon[0], static[0], static_int[0]),
                    (target[0], P[0], pc_init[0])
                    ))
        return P

model_options = {
    "acclstm":AccLSTM
    }

def get_model_from_config(config):
    """
    Initialize and return an instance of the configured model
    """
    assert config["model"]["type"] in model_options.keys(),model_options.keys()
    ## initialize the model, providing default args that would be redundant
    return model_options[config["model"]["type"]](**{
        ## defaults for feature sizes to prevent repetition
        "window_feats":config["feats"]["window_feats"],
        "horizon_feats":config["feats"]["horizon_feats"],
        "target_feats":config["feats"]["target_feats"],
        "static_feats":config["feats"]["static_feats"],
        "static_int_feats":config["feats"]["static_int_feats"],
        "static_embed_maps":config["feats"]["static_embed_maps"],
        "norm_coeffs":config["feats"]["norm_coeffs"],
        ## user-defined other model parameters
        **config["model"]["args"],
        })

