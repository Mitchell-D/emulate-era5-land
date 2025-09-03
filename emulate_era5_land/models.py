import torch
import torch.nn as tnn

class LSTM_S2S(tnn.Module):
    """
    Sequence-to-sequence style LSTM that shares weights between the window and
    horizon sequences, and enables output cycling, accumulation of
    differentiated outputs, as well as

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
            num_hidden_layers=1, normalized_inputs=True,
            normalized_outputs=True, cycle_targets=[], teacher_forcing=False):
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
        """
        super(LSTM_S2S, self).__init__()
        self._norm_in = normalized_inputs
        self._norm_out = normalized_outputs
        self._teacher_forcing = teacher_forcing
        self._nhnodes = num_hidden_feats
        self._nhlayers = num_hidden_layers

        self.feats = {
                "w":window_feats, "h":horizon_feats, "y":target_feats,
                "s":static_feats, "si":static_int_feats,
                }

        ## Identify target features to cycle as inputs at each step, and which
        ## ones will need to be integrated.
        base_cycle_feats = [tfk.split(" ")[-1] for tfk in cycle_feats]
        ## make sure window and horizon features match since weights are shared
        assert tuple(window_feats)==tuple(*horizon_feats, *base_cycle_feats), \
                "Window feats must match and be ordered the same as the "\
                "concatenation of horizon feats with undifferentiated cycled "\
                "feats since model shares weights between window & horizon"\
                f"{horizon_feats=}\n{horizon_feats=}\n{base_cycle_feats}"

        ## for each differentiated cycled feat, get the normalization coeffs
        ## for both the differentiated and integrated components
        dcf_info = [
            (i,norm_coeffs.get(fk,(0,1)),norm_coeffs.get(bt[-1],(0,1)))
            for i,fk in enumerate(cycle_feats)
            if (bt:=fk.split(" "))[0]=="diff"
            ]
        if len(dcf_info) == 0:
            self._diff_cycle_ixs = []
        else:
            self._diff_cycle_ixs,dnorms,inorms = list(zip(*dcf_info))
            ## use the norm coeffs to calculate the values for linearly
            ## rescaling a normalized differential to normalized integrated
            ## coordinates as shown in the class doc string
            self._dcf_norms = np.array([
                    (md/si, sd/si) for ((md,sd),(mi,si)) in zip(dnorms, inorms)
                    ]).T

        ## cycled feat indices wrt model output vector
        self._cycle_ixs = tuple(target_feats.index(fk) for fk in cycle_feats)

        ## Get a dict of all the normalization vectors
        self.norms = {
                k:np.array([norm_coeffs.get(fk, (0,1)) for fk in fl]).T
                for k,fl in self.feats.items()
                }
        ## make a trainable affine embedding matrix for all static int inputs
        self._embed = tnn.Linear(
                sum([len(static_embed_maps[sik]) for sik in static_int_feats]),
                static_int_encoding_size,
                )
        ## declare the lstm
        self._lstm = tnn.LSTM(
                input_size=window_feats,
                hidden_size=num_hidden_feats,
                batch_first=True,
                proj_size=0, ## don't use proj; non-recurrent layer is separate
                **lstm_kwargs,
                )
        ## initialize a non-recurrent affine projection to the output space
        self._proj = tnn.Linear(num_hidden_feats, len(self.feats["y"]))

        return self

    def forward(self, window, horizon, static, static_int,
            target=None, random_init_state=False):
        """
        Run the RNN on a batch of sequence samples.

        :@param window: (B, Sw, Fh)
        """
        if not self._norm_in:
            window = (window-self.norm["w"][0])/self.norm["w"][1]
            horizon = (horizon-self.norm["h"][0])/self.norm["h"][1]
            static = (static-self.norm["s"][0])/self.norm["s"][1]
            if not target is None:
                target = (target-self.norm["y"][0])/self.norm["y"][1]

        si_embed = self._embed(torch.cat(static_int, axis=-1))
        s = torch.cat([static,si_embed], axis=-1)[:,None]
        w = torch.cat([s.expand(-1,window.size(1),-1), window])

        if random_init_state:
            C_init = torch.rand(self._nhlayers, self._nhnodes)
            L_init = torch.rand(self._nhlayers, self._nhnodes)
        else:
            C_init = torch.zeros(self._nhlayers, self._nhnodes)
            L_init = torch.zeros(self._nhlayers, self._nhnodes)

        ## get the hidden state and context from the window encoder
        _,(L,C) = self._lstm(w, (L_init, C_init))

        ## last feats from last window seq; guaranteed to be base cycle feats
        pc_init = w[:,-1,-len(self.feats["c"]):][:,None,:]
        if self._teacher_forcing and self.training:
            ## extract all cycled output features
            tforce = target[...,self._cycle_ixs]
            if len(self._diff_cycle_ixs) != 0:
                ## renorm just the diff'd output feats to integrated coords
                dcf = tforce[...,self._diff_cycle_ixs]
                # from above documentation: dg = dr (u/m) + (c/m)
                dcf = dcf * self._dcf_norms[1] + self._dcf_norms[0]
                ## accumulate differentials and scale to start with init state
                tforce[...,self._diff_cycle_ixs] = \
                        pc_init[...,self._diff_cycle_ixs] \
                        + torch.cumsum(dcf, dim=1)

        P = torch.zeros(horizon.size(0),horizon.size(1),len(self.feats["y"]))
        ## input_seq without the cycled features
        input_seq = torch.cat([s.expand(-1,horizon.size(1),-1), horizon], -1)
        for ix in range(horizon.size(1)):
            ## run all samples/layers of the LSTM for this timestep
            tmpp,(L,C) = self._lstm(
                    torch.cat([input_seq[:,ix,:][:,None,:], pc_init], axis=-1),
                    (L,C)
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
                pc_init = tforce[:,ix+1,:][:,None,:]
            else:
                ## extract all outputs that are to be cycled back in
                cfeats = tmpp[:,None,:][...,self._cycle_ixs]
                if len(self._diff_cycle_ixs) != 0:
                    ## subset to only the differentiated cycled features
                    dcf = cfeats[...,self._diff_cycle_ixs]
                    ## convert norm'd increment to integral coordinates
                    dcf = dcf * self._dcf_norms[1] + self._dcf_norms[0]
                    ## accumulate previous state with updated one
                    cfeats[...,self._diff_cycle_ixs] = \
                            pc_init[...,self._diff_cycle_ixs] + dcf
                ## update new initial state as the calculated cycled feats.
                pc_init = cfeats

        ## if output normalization isn't requested, rescale to data coords
        if not self._norm_out:
            P = P * self.norms["y"][1] + self.norms["y"][0]
        return P
